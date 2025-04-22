import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from focal_loss_balanced_sampling import FocalLoss, DynaClusterTrainer
from attention_drl_policy import AttentionDRLPolicy
from diffusers import UNet2DModel
from transformers import AutoModel, CLIPProcessor
import torchvision.transforms as T

class TemporalAttention(nn.Module):
    def __init__(self, feature_dim=384, num_heads=4):
        super(TemporalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
    
    def forward(self, frame_features):
        # frame_features: [seq_len, batch_size, feature_dim]
        attn_output, _ = self.attention(frame_features, frame_features, frame_features)
        return attn_output.mean(dim=0)  # Aggregate over sequence

class FederatedDynaClusterVideoMultiModal:
    def __init__(self, num_clients, num_clusters, feature_extractor='dinov2', labeled_data_list=None, alpha=0.25, gamma=2.0):
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        self.feature_extractor_type = feature_extractor
        if feature_extractor == 'dinov2':
            self.feature_model = AutoModel.from_pretrained('facebook/dinov2-large').cuda().eval()
        else:
            self.feature_model = AutoModel.from_pretrained('laion/CLIP-ViT-L-336px').cuda().eval()
            self.clip_processor = CLIPProcessor.from_pretrained('laion/CLIP-ViT-L-336px')
        self.temporal_attention = TemporalAttention(feature_dim=384).cuda()
        self.clients = [
            DynaClusterTrainer(self.feature_model, num_clusters, labeled_data_list[k], alpha, gamma)
            for k in range(num_clients)
        ]
        self.diffusion_initializers = [UNet2DModel(
            sample_size=32, in_channels=384, out_channels=num_clusters, block_out_channels=(64, 128, 256), cross_attention_dim=128
        ).cuda() for _ in range(num_clients)]
        self.global_micro_clusters = None
        self.drl_policy = AttentionDRLPolicy(input_dim=384, hidden_dim=256, num_clusters=num_clusters).cuda()
    
    def extract_video_features(self, frames):
        with torch.no_grad():
            frame_features = []
            for frame in frames:
                features = self.feature_model(frame).last_hidden_state[:, 1:].mean(dim=1)
                frame_features.append(features)
            frame_features = torch.stack(frame_features)  # [seq_len, batch_size, feature_dim]
            video_features = self.temporal_attention(frame_features)  # [batch_size, feature_dim]
        return video_features
    
    def extract_text_image_features(self, images, texts):
        with torch.no_grad():
            inputs = self.clip_processor(text=texts, images=images, return_tensors="pt", padding=True).to('cuda')
            outputs = self.feature_model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            joint_features = (image_features + text_features) / 2  # Simple averaging
        return joint_features
    
    def local_update(self, client_id, local_data, optimizer, num_epochs=5, data_type='video'):
        trainer = self.clients[client_id]
        diffusion_initializer = self.diffusion_initializers[client_id]
        diffusion_optimizer = torch.optim.Adam(diffusion_initializer.parameters(), lr=1e-4)
        
        features = []
        for batch in local_data:
            if data_type == 'video':
                frames, _ = batch  # frames: [batch_size, seq_len, C, H, W]
                frames = frames.permute(1, 0, 2, 3, 4)  # [seq_len, batch_size, C, H, W]
                batch_features = self.extract_video_features(frames.cuda())
            else:  # image-text
                images, texts = batch
                batch_features = self.extract_text_image_features(images.cuda(), texts)
            features.append(batch_features.cpu().numpy())
        features = np.concatenate(features, axis=0)
        
        # Train diffusion model
        features_tensor = torch.tensor(features, dtype=torch.float32).cuda()
        for _ in range(5):
            diffusion_optimizer.zero_grad()
            noise = torch.randn_like(features_tensor)
            t = torch.randint(0, 1000, (features_tensor.shape[0],)).cuda()
            noisy_features = noise * (1 - t / 1000)[:, None] + features_tensor * (t / 1000)[:, None]
            pred = diffusion_initializer(noisy_features, t, return_dict=False)[0]
            loss = nn.MSELoss()(pred, features_tensor)
            loss.backward()
            diffusion_optimizer.step()
        
        # Initialize micro-clusters
        with torch.no_grad():
            pred = diffusion_initializer(features_tensor, torch.zeros(features_tensor.shape[0]).cuda(), return_dict=False)[0]
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        labels = kmeans.fit_predict(pred.cpu().numpy())
        micro_clusters = {
            'centroids': kmeans.cluster_centers_,
            'variances': np.var(pred.cpu().numpy(), axis=0),
            'labels': labels
        }
        
        # Local training
        for _ in range(num_epochs):
            batch = trainer.balanced_sampling(batch_size=32)
            images = torch.stack([item[0] for item in batch]).cuda()
            total_loss, focal_loss, recon_loss = trainer.train_step(batch, optimizer)
            
            cluster_probs, _ = trainer.model(images)
            cluster_features = trainer.model.get_cluster_features()
            image_embedding = self.extract_video_features(images[:, None, :, :, :]) if data_type == 'video' else self.extract_text_image_features(images, [item[1] for item in batch])
            weights, reward = trainer.update_mc_weights(cluster_features, image_embedding[0:1], self.drl_policy)
        
        return trainer.model.state_dict(), micro_clusters, total_loss, reward
    
    def aggregate_micro_clusters(self, local_micro_clusters, client_sample_counts):
        all_centroids = np.concatenate([mc['centroids'] for mc in local_micro_clusters], axis=0)
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        kmeans.fit(all_centroids)
        global_centroids = kmeans.cluster_centers_
        
        global_variances = np.zeros_like(global_centroids)
        total_samples = sum(client_sample_counts)
        for mc, count in zip(local_micro_clusters, client_sample_counts):
            for i, centroid in enumerate(mc['centroids']):
                closest_global = np.argmin(np.linalg.norm(global_centroids - centroid, axis=1))
                global_variances[closest_global] += mc['variances'][i] * count / total_samples
        
        global_weights = np.array([count / total_samples for count in client_sample_counts])
        return {'centroids': global_centroids, 'variances': global_variances, 'weights': global_weights}
    
    def global_aggregation(self, local_models, local_micro_clusters, client_sample_counts):
        global_state_dict = self.feature_model.state_dict()
        total_samples = sum(client_sample_counts)
        for key in global_state_dict:
            global_state_dict[key] = sum(
                model[key] * (count / total_samples)
                for model, count in zip(local_models, client_sample_counts)
            )
        self.feature_model.load_state_dict(global_state_dict)
        
        self.global_micro_clusters = self.aggregate_micro_clusters(local_micro_clusters, client_sample_counts)
        self.drl_policy.load_state_dict(local_models[0])
    
    def train_federated(self, local_data_list, num_rounds=50, data_type='video'):
        optimizer = torch.optim.Adam(self.feature_model.parameters(), lr=0.001)
        for round in range(num_rounds):
            local_models = []
            local_micro_clusters = []
            client_sample_counts = []
            total_loss = 0
            
            for client_id in range(self.num_clients):
                local_model, micro_clusters, loss, reward = self.local_update(
                    client_id, local_data_list[client_id], optimizer, data_type=data_type
                )
                local_models.append(local_model)
                local_micro_clusters.append(micro_clusters)
                client_sample_counts.append(len(local_data_list[client_id]))
                total_loss += loss
            
            self.global_aggregation(local_models, local_micro_clusters, client_sample_counts)
            print(f"Round {round+1}: Average Loss = {total_loss / self.num_clients:.4f}")
    
    def evaluate(self, test_data, data_type='video'):
        self.feature_model.eval()
        # Placeholder for metrics
        pass

def run_federated_dyna_cluster_video_multimodal(num_clients, local_data_list, feature_extractor='dinov2', num_rounds=50, data_type='video'):
    fed_dyna_cluster = FederatedDynaClusterVideoMultiModal(num_clients, num_clusters=10, feature_extractor=feature_extractor, labeled_data_list=local_data_list)
    fed_dyna_cluster.train_federated(local_data_list, num_rounds, data_type)