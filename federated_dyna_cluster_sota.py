import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from focal_loss_balanced_sampling import FocalLoss, DynaClusterTrainer  # From previous response
from attention_drl_policy import AttentionDRLPolicy  # From previous response
from diffusers import UNet2DModel  # Diffusion model
from transformers import AutoModel, CLIPProcessor  # For DINOv2/OpenCLIP

class DiffusionClusterInitializer:
    def __init__(self, feature_dim=384, num_clusters=10):
        self.model = UNet2DModel(
            sample_size=32,
            in_channels=feature_dim,
            out_channels=num_clusters,
            block_out_channels=(64, 128, 256),
            cross_attention_dim=128
        ).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.num_clusters = num_clusters
    
    def train(self, features, epochs=5):
        features = torch.tensor(features, dtype=torch.float32).cuda()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            noise = torch.randn_like(features)
            t = torch.randint(0, 1000, (features.shape[0],)).cuda()
            noisy_features = noise * (1 - t / 1000)[:, None] + features * (t / 1000)[:, None]
            pred = self.model(noisy_features, t, return_dict=False)[0]
            loss = nn.MSELoss()(pred, features)
            loss.backward()
            self.optimizer.step()
    
    def initialize_clusters(self, features):
        features = torch.tensor(features, dtype=torch.float32).cuda()
        with torch.no_grad():
            pred = self.model(features, torch.zeros(features.shape[0]).cuda(), return_dict=False)[0]
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        labels = kmeans.fit_predict(pred.cpu().numpy())
        centroids = kmeans.cluster_centers_
        variances = np.var(pred.cpu().numpy(), axis=0)
        return {'centroids': centroids, 'variances': variances, 'labels': labels}

class FederatedDynaClusterSOTA:
    def __init__(self, num_clients, num_clusters, feature_extractor='dinov2', labeled_data_list=None, alpha=0.25, gamma=2.0):
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        self.feature_extractor_type = feature_extractor
        if feature_extractor == 'dinov2':
            self.feature_model = AutoModel.from_pretrained('facebook/dinov2-large').cuda().eval()
        else:
            self.feature_model = AutoModel.from_pretrained('laion/CLIP-ViT-L-336px').cuda().eval()
        self.clients = [
            DynaClusterTrainer(self.feature_model, num_clusters, labeled_data_list[k], alpha, gamma)
            for k in range(num_clients)
        ]
        self.diffusion_initializers = [DiffusionClusterInitializer(feature_dim=384, num_clusters=num_clusters) for _ in range(num_clients)]
        self.global_micro_clusters = None
        self.drl_policy = AttentionDRLPolicy(input_dim=384, hidden_dim=256, num_clusters=num_clusters).cuda()
    
    def extract_features(self, images):
        with torch.no_grad():
            if self.feature_extractor_type == 'dinov2':
                features = self.feature_model(images).last_hidden_state[:, 1:].mean(dim=1)  # Patch-level features
            else:
                features = self.feature_model.encode_image(images).float()
        return features
    
    def local_update(self, client_id, local_data, optimizer, num_epochs=5):
        trainer = self.clients[client_id]
        diffusion_initializer = self.diffusion_initializers[client_id]
        
        # Extract features for diffusion clustering
        features = []
        for batch in local_data:
            images, _ = batch
            images = images.cuda()
            features.append(self.extract_features(images).cpu().numpy())
        features = np.concatenate(features, axis=0)
        
        # Train diffusion model and initialize micro-clusters
        diffusion_initializer.train(features)
        micro_clusters = diffusion_initializer.initialize_clusters(features)
        
        # Local training
        for _ in range(num_epochs):
            batch = trainer.balanced_sampling(batch_size=32)
            total_loss, focal_loss, recon_loss = trainer.train_step(batch, optimizer)
            
            # Update micro-clusters with DRL
            images = torch.stack([item[0] for item in batch]).cuda()
            cluster_probs, _ = trainer.model(images)
            cluster_features = trainer.model.get_cluster_features()
            image_embedding = self.extract_features(images[0:1])
            weights, reward = trainer.update_mc_weights(cluster_features, image_embedding, self.drl_policy)
        
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
        self.drl_policy.load_state_dict(local_models[0])  # Simplified
    
    def train_federated(self, local_data_list, num_rounds=50):
        optimizer = torch.optim.Adam(self.feature_model.parameters(), lr=0.001)
        for round in range(num_rounds):
            local_models = []
            local_micro_clusters = []
            client_sample_counts = []
            total_loss = 0
            
            for client_id in range(self.num_clients):
                local_model, micro_clusters, loss, reward = self.local_update(
                    client_id, local_data_list[client_id], optimizer
                )
                local_models.append(local_model)
                local_micro_clusters.append(micro_clusters)
                client_sample_counts.append(len(local_data_list[client_id]))
                total_loss += loss
            
            self.global_aggregation(local_models, local_micro_clusters, client_sample_counts)
            print(f"Round {round+1}: Average Loss = {total_loss / self.num_clients:.4f}")
    
    def evaluate(self, test_data):
        self.feature_model.eval()
        # Placeholder for ACC, NMI, ARI, F-score, LCP, DAS, VIS
        pass

def run_federated_dyna_cluster_sota(num_clients, local_data_list, feature_extractor='dinov2', num_rounds=50):
    fed_dyna_cluster = FederatedDynaClusterSOTA(num_clients, num_clusters=10, feature_extractor=feature_extractor, labeled_data_list=local_data_list)
    fed_dyna_cluster.train_federated(local_data_list, num_rounds)