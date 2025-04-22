import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from focal_loss_balanced_sampling import FocalLoss, DynaClusterTrainer  # From previous response
from attention_drl_policy import AttentionDRLPolicy  # From previous response

class FederatedDynaCluster:
    def __init__(self, num_clients, num_clusters, model, labeled_data_list, alpha=0.25, gamma=2.0):
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        self.global_model = model
        self.clients = [
            DynaClusterTrainer(model, num_clusters, labeled_data_list[k], alpha, gamma)
            for k in range(num_clients)
        ]
        self.global_micro_clusters = None
        self.drl_policy = AttentionDRLPolicy(input_dim=512, hidden_dim=256, num_clusters=num_clusters)
    
    def local_update(self, client_id, local_data, optimizer, num_epochs=5):
        trainer = self.clients[client_id]
        for _ in range(num_epochs):
            batch = trainer.balanced_sampling(batch_size=32)
            total_loss, focal_loss, recon_loss = trainer.train_step(batch, optimizer)
            
            # Update local micro-clusters
            images = torch.stack([item[0] for item in batch]).cuda()
            cluster_probs, _ = trainer.model(images)
            cluster_features = trainer.model.get_cluster_features()  # Assume method exists
            image_embedding = trainer.model.get_image_embedding(images[0])  # Example embedding
            weights, reward = trainer.update_mc_weights(cluster_features, image_embedding, self.drl_policy)
        
        return trainer.model.state_dict(), trainer.get_micro_clusters(), total_loss, reward
    
    def aggregate_micro_clusters(self, local_micro_clusters, client_sample_counts):
        # Aggregate centroids and variances using k-means
        all_centroids = np.concatenate([mc['centroids'] for mc in local_micro_clusters], axis=0)
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        kmeans.fit(all_centroids)
        global_centroids = kmeans.cluster_centers_
        
        # Aggregate variances (weighted by sample counts)
        global_variances = np.zeros_like(global_centroids)
        total_samples = sum(client_sample_counts)
        for mc, count in zip(local_micro_clusters, client_sample_counts):
            for i, centroid in enumerate(mc['centroids']):
                closest_global = np.argmin(np.linalg.norm(global_centroids - centroid, axis=1))
                global_variances[closest_global] += mc['variances'][i] * count / total_samples
        
        # Compute global weights
        global_weights = np.array([count / total_samples for count in client_sample_counts])
        return {'centroids': global_centroids, 'variances': global_variances, 'weights': global_weights}
    
    def global_aggregation(self, local_models, local_micro_clusters, client_sample_counts):
        # FedAvg for model parameters
        global_state_dict = self.global_model.state_dict()
        total_samples = sum(client_sample_counts)
        for key in global_state_dict:
            global_state_dict[key] = sum(
                model[key] * (count / total_samples)
                for model, count in zip(local_models, client_sample_counts)
            )
        self.global_model.load_state_dict(global_state_dict)
        
        # Aggregate micro-clusters
        self.global_micro_clusters = self.aggregate_micro_clusters(local_micro_clusters, client_sample_counts)
        
        # Update DRL policy (simplified)
        self.drl_policy.load_state_dict(local_models[0])  # Use first client's policy for simplicity
    
    def train_federated(self, local_data_list, num_rounds=50):
        optimizer = torch.optim.Adam(self.global_model.parameters(), lr=0.001)
        for round in range(num_rounds):
            local_models = []
            local_micro_clusters = []
            client_sample_counts = []
            total_loss = 0
            
            # Local updates
            for client_id in range(self.num_clients):
                local_model, micro_clusters, loss, reward = self.local_update(
                    client_id, local_data_list[client_id], optimizer
                )
                local_models.append(local_model)
                local_micro_clusters.append(micro_clusters)
                client_sample_counts.append(len(local_data_list[client_id]))
                total_loss += loss
            
            # Global aggregation
            self.global_aggregation(local_models, local_micro_clusters, client_sample_counts)
            
            print(f"Round {round+1}: Average Loss = {total_loss / self.num_clients:.4f}")
    
    def evaluate(self, test_data):
        self.global_model.eval()
        # Implement evaluation logic (e.g., compute ACC, NMI, ARI, F-score)
        pass

# Example usage
def run_federated_dyna_cluster(num_clients, local_data_list, model, num_rounds=50):
    fed_dyna_cluster = FederatedDynaCluster(num_clients, num_clusters=10, model=model, labeled_data_list=local_data_list)
    fed_dyna_cluster.train_federated(local_data_list, num_rounds)