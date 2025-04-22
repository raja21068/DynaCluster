import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # inputs: [batch_size, num_clusters], targets: [batch_size]
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class DynaClusterTrainer:
    def __init__(self, model, num_clusters, labeled_data, alpha=0.25, gamma=2.0):
        self.model = model
        self.num_clusters = num_clusters
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.reconstruction_loss = nn.MSELoss()
        self.labeled_data = labeled_data
        self.cluster_weights = self.compute_cluster_weights()
    
    def compute_cluster_weights(self):
        # Compute inverse frequency weights for balanced sampling
        labels = [label for _, label in self.labeled_data]
        label_counts = Counter(labels)
        total_samples = len(labels)
        weights = {label: total_samples / (len(label_counts) * count) for label, count in label_counts.items()}
        return weights
    
    def balanced_sampling(self, batch_size):
        # Sample images with balanced micro-cluster representation
        labels = [label for _, label in self.labeled_data]
        weights = [self.cluster_weights[label] for label in labels]
        weights = np.array(weights) / np.sum(weights)
        indices = np.random.choice(len(self.labeled_data), size=batch_size, p=weights)
        batch = [self.labeled_data[i] for i in indices]
        return batch
    
    def train_step(self, batch, optimizer):
        self.model.train()
        optimizer.zero_grad()
        
        # Extract images and labels from batch
        images = torch.stack([item[0] for item in batch]).cuda()
        targets = torch.tensor([item[1] for item in batch], dtype=torch.long).cuda()
        
        # Forward pass
        cluster_probs, reconstructed = self.model(images)  # Assume model outputs probs and reconstruction
        
        # Compute losses
        focal_loss = self.focal_loss(cluster_probs, targets)
        recon_loss = self.reconstruction_loss(reconstructed, images)
        total_loss = focal_loss + recon_loss  # Add other losses (e.g., contrastive, meta) as needed
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item(), focal_loss.item(), recon_loss.item()
    
    def update_mc_weights(self, cluster_features, image_embedding, policy_network):
        # Balanced sampling for DRL policy updates
        batch = self.balanced_sampling(batch_size=32)
        images = torch.stack([item[0] for item in batch]).cuda()
        cluster_probs = self.model(images)[0]  # Get cluster probabilities
        
        # Incorporate focal loss into reward
        targets = torch.tensor([item[1] for item in batch], dtype=torch.long).cuda()
        focal_loss = self.focal_loss(cluster_probs, targets)
        reward = -focal_loss.item()  # Higher reward for lower focal loss
        
        # Update weights using attention-enhanced DRL policy (from previous response)
        action_probs, value, attn_weights, temporal_weights = policy_network(
            cluster_features, image_embedding, temporal_features=None
        )
        weights = action_probs.detach()
        return weights, reward

# Example usage
def train_dyna_cluster(model, labeled_data, num_epochs, batch_size=32):
    trainer = DynaClusterTrainer(model, num_clusters=10, labeled_data=labeled_data)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        batch = trainer.balanced_sampling(batch_size)
        total_loss, focal_loss, recon_loss = trainer.train_step(batch, optimizer)
        print(f"Epoch {epoch+1}: Total Loss = {total_loss:.4f}, Focal Loss = {focal_loss:.4f}, Recon Loss = {recon_loss:.4f}")