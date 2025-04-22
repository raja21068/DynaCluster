import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        batch_size = Q.size(0)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return context, attn

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        context, attn = self.scaled_dot_product_attention(Q, K, V)
        output = self.W_o(context)
        return output, attn

class AttentionDRLPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_clusters, num_heads=4):
        super(AttentionDRLPolicy, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        self.attention = MultiHeadAttention(input_dim, num_heads)
        self.temporal_attention = MultiHeadAttention(input_dim, num_heads)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, num_clusters)  # Output weight updates
        self.critic = nn.Linear(hidden_dim, 1)  # Output value estimate

    def forward(self, cluster_features, image_embedding, temporal_features):
        # cluster_features: [batch_size, num_clusters, input_dim]
        # image_embedding: [batch_size, input_dim]
        # temporal_features: [batch_size, window_size, input_dim]
        
        # Apply self-attention to cluster features
        cluster_attn, attn_weights = self.attention(cluster_features)
        cluster_attn = cluster_attn + cluster_features  # Residual connection
        
        # Incorporate image embedding
        combined = cluster_attn + image_embedding.unsqueeze(1)  # Broadcast embedding
        
        # Apply temporal attention
        temporal_attn, temporal_weights = self.temporal_attention(temporal_features)
        temporal_attn = temporal_attn[:, -1, :]  # Focus on most recent time step
        combined = combined + temporal_attn.unsqueeze(1)
        
        # Feed through policy network
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        
        # Actor: Output weight updates for clusters
        action_logits = self.actor(x).squeeze(-1)  # [batch_size, num_clusters]
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic: Output value estimate
        value = self.critic(x).squeeze(-1)  # [batch_size]
        
        return action_probs, value, attn_weights, temporal_weights

# Example usage in DynaCluster
def update_mc_weights(cluster_features, image_embedding, temporal_features, policy_network):
    action_probs, value, attn_weights, temporal_weights = policy_network(
        cluster_features, image_embedding, temporal_features
    )
    # Sample action (weight updates) using PPO
    action_dist = torch.distributions.Categorical(action_probs)
    action = action_dist.sample()
    log_prob = action_dist.log_prob(action)
    
    # Update micro-cluster weights (simplified)
    weights = action_probs.detach()
    return weights, log_prob, value, attn_weights, temporal_weights