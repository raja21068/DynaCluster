import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score
from collections import deque
import random
import pandas as pd
import time
import psutil

# Define necessary components

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, z):
        x_recon = self.decoder(z)
        return x_recon

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

def cauchy_similarity(z, c, kappa):
    return 1 / (np.pi * kappa * (1 + ((z - c) ** 2 / kappa ** 2)))

def decision_probability(z, c, kappa):
    s_ij = cauchy_similarity(z, c, kappa)
    return 1 / (1 + torch.exp(-s_ij))

def compute_reward(a_ij, y_ij, v):
    if a_ij == 1 and y_ij == 1:
        return v
    elif a_ij == 1 and y_ij == 0:
        return -v
    else:
        return 0

def print_training_status(epoch, epochs, loss):
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}')

class MicroCluster:
    def __init__(self, center, points):
        self.center = center
        self.points = points

def create_micro_clusters(data, num_micro_clusters):
    kmeans = KMeans(n_clusters=num_micro_clusters, random_state=0).fit(data)
    micro_clusters = [MicroCluster(center, []) for center in kmeans.cluster_centers_]
    for i, label in enumerate(kmeans.labels_):
        micro_clusters[label].points.append(data[i])
    return micro_clusters

def update_mc_weights(agent, replay_memory, batch_size, gamma):
    if len(replay_memory) > batch_size:
        minibatch = random.sample(replay_memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward
            if next_state is not None:
                target = reward + gamma * torch.max(agent(next_state))
            target_f = agent(state)
            target_f[action] = target
            agent.zero_grad()
            loss = nn.MSELoss()(agent(state), target_f)
            loss.backward()
            agent.optimizer.step()

def train_drc(data, labels, K, epochs, lr, gamma, v, num_micro_clusters, batch_size, epsilon):
    input_dim = data.shape[1]
    latent_dim = 10  # Latent dimension size for autoencoder
    kappa = 1.0  # Parameter for cauchy similarity

    model = Autoencoder(input_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Initialize micro-clusters with initial data
    micro_clusters = create_micro_clusters(data, num_micro_clusters)
    micro_cluster_centers = torch.tensor([mc.center for mc in micro_clusters], dtype=torch.float32, requires_grad=True)
    prototypes = torch.randn(K, latent_dim, requires_grad=True)
    prototypes_optimizer = optim.Adam([prototypes], lr=lr)
    micro_cluster_optimizer = optim.Adam([micro_cluster_centers], lr=lr)

    replay_memory = deque(maxlen=2000)
    agent = nn.Sequential(
        nn.Linear(latent_dim, 128),
        nn.ReLU(),
        nn.Linear(128, K)
    )
    agent.optimizer = optim.Adam(agent.parameters(), lr=lr)

    start_time = time.time()
    for epoch in range(epochs):
        total_loss = 0
        
        for x in data:
            x = torch.tensor(x, dtype=torch.float32)
            x_recon, z = model(x)

            # Initialize state and perform actions
            state = z
            for t in range(1):  # Single-step transition for simplicity
                # ε-greedy strategy
                if random.random() < epsilon:
                    action = random.choice(range(K))
                else:
                    action = torch.argmax(agent(state)).item()

                # Transition to next state
                next_state = state + torch.randn(state.size())  # Simulate state transition
                reward = compute_reward(action, action, v)

                # Store transition in replay memory
                replay_memory.append((state, action, reward, next_state))
                state = next_state

                # Perform training update
                if len(replay_memory) > batch_size:
                    update_mc_weights(agent, replay_memory, batch_size, gamma)

                # Compute loss for reconstruction and clustering
                p_ij = torch.zeros(K)
                for j in range(K):
                    p_ij[j] = decision_probability(state, prototypes[j], kappa)
                
                L_rec = criterion(x_recon, x)
                L_rc = -gamma * reward * torch.log(p_ij[action]) - (1 - p_ij[action])
                loss = L_rec + L_rc

                optimizer.zero_grad()
                prototypes_optimizer.zero_grad()
                micro_cluster_optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                prototypes_optimizer.step()
                micro_cluster_optimizer.step()

                total_loss += loss.item()
        
        print_training_status(epoch, epochs, total_loss)
        # Update micro-clusters at the end of each epoch
        micro_clusters = create_micro_clusters(data.numpy(), num_micro_clusters)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Evaluation Metrics
    predicted_labels = np.zeros(len(data))
    for i, x in enumerate(data):
        x = torch.tensor(x, dtype=torch.float32)
        _, z = model(x)
        probabilities = [decision_probability(z, prototypes[j], kappa).item() for j in range(K)]
        predicted_labels[i] = np.argmax(probabilities)

    accuracy = np.mean(predicted_labels == labels)
    nmi = normalized_mutual_information(labels, predicted_labels)
    ari = adjusted_rand_index(labels, predicted_labels)
    fscore = f1_score(labels, predicted_labels, average='weighted')

    # Memory Usage
    memory_usage = psutil.Process().memory_info().rss / 1024 ** 2  # in MB

    print(f"Accuracy: {accuracy:.4f}")
    print(f"NMI: {nmi:.4f}")
    print(f"ARI: {ari:.4f}")
    print(f"F-score: {fscore:.4f}")
    print(f"Memory Usage: {memory_usage:.2f} MB")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")

    return model, prototypes, micro_clusters

def normalized_mutual_information(true_labels, predicted_labels):
    return normalized_mutual_info_score(true_labels, predicted_labels)

def adjusted_rand_index(true_labels, predicted_labels):
    return adjusted_rand_score(true_labels, predicted_labels)

# Example usage with a mock dataset
if __name__ == "__main__":
    # Create a mock dataset similar to MNIST
    num_samples = 100
    num_features = 20
    data = np.random.rand(num_samples, num_features)
    labels = np.random.randint(0, 5, num_samples)

    K = 5  # Number of clusters
    epochs = 10  # Number of epochs
    lr = 0.001
    gamma = 0.9
    v = 1.0
    num_micro_clusters = 10  # Number of micro-clusters
    batch_size = 20  # Batch size for replay memory
    epsilon = 0.1  # ε-greedy strategy parameter

    model, prototypes, micro_clusters = train_drc(data, labels, K, epochs, lr, gamma, v, num_micro_clusters, batch_size, epsilon)
