import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
from collections import deque
import random

# Import necessary components from other files
from model_config import Autoencoder
from OSGM import cauchy_similarity, decision_probability, compute_reward
from PrintingFile import print_training_status
from dataset_loader import load_stl10, load_imagenet_tiny, load_mnist, load_usps, load_fmnist, load_cifar10

# Define MicroCluster class
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

def train_drc(train_loader, K, epochs, lr, gamma, v, num_micro_clusters, batch_size, epsilon):
    input_dim = next(iter(train_loader))[0].shape[1:].numel()
    latent_dim = 10  # Latent dimension size for autoencoder
    kappa = 1.0  # Parameter for cauchy similarity

    model = Autoencoder(input_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Initialize micro-clusters with initial data
    data, labels = next(iter(train_loader))
    data = data.view(data.size(0), -1).numpy()
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

    for epoch in range(epochs):
        total_loss = 0
        
        for data, _ in train_loader:
            data = data.view(data.size(0), -1)
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
        data, _ = next(iter(train_loader))
        data = data.view(data.size(0), -1).numpy()
        micro_clusters = create_micro_clusters(data, num_micro_clusters)

    return model, prototypes, micro_clusters

# Example usage
if __name__ == "__main__":
    # Choose the dataset to load
    dataset = 'mnist'  # Change this to 'stl10', 'imagenet_tiny', 'usps', 'fmnist', 'cifar10' as needed

    if dataset == 'stl10':
        train_loader, _ = load_stl10()
    elif dataset == 'imagenet_tiny':
        train_loader, _ = load_imagenet_tiny()
    elif dataset == 'mnist':
        train_loader, _ = load_mnist()
    elif dataset == 'usps':
        train_loader, _ = load_usps()
    elif dataset == 'fmnist':
        train_loader, _ = load_fmnist()
    elif dataset == 'cifar10':
        train_loader, _ = load_cifar10()
    else:
        raise ValueError("Unsupported dataset")

    K = 5  # Number of clusters
    epochs = 10  # Number of epochs
    lr = 0.001
    gamma = 0.9
    v = 1.0
    num_micro_clusters = 10  # Number of micro-clusters
    batch_size = 20  # Batch size for replay memory
    epsilon = 0.1  # ε-greedy strategy parameter

    model, prototypes, micro_clusters = train_drc(train_loader, K, epochs, lr, gamma, v, num_micro_clusters, batch_size, epsilon)
