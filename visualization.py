import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def visualize_clusters(data, labels, model, title):
    """
    Visualizes the clustering result using t-SNE.

    Parameters:
    - data: numpy array, the input data
    - labels: numpy array, the true labels or cluster assignments
    - model: trained autoencoder model to transform data into latent space
    - title: string, the title of the plot
    """
    # Transform data to latent space
    data_tensor = torch.tensor(data, dtype=torch.float32)
    with torch.no_grad():
        _, latent_data = model(data_tensor)

    latent_data = latent_data.numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(latent_data)

    # Plot t-SNE result
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.title(title)
    plt.colorbar(scatter)
    plt.show()

if __name__ == "__main__":
    # Mock dataset similar to MNIST for testing
    num_samples = 100
    num_features = 20
    data = np.random.rand(num_samples, num_features)
    labels = np.random.randint(0, 5, num_samples)

    # Load the trained model
    from OSGM import train_drc, Autoencoder

    K = 5  # Number of clusters
    epochs = 10  # Number of epochs
    lr = 0.001
    gamma = 0.9
    v = 1.0
    num_micro_clusters = 10  # Number of micro-clusters
    batch_size = 20  # Batch size for replay memory
    epsilon = 0.1  # ε-greedy strategy parameter

    # Train the model
    model, prototypes, micro_clusters = train_drc(data, labels, K, epochs, lr, gamma, v, num_micro_clusters, batch_size, epsilon)

    # Visualize the clusters
    visualize_clusters(data, labels, model, "Mock Dataset Clustering Visualization")
