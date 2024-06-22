import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score

# Function to perform hyper-parameter analysis
def hyperparameter_analysis(data, labels, lr_values, gamma_values, v_values):
    accuracies = np.zeros((len(lr_values), len(gamma_values), len(v_values)))
    nmis = np.zeros((len(lr_values), len(gamma_values), len(v_values)))

    for i, lr in enumerate(lr_values):
        for j, gamma in enumerate(gamma_values):
            for k, v in enumerate(v_values):
                model, prototypes, micro_clusters = train_drc(data, labels, K=5, epochs=10, lr=lr, gamma=gamma, v=v, num_micro_clusters=10, batch_size=20, epsilon=0.1)

                predicted_labels = np.zeros(len(data))
                for idx, x in enumerate(data):
                    x = torch.tensor(x, dtype=torch.float32)
                    _, z = model(x)
                    probabilities = [decision_probability(z, prototypes[c], 1.0).item() for c in range(5)]
                    predicted_labels[idx] = np.argmax(probabilities)

                accuracies[i, j, k] = np.mean(predicted_labels == labels)
                nmis[i, j, k] = normalized_mutual_info_score(labels, predicted_labels)

    return accuracies, nmis

# Function to create 3D bar plots
def plot_3d_bars(x, y, z, values, title, xlabel, ylabel, zlabel):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xpos, ypos = np.meshgrid(x, y)
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = values.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()

# Example usage
if __name__ == "__main__":
    import torch
    from OSGM import train_drc, decision_probability

    # Create a mock dataset similar to MNIST for testing
    num_samples = 100
    num_features = 20
    data = np.random.rand(num_samples, num_features)
    labels = np.random.randint(0, 5, num_samples)

    lr_values = [1e-4, 1e-3, 1e-2]
    gamma_values = [0.5, 0.9, 0.99]
    v_values = [1, 10, 100]

    accuracies, nmis = hyperparameter_analysis(data, labels, lr_values, gamma_values, v_values)

    plot_3d_bars(lr_values, gamma_values, v_values, accuracies, 'ACC: γ vs lr', 'Learning Rate (lr)', 'Gamma (γ)', 'Accuracy')
    plot_3d_bars(v_values, gamma_values, lr_values, nmis, 'NMI: v vs γ', 'V', 'Gamma (γ)', 'NMI')
