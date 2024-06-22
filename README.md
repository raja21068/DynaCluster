# Deep Reinforcement Clustering

This repository contains an implementation of a Deep Reinforcement Clustering (DRC) algorithm. The algorithm uses an autoencoder for feature extraction, KMeans for initial clustering, and reinforcement learning to update the clustering model.

## Features

- Autoencoder for feature extraction
- KMeans for initial clustering
- Reinforcement learning for updating clusters
- Evaluation metrics: Accuracy, Normalized Mutual Information (NMI), Adjusted Rand Index (ARI), F-score
- Memory usage and execution time tracking

## Requirements

- Python 3.7+
- torch
- numpy
- scikit-learn
- psutil
- pandas

## Installation

Install the required libraries using pip:

```bash
pip install torch numpy scikit-learn psutil pandas

Files
deep_reinforcement_clustering_test-v1.py: Main script to run the DRC algorithm with a mock dataset.
model_config.py: Defines the autoencoder model.
DynaCluster.py: Contains functions for calculating similarity, decision probability, and reward.
PrintingFile.py: Contains the function to print training status.
dataset_loader.py: Contains functions for loading various datasets (STL-10, ImageNet-Tiny, MNIST, USPS, F-MNIST, CIFAR-10).
![image](https://github.com/raja21068/DynaCluster/assets/10251446/58b84f06-9bae-41ad-a4ee-b13cced667e8)
