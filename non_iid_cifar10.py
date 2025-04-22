import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from torch.utils.data import Dataset, DataLoader

class NonIIDCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, client_id=0, num_clients=10):
        self.cifar10 = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform
        self.client_id = client_id
        self.num_clients = num_clients
        self.data, self.targets = self.create_non_iid_subset()
    
    def create_non_iid_subset(self):
        data = self.cifar10.data
        targets = np.array(self.cifar10.targets)
        num_classes = 10
        
        # Sort by class for sharding
        sorted_indices = np.argsort(targets)
        data = data[sorted_indices]
        targets = targets[sorted_indices]
        
        # Assign skewed class distributions to clients
        samples_per_client = len(targets) // self.num_clients
        client_data = []
        client_targets = []
        
        # Assign primary classes to each client (80% of samples)
        primary_classes = [(self.client_id + i) % num_classes for i in range(2)]  # 2 primary classes
        other_classes = [i for i in range(num_classes) if i not in primary_classes]
        
        for class_id in range(num_classes):
            class_indices = np.where(targets == class_id)[0]
            num_samples = len(class_indices)
            
            if class_id in primary_classes:
                # Allocate 40% of samples per primary class (80% total)
                selected_samples = int(0.4 * samples_per_client)
            else:
                # Allocate remaining 20% across other classes
                selected_samples = int(0.2 * samples_per_client / len(other_classes))
            
            selected_indices = np.random.choice(class_indices, size=min(selected_samples, num_samples), replace=False)
            client_data.append(data[selected_indices])
            client_targets.extend([class_id] * len(selected_indices))
        
        client_data = np.concatenate(client_data, axis=0)
        return client_data, client_targets
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = T.ToPILImage()(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, target

# Example usage
def create_non_iid_cifar10_dataloaders(root, num_clients=10, batch_size=32):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    client_loaders = []
    for client_id in range(num_clients):
        dataset = NonIIDCIFAR10(root=root, train=True, transform=transform, client_id=client_id, num_clients=num_clients)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
    
    test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return client_loaders, test_loader

# Verify non-IID distribution
def verify_non_iid_distribution(client_loaders):
    from collections import Counter
    for client_id, loader in enumerate(client_loaders):
        targets = [target.item() for _, target in loader.dataset]
        counts = Counter(targets)
        print(f"Client {client_id} class distribution:", counts)