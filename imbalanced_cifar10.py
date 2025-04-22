import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ImbalancedCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, imbalance_ratio=0.1):
        # Load CIFAR-10 dataset
        self.cifar10 = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform
        self.imbalance_ratio = imbalance_ratio
        self.minority_classes = [2, 3, 4, 5]  # bird, cat, deer, dog
        self.data, self.targets = self.subsample_dataset()
    
    def subsample_dataset(self):
        data = self.cifar10.data
        targets = np.array(self.cifar10.targets)
        new_data = []
        new_targets = []
        
        for class_id in range(10):
            class_indices = np.where(targets == class_id)[0]
            if class_id in self.minority_classes:
                # Sub-sample minority classes
                num_samples = int(len(class_indices) * self.imbalance_ratio)
                selected_indices = np.random.choice(class_indices, size=num_samples, replace=False)
            else:
                # Keep all samples for majority classes
                selected_indices = class_indices
            
            new_data.append(data[selected_indices])
            new_targets.extend([class_id] * len(selected_indices))
        
        new_data = np.concatenate(new_data, axis=0)
        return new_data, new_targets
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = T.ToPILImage()(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, target

# Example usage
def create_imbalanced_cifar10_dataloader(root, batch_size=32):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImbalancedCIFAR10(root=root, train=True, transform=transform, imbalance_ratio=0.1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = ImbalancedCIFAR10(root=root, train=False, transform=transform, imbalance_ratio=0.1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Verify dataset imbalance
def verify_imbalance(dataset):
    from collections import Counter
    targets = dataset.targets
    counts = Counter(targets)
    print("Class distribution:", counts)