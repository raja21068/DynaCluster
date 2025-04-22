import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture gradients and activations
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output
    
    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]
    
    def generate_heatmap(self, input_image, cluster_id):
        self.model.eval()
        input_image = input_image.unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        cluster_probs = self.model(input_image)  # Assume model outputs cluster probabilities
        target_prob = cluster_probs[0, cluster_id]
        
        # Backward pass for target cluster
        self.model.zero_grad()
        target_prob.backward()
        
        # Compute Grad-CAM
        gradients = self.gradients.mean(dim=[2, 3], keepdim=True)  # Global average pooling
        heatmap = F.relu((gradients * self.activations).sum(dim=1)).squeeze(0)
        heatmap = heatmap / (heatmap.max() + 1e-8)  # Normalize
        
        return heatmap.cpu().detach().numpy()

def visualize_gradcam(image_path, model, target_layer, cluster_id, output_path):
    # Load and preprocess image
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    input_image = transform(image).cuda()
    
    # Generate heatmap
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate_heatmap(input_image, cluster_id)
    
    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    
    # Convert to RGB heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay heatmap