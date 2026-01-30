import torch
import time
import numpy as np
import torch.nn.functional as F
import cv2

def measure_runtime(explainer_func, *args, **kwargs):
    """
    Measure the time it takes to run an explainer function.
    """
    start_time = time.time()
    result = explainer_func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def calculate_deletion_score(model, image_tensor, heatmap, target_class, device, steps=10):
    """
    Calculate Deletion Score (Faithfulness).
    Gradually removes important pixels and measures the drop in target class probability.
    Lower AUC of the deletion curve indicates a more 'faithful' explanation.
    
    image_tensor: (1, 3, 32, 32) normalized
    heatmap: (32, 32) normalized [0, 1]
    """
    model.eval()
    h, w = heatmap.shape
    total_pixels = h * w
    
    # Flatten heatmap and get indices of pixels sorted by importance (descending)
    flat_heatmap = heatmap.flatten()
    sorted_indices = np.argsort(-flat_heatmap)
    
    # Pre-calculate mean for 'removal' (CIFAR-10 mean)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
    
    probs = []
    
    # Initial probability
    with torch.no_grad():
        output = model(image_tensor.to(device))
        initial_prob = F.softmax(output, dim=1)[0, target_class].item()
        probs.append(initial_prob)
        
    # Masking steps
    mask = torch.ones((1, 1, h, w)).to(device)
    
    pixels_per_step = total_pixels // steps
    
    current_image = image_tensor.clone().to(device)
    
    for i in range(steps):
        # Indices to mask in this step
        idx_to_mask = sorted_indices[i * pixels_per_step : (i + 1) * pixels_per_step]
        
        # Create a mask for these pixels
        for idx in idx_to_mask:
            row = idx // w
            col = idx % w
            # Replace with mean
            current_image[0, :, row, col] = mean[0, :, 0, 0]
            
        with torch.no_grad():
            output = model(current_image)
            prob = F.softmax(output, dim=1)[0, target_class].item()
            probs.append(prob)
            
    # Calculate Area Under Curve (AUC) for the deletion curve
    # Normalized by the initial probability and number of steps
    auc = np.trapz(probs, dx=1.0/steps)
    
    return auc, probs
