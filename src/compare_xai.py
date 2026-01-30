import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm

from model import vgg16_cifar
from dataset import get_dataloaders
from gradcam import GradCAM
from lime_explainer import LimeExplainer
from shap_explainer import ShapExplainer
from metrics import measure_runtime, calculate_deletion_score
from utils import denormalize, show_cam_on_image

def load_model(checkpoint_path, device):
    model = vgg16_cifar(num_classes=10).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data
    _, test_loader, classes = get_dataloaders(batch_size=100)
    test_batch = next(iter(test_loader))
    images, labels = test_batch
    
    # 2. Load model
    model_path = "outputs/models/vgg16_best.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    model = load_model(model_path, device)

    # 3. Initialize Explainers
    # Grad-CAM target layer (last conv layer)
    target_layer = None
    for module in reversed(list(model.features.modules())):
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            break
    gradcam = GradCAM(model, target_layer)
    
    lime_explainer = LimeExplainer(model, device)
    
    # Use first 100 images from test set as background for SHAP
    shap_explainer = ShapExplainer(model, images[:100], device)

    # 4. Prepare for comparison
    results_dir = "outputs/comparison"
    os.makedirs(results_dir, exist_ok=True)
    
    num_test_samples = 5
    metrics_list = []

    print(f"Running comparison on {num_test_samples} samples...")
    for i in range(num_test_samples):
        img_tensor = images[i:i+1] # (1, 3, 32, 32)
        label = labels[i].item()
        
        # Original image for LIME and visualization
        img_np = denormalize(img_tensor.clone()).cpu().squeeze().numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)

        # Target class (ground truth or prediction? let's use prediction)
        with torch.no_grad():
            pred = model(img_tensor.to(device)).argmax(dim=1).item()
        
        import cv2
        # --- Grad-CAM ---
        (gc_heatmap), gc_time = measure_runtime(gradcam.generate_cam, img_tensor.to(device), target_class=pred)
        gc_heatmap = cv2.resize(gc_heatmap, (img_np.shape[1], img_np.shape[0]))
        gc_faith, _ = calculate_deletion_score(model, img_tensor, gc_heatmap, pred, device)
        
        # --- LIME ---
        # LIME needs more samples for better quality, but it's slow. num_samples=500 is a compromise.
        (lime_heatmap, _), lime_time = measure_runtime(lime_explainer.generate_explanation, img_np, target_class=pred, num_samples=500)
        lime_heatmap = cv2.resize(lime_heatmap, (img_np.shape[1], img_np.shape[0]))
        lime_faith, _ = calculate_deletion_score(model, img_tensor, lime_heatmap, pred, device)
        
        # --- SHAP ---
        shap_heatmap, shap_time = measure_runtime(shap_explainer.generate_explanation, img_tensor, target_class=pred)
        shap_heatmap = cv2.resize(shap_heatmap, (img_np.shape[1], img_np.shape[0]))
        shap_faith, _ = calculate_deletion_score(model, img_tensor, shap_heatmap, pred, device)

        # Store metrics
        metrics_list.append({'Method': 'Grad-CAM', 'Sample': i, 'Runtime': gc_time, 'Deletion Score': gc_faith})
        metrics_list.append({'Method': 'LIME', 'Sample': i, 'Runtime': lime_time, 'Deletion Score': lime_faith})
        metrics_list.append({'Method': 'SHAP', 'Sample': i, 'Runtime': shap_time, 'Deletion Score': shap_faith})

        # Visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(img_np)
        axes[0].set_title(f"Original\nPred: {classes[pred]}")
        axes[0].axis('off')

        gc_vis = show_cam_on_image(img_np, gc_heatmap, use_rgb=True)
        axes[1].imshow(gc_vis)
        axes[1].set_title(f"Grad-CAM\nTime: {gc_time:.2f}s, DS: {gc_faith:.3f}")
        axes[1].axis('off')

        lime_vis = show_cam_on_image(img_np, lime_heatmap, use_rgb=True)
        axes[2].imshow(lime_vis)
        axes[2].set_title(f"LIME\nTime: {lime_time:.2f}s, DS: {lime_faith:.3f}")
        axes[2].axis('off')

        shap_vis = show_cam_on_image(img_np, shap_heatmap, use_rgb=True)
        axes[3].imshow(shap_vis)
        axes[3].set_title(f"SHAP\nTime: {shap_time:.2f}s, DS: {shap_faith:.3f}")
        axes[3].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"comparison_{i}.png"))
        plt.close()

    # 5. Summary
    df = pd.DataFrame(metrics_list)
    summary = df.groupby('Method').agg({'Runtime': 'mean', 'Deletion Score': 'mean'}).reset_index()
    print("\nComparison Summary (Mean Values):")
    print(summary)
    
    summary.to_csv(os.path.join(results_dir, "summary_metrics.csv"), index=False)
    
    # Plot summary metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    summary.plot(kind='bar', x='Method', y='Runtime', ax=ax1, color=['blue', 'green', 'orange'])
    ax1.set_title('Average Runtime (seconds)')
    ax1.set_ylabel('Seconds')
    
    summary.plot(kind='bar', x='Method', y='Deletion Score', ax=ax2, color=['blue', 'green', 'orange'])
    ax2.set_title('Average Deletion Score (Lower is better)')
    ax2.set_ylabel('AUC')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "summary_plot.png"))
    plt.show()

if __name__ == "__main__":
    main()
