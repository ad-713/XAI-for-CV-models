import torch
from model import vgg16_cifar
from dataset import get_dataloaders
from gradcam import GradCAM
from utils import plot_gradcam, denormalize, show_cam_on_image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

def load_model(checkpoint_path, device):
    model = vgg16_cifar(num_classes=10).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def save_gradcam_result(img_tensor, heatmap, save_path, title):
    img = denormalize(img_tensor.clone()).cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    cam_image = show_cam_on_image(img, heatmap_resized, use_rgb=True)
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cam_image)
    plt.title(title)
    plt.axis('off')
    
    plt.savefig(save_path)
    plt.close()

def run_analysis(model, dataloader, classes, device, num_samples=20, save_dir="outputs/gradcam_gallery"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "correct"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "incorrect"), exist_ok=True)

    # Find correct and incorrect samples
    correct_samples = []
    incorrect_samples = []
    
    print(f"Collecting {num_samples} correct and {num_samples} incorrect samples...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            for i in range(inputs.size(0)):
                if predicted[i] == labels[i]:
                    if len(correct_samples) < num_samples:
                        correct_samples.append((inputs[i], labels[i], predicted[i]))
                else:
                    if len(incorrect_samples) < num_samples:
                        incorrect_samples.append((inputs[i], labels[i], predicted[i]))
                
                if len(correct_samples) >= num_samples and len(incorrect_samples) >= num_samples:
                    break
            if len(correct_samples) >= num_samples and len(incorrect_samples) >= num_samples:
                break
                
    # Initialize GradCAM
    target_layer = None
    for module in reversed(list(model.features.modules())):
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            break
            
    cam = GradCAM(model, target_layer)
    
    print("Generating Grad-CAM for Correct Predictions...")
    for idx, (img, label, pred) in enumerate(correct_samples):
        heatmap = cam.generate_cam(img.unsqueeze(0).to(device))
        title = f"Correct: {classes[label]}"
        save_path = os.path.join(save_dir, "correct", f"sample_{idx}.png")
        save_gradcam_result(img, heatmap, save_path, title)
        # Also plot the first 5 in notebook
        if idx < 5:
            plot_gradcam(img, heatmap, title=title)
        
    print("Generating Grad-CAM for Incorrect Predictions...")
    for idx, (img, label, pred) in enumerate(incorrect_samples):
        heatmap = cam.generate_cam(img.unsqueeze(0).to(device))
        title = f"True: {classes[label]}, Pred: {classes[pred]}"
        save_path = os.path.join(save_dir, "incorrect", f"sample_{idx}.png")
        save_gradcam_result(img, heatmap, save_path, title)
        # Also plot the first 5 in notebook
        if idx < 5:
            plot_gradcam(img, heatmap, title=title)
            
    print(f"Analysis complete. Gallery saved to {save_dir}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, classes = get_dataloaders(batch_size=32)
    
    model_path = "outputs/models/vgg16_best.pth"
    if os.path.exists(model_path):
        model = load_model(model_path, device)
        run_analysis(model, test_loader, classes, device, num_samples=20)
    else:
        print("Trained model not found. Please run training first.")
