import torch
from model import vgg16_cifar
from dataset import get_dataloaders
from gradcam import GradCAM
from utils import plot_gradcam, denormalize
import matplotlib.pyplot as plt
import numpy as np

def load_model(checkpoint_path, device):
    model = vgg16_cifar(num_classes=10).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def run_analysis(model, dataloader, classes, device, num_samples=5):
    # Find correct and incorrect samples
    correct_samples = []
    incorrect_samples = []
    
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
    # For VGG16, the last conv layer is usually features[40] or similar
    # In my make_layers, I'll find the last Conv2d layer
    target_layer = None
    for module in reversed(list(model.features.modules())):
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            break
            
    cam = GradCAM(model, target_layer)
    
    print("Visualizing Correct Predictions:")
    for img, label, pred in correct_samples:
        heatmap = cam.generate_cam(img.unsqueeze(0).to(device))
        plot_gradcam(img, heatmap, title=f"Correct: {classes[label]}")
        
    print("Visualizing Incorrect Predictions:")
    for img, label, pred in incorrect_samples:
        heatmap = cam.generate_cam(img.unsqueeze(0).to(device))
        plot_gradcam(img, heatmap, title=f"True: {classes[label]}, Pred: {classes[pred]}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, classes = get_dataloaders(batch_size=32)
    # This requires a trained model
    if torch.os.path.exists("outputs/models/vgg16_best.pth"):
        model = load_model("outputs/models/vgg16_best.pth", device)
        run_analysis(model, test_loader, classes, device)
    else:
        print("Trained model not found. Please run training first.")
