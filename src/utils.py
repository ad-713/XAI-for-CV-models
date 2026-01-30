import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def denormalize(tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def show_cam_on_image(img: np.ndarray, mask: np.ndarray, use_rgb: bool = False, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1.1:
        img = img / 255.0

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def plot_gradcam(img_tensor, heatmap, title=None):
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
    plt.title(f"Grad-CAM {title if title else ''}")
    plt.axis('off')
    plt.show()

def plot_history(history, save_path=None):
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def get_classification_report(y_true, y_pred, classes):
    return classification_report(y_true, y_pred, target_names=classes)
