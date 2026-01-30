import torch
import numpy as np
import shap

class ShapExplainer:
    def __init__(self, model, background_images, device):
        """
        model: PyTorch model
        background_images: torch.Tensor used to initialize the explainer (representative subset)
        device: torch device
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Initialize GradientExplainer
        # We use a subset of the dataset as background
        self.explainer = shap.GradientExplainer(self.model, background_images.to(self.device))

    def generate_explanation(self, image_tensor, target_class=None):
        """
        image_tensor: torch.Tensor of shape (1, C, H, W)
        """
        image_tensor = image_tensor.to(self.device)
        
        # If target_class is None, get the top prediction
        if target_class is None:
            with torch.no_grad():
                output = self.model(image_tensor)
                target_class = output.argmax(dim=1).item()
        
        # shap_values is a list of arrays (one for each class)
        # Each array has shape (1, C, H, W)
        shap_values = self.explainer.shap_values(image_tensor, ranked_outputs=None)
        
        print(f"DEBUG: shap_values type: {type(shap_values)}")
        if isinstance(shap_values, list):
            print(f"DEBUG: shap_values list len: {len(shap_values)}")
            if len(shap_values) > 0:
                 print(f"DEBUG: shap_values[0] shape: {shap_values[0].shape}")
        else:
             print(f"DEBUG: shap_values shape: {shap_values.shape}")

        # Handle different SHAP return formats
        if isinstance(shap_values, list):
            target_shap_values = shap_values[target_class]
        elif len(shap_values.shape) == 5 and shap_values.shape[-1] == 10:
            # (batch, C, H, W, classes)
            target_shap_values = shap_values[..., target_class]
        elif len(shap_values.shape) == 5:
            # (batch, classes, C, H, W)
            target_shap_values = shap_values[:, target_class]
        else:
            target_shap_values = shap_values
        
        # To create a heatmap, we often sum across the color channels
        # and take the absolute value or just positive contributions
        heatmap = np.sum(target_shap_values[0], axis=0)
        
        # Absolute values often show "importance" regardless of direction
        heatmap = np.abs(heatmap)
        
        # Normalize to [0, 1]
        if np.max(heatmap) > 0:
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
            
        return heatmap
