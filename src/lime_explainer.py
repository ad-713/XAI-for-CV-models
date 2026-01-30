import torch
import numpy as np
from lime import lime_image
from torchvision import transforms
import torch.nn.functional as F

class LimeExplainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.explainer = lime_image.LimeImageExplainer()
        
        # CIFAR-10 normalization constants
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def predict_fn(self, images):
        """
        LIME prediction function.
        images: numpy array of shape (N, H, W, C) in range [0, 1]
        """
        self.model.eval()
        batch = torch.stack([self.preprocess(img.astype('float32')) for img in images]).to(self.device)
        
        with torch.no_grad():
            logits = self.model(batch.float())
            probs = F.softmax(logits, dim=1)
            
        return probs.detach().cpu().numpy()

    def generate_explanation(self, image_np, target_class=None, num_samples=1000):
        """
        image_np: numpy array (H, W, C) in range [0, 1]
        """
        explanation = self.explainer.explain_instance(
            image_np.astype('double'), 
            self.predict_fn, 
            top_labels=10, 
            hide_color=0, 
            num_samples=num_samples
        )
        
        if target_class is None:
            # Get the top predicted class from the explanation
            target_class = explanation.top_labels[0]
            
        # Get mask for the target class
        # positive_only=True means we only show regions that contribute positively to the target class
        dict_masks = dict(explanation.local_exp[target_class])
        
        # Create a heatmap from the segments
        mask = np.zeros(explanation.segments.shape)
        for segment_id, importance in dict_masks.items():
            mask[explanation.segments == segment_id] = importance
            
        # Normalize mask to [0, 1] for visualization consistency with Grad-CAM
        if np.max(mask) > 0:
            mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        
        return mask, explanation
