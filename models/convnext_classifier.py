import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, convnext_base, ConvNeXt_Tiny_Weights, ConvNeXt_Base_Weights

class ConvNeXtClassifier(nn.Module):
    def __init__(self, model_name='convnext_tiny', pretrained=True, num_classes=1, dropout=0.5):
        super().__init__()
        
        # Select weights
        if model_name == 'convnext_tiny':
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = convnext_tiny(weights=weights)
        elif model_name == 'convnext_base':
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = convnext_base(weights=weights)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        
        # Preserve Norm & Flatten, inject Dropout, update Linear
        old_classifier = self.backbone.classifier
        in_features = old_classifier[-1].in_features
        
        self.backbone.classifier = nn.Sequential(
            old_classifier[0], # LayerNorm
            old_classifier[1], # Flatten
            nn.Dropout(dropout),            
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)