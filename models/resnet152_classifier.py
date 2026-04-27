import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights

class ResNet152Classifier(nn.Module):
    def __init__(self, pretrained=True, num_classes=1, dropout=0.5):
        super().__init__()
        weights = ResNet152_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet152(weights=weights)
        
        # Replace final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)