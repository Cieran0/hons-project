import torch
import torch.nn as nn
from torchvision.models import densenet121, densenet169, DenseNet121_Weights, DenseNet169_Weights

class DenseNetClassifier(nn.Module):
    def __init__(self, model_name='densenet121', pretrained=True, num_classes=1, dropout=0.5):
        super().__init__()
        
        # Select weights
        if model_name == 'densenet121':
            weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = densenet121(weights=weights)
        elif model_name == 'densenet169':
            weights = DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = densenet169(weights=weights)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        
        # Replace final layer (DenseNet uses .classifier instead of .fc)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)