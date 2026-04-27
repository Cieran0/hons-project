import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet101_Weights


class ResNet101Classifier(nn.Module):

    def __init__(self, pretrained=True, num_classes=1, dropout=0.5):
        super().__init__()
        
        # Load ResNet101 with pretrained weights
        if pretrained:
            weights = ResNet101_Weights.IMAGENET1K_V2
            self.resnet = models.resnet101(weights=weights)
        else:
            self.resnet = models.resnet101(weights=None)
        
        # Add intermediate dropout for regularization
        self._add_intermediate_dropout()
        
        # Replace final FC layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, num_classes)
        )
        
        print(f"✅ ResNet101 initialized (pretrained={pretrained}, dropout={dropout})")
    
    def _add_intermediate_dropout(self):
        # Add dropout after each layer block
        self.resnet.layer1 = nn.Sequential(
            self.resnet.layer1,
            nn.Dropout2d(p=0.01)
        )
        self.resnet.layer2 = nn.Sequential(
            self.resnet.layer2,
            nn.Dropout2d(p=0.01)
        )
        self.resnet.layer3 = nn.Sequential(
            self.resnet.layer3,
            nn.Dropout2d(p=0.01)
        )
        self.resnet.layer4 = nn.Sequential(
            self.resnet.layer4,
            nn.Dropout2d(p=0.01)
        )
    
    def forward(self, x):
        return self.resnet(x)


def create_model(pretrained=True, device='cuda', num_classes=1, dropout=0.5):
    model = ResNet101Classifier(pretrained=pretrained, num_classes=num_classes, dropout=dropout)
    model = model.to(device)
    return model