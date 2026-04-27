import torch
import torch.nn as nn
import timm

class ImageOnlyViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, num_classes=1, dropout=0.5):
        super().__init__()
        # Load ViT without classification head
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.hidden_dim = self.vit.embed_dim
        # Classification Head with Dropout
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, num_classes)
        )
    
    def forward(self, x):
        features = self.vit(x)
        return self.head(features)

def create_model(pretrained=True, device='cuda'):
    model = ImageOnlyViT(pretrained=pretrained)
    model = model.to(device)
    return model