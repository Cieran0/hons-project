import torch
import torch.nn as nn

class WeightedCrossEntropyLoss(nn.Module):
    """
    Binary Cross Entropy Loss with optional class weighting.
    For binary classification with logits output.
    """
    def __init__(self, pos_weight=None):
        super().__init__()
        # Ensure pos_weight is a scalar tensor
        if pos_weight is not None:
            self.pos_weight = pos_weight.squeeze()
        else:
            self.pos_weight = None
    
    def forward(self, inputs, targets):
        # Squeeze to 1D if needed
        targets = targets.squeeze()
        inputs = inputs.squeeze()
        
        # Compute BCE with logits
        loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets,
            reduction='mean',
            pos_weight=self.pos_weight
        )
        return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight
    
    def forward(self, inputs, targets):
        return nn.functional.cross_entropy(inputs, targets, weight=self.weight)