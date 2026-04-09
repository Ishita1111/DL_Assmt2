import torch
import torch.nn as nn

class CustomDropout(nn.Module):
    def __init__(self, p=0.5):
        super(CustomDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout probability must be between 0 and 1.")
        self.p = p

    def forward(self, x):
        # Deterministic behavior when not in training mode [cite: 52, 53]
        if not self.training or self.p == 0.0:
            return x
        
        # Generate binary mask [cite: 52]
        mask = (torch.rand(x.shape, device=x.device) > self.p).float()
        
        # Apply inverted dropout scaling [cite: 52]
        return (x * mask) / (1.0 - self.p)