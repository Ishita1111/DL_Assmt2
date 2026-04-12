import torch
import torch.nn as nn

class CustomDropout(nn.Module):
    def __init__(self, p=0.5):
        super(CustomDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout probability must be between 0 and 1.")
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x

        keep = (torch.rand(x.shape, device=x.device) > self.p).float()
        # inverted dropout: scale at train time so inference is a plain forward
        return (x * keep) / (1.0 - self.p)