from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout
import torch
import torch.nn as nn


class VGG11Classifier(nn.Module):
    """VGG trunk + MLP. Backbone returns (7x7 bottleneck, skip list) — we only use the bottleneck."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        """in_channels is ignored (3-channel encoder); kept for a stable constructor signature."""
        super(VGG11Classifier, self).__init__()
        self.encoder = VGG11Encoder()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, _ = self.encoder(x)
        return self.classifier(bottleneck)