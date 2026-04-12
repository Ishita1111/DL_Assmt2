import torch.nn as nn
from .vgg11 import VGG11Backbone

class RegressionHead(nn.Module):
    def __init__(self):
        super(RegressionHead, self).__init__()
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            # Sigmoid: matches dataset boxes normalized to [0,1] (see pets_dataset)
            nn.Linear(256, 4),  # cx, cy, w, h
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.regressor(x)

class VGG11Localizer(nn.Module):
    def __init__(self, freeze_backbone=False):
        super(VGG11Localizer, self).__init__()
        self.backbone = VGG11Backbone()
        self.locator = RegressionHead()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        # localization only needs the bottleneck, not skip tensors
        bottleneck, _ = self.backbone(x)
        return self.locator(bottleneck)