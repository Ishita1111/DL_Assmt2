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
            nn.Linear(256, 4), # [Xcenter, Ycenter, width, height]
            nn.Sigmoid() # Assuming target coordinates are normalized between 0 and 1
        )

    def forward(self, x):
        return self.regressor(x)

# Added to satisfy the skeleton's models/__init__.py and Task 2 requirements
class VGG11Localizer(nn.Module):
    def __init__(self, freeze_backbone=False):
        super(VGG11Localizer, self).__init__()
        self.backbone = VGG11Backbone()
        self.locator = RegressionHead()
        
        # Task 2 Requirement: Option to freeze backbone weights
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        bottleneck, _ = self.backbone(x) # We don't need skip connections for just localization
        return self.locator(bottleneck)