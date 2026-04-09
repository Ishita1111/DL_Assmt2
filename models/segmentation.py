import torch
import torch.nn as nn
from .vgg11 import VGG11Backbone

class UNetDecoder(nn.Module):
    def __init__(self, num_classes=3): # 3 classes for trimap
        super(UNetDecoder, self).__init__()
        
        # Block 5: Upsample 7x7 -> 14x14 (Matches features[4])
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1), # 512 upsampled + 512 skip
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Block 4: Upsample 14x14 -> 28x28 (Matches features[3])
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1), # 512 upsampled + 512 skip
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Block 3: Upsample 28x28 -> 56x56 (Matches features[2])
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), # 256 upsampled + 256 skip
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Block 2: Upsample 56x56 -> 112x112 (Matches features[1])
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), # 128 upsampled + 128 skip
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Block 1: Upsample 112x112 -> 224x224 (Matches features[0])
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), # 64 upsampled + 64 skip
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final projection to num_classes
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x, features):
        # x starts as 7x7
        x = self.up5(x)
        x = torch.cat([x, features[4]], dim=1) # Concat 14x14
        x = self.dec5(x)
        
        x = self.up4(x)
        x = torch.cat([x, features[3]], dim=1) # Concat 28x28
        x = self.dec4(x)
        
        x = self.up3(x)
        x = torch.cat([x, features[2]], dim=1) # Concat 56x56
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, features[1]], dim=1) # Concat 112x112
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, features[0]], dim=1) # Concat 224x224
        x = self.dec1(x)
        
        return self.final_conv(x)

# Added to satisfy the skeleton's models/__init__.py and Task 3 requirements
class VGG11UNet(nn.Module):
    def __init__(self, num_classes=3, freeze_backbone=False):
        super(VGG11UNet, self).__init__()
        self.backbone = VGG11Backbone()
        self.segmenter = UNetDecoder(num_classes)
        
        # Useful for Task 2.3: Transfer Learning Showdown (Strict Feature Extractor)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        bottleneck, skip_features = self.backbone(x)
        return self.segmenter(bottleneck, skip_features)