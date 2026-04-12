import torch
import torch.nn as nn
from .vgg11 import VGG11Backbone

class UNetDecoder(nn.Module):
    """Upsample from 7x7 bottleneck; `features` are pre-pool maps (224..14) from the backbone."""

    def __init__(self, num_classes=3):
        super(UNetDecoder, self).__init__()
        # trimap: 3 classes (bg / fg / boundary)

        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = nn.Sequential(
            # 512 from upsample + 512 skip -> 1024
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),  # concat skip
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # concat skip
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # concat skip
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # concat skip
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x, features):
        x = self.up5(x)
        x = torch.cat([x, features[4]], dim=1)
        x = self.dec5(x)

        x = self.up4(x)
        x = torch.cat([x, features[3]], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, features[2]], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, features[1]], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, features[0]], dim=1)
        x = self.dec1(x)

        return self.final_conv(x)

class VGG11UNet(nn.Module):
    def __init__(self, num_classes=3, freeze_backbone=False):
        super(VGG11UNet, self).__init__()
        self.backbone = VGG11Backbone()
        self.segmenter = UNetDecoder(num_classes)

        # for transfer-learning ablations (e.g. frozen encoder)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        bottleneck, skips = self.backbone(x)
        return self.segmenter(bottleneck, skips)