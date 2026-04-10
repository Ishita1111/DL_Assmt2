"""
experiments/utils.py
Shared helpers used across all Section 2 experiment tasks.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import wandb
from models.layers import CustomDropout


# ---------------------------------------------------------------------------
# Backbone variants (used by tasks 2.1, 2.2)
# ---------------------------------------------------------------------------

class VGG11Backbone_WithBN(nn.Module):
    """VGG11 backbone WITH BatchNorm."""
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(2, 2)
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
        )
        self.pool5 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        x = self.pool4(self.enc4(x))
        x = self.pool5(self.enc5(x))
        return x


class VGG11Backbone_NoBN(nn.Module):
    """VGG11 backbone WITHOUT BatchNorm."""
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(2, 2)
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.pool5 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        x = self.pool4(self.enc4(x))
        x = self.pool5(self.enc5(x))
        return x


class SimpleClassifier(nn.Module):
    """Backbone + classification head wrapper."""
    def __init__(self, backbone, num_classes=37):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), CustomDropout(0.5),
            nn.Linear(4096, 4096),         nn.ReLU(True), CustomDropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------------
# Hook utility
# ---------------------------------------------------------------------------

def register_hook(module):
    """
    Registers a forward hook on a given module.
    Returns (handle, store) where store['output'] is populated after forward.
    """
    store = {}
    def hook_fn(mod, inp, out):
        store['output'] = out.detach().cpu()
    handle = module.register_forward_hook(hook_fn)
    return handle, store


# ---------------------------------------------------------------------------
# Generic train / val epoch helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for images, labels, _, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def val_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, labels, _, _ in loader:
            images, labels = images.to(device), labels.to(device)
            total += criterion(model(images), labels).item()
    return total / len(loader)


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")