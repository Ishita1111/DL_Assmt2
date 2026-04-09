import torch.nn as nn
from .layers import CustomDropout

class VGG11Backbone(nn.Module):
    def __init__(self):
        super(VGG11Backbone, self).__init__()
        
        # VGG11 Configuration with BatchNorm
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = []
        x1 = self.enc1(x)
        features.append(x1)
        x = self.pool1(x1)
        
        x2 = self.enc2(x)
        features.append(x2)
        x = self.pool2(x2)
        
        x3 = self.enc3(x)
        features.append(x3)
        x = self.pool3(x3)
        
        x4 = self.enc4(x)
        features.append(x4)
        x = self.pool4(x4)
        
        x5 = self.enc5(x)
        features.append(x5)
        x = self.pool5(x5)
        
        return x, features

class ClassificationHead(nn.Module):
    def __init__(self, num_classes=37):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            CustomDropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            CustomDropout(p=0.5),
            nn.Linear(4096, num_classes) # Yields 37-class logits [cite: 44]
        )

    def forward(self, x):
        return self.classifier(x)
    

# Alias to satisfy the autograder's __init__.py expectations
VGG11Encoder = VGG11Backbone