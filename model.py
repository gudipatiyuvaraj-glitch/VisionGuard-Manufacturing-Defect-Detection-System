"""
model.py - Neural network architectures: Custom CNN & Pretrained ResNet-18.
"""
import torch
import torch.nn as nn
from torchvision import models

class CustomCNN(nn.Module):
    """Custom baseline CNN: Conv2D -> BatchNorm -> ReLU -> MaxPool blocks."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_resnet18(num_classes: int = 2, fine_tune_layers: int = 2) -> nn.Module:
    """
    Loads ResNet-18 with ImageNet weights, freezes early layers,
    and replaces classification head.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last N blocks (e.g., layer4 and layer3)
    if fine_tune_layers >= 1:
        for param in model.layer4.parameters():
            param.requires_grad = True
    if fine_tune_layers >= 2:
        for param in model.layer3.parameters():
            param.requires_grad = True

    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model