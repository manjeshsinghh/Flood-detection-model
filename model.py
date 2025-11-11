"""Deep learning model for binary classification of flood-prone river basins."""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class FloodClassifier(nn.Module):
    """
    CNN-based binary classifier for flood-prone vs non-flood-prone river basins.
    
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        backbone: str = 'resnet18',
        dropout_rate: float = 0.5
    ):
    
        super(FloodClassifier, self).__init__()
        
        self.backbone_name = backbone
        
        # Load pretrained backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'vgg16':
            self.backbone = models.vgg16(pretrained=pretrained)
            num_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """Forward pass through the model."""
        # Extract features using backbone
        if self.backbone_name in ['resnet18', 'resnet34', 'resnet50']:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
        elif self.backbone_name == 'vgg16':
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
        elif self.backbone_name == 'mobilenet_v2':
            x = self.backbone.features(x)
            x = nn.AdaptiveAvgPool2d((1, 1))(x)
            x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        return x


class BasicCNN(nn.Module):
    
    
    def __init__(self, num_classes: int = 2):
        super(BasicCNN, self).__init__()
        
        # Simple convolutional layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224x224 -> 112x112
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112 -> 56x56
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56 -> 28x28
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SimpleCNN(nn.Module):
    
    
    def __init__(self, num_classes: int = 2):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(
    model_type: str = 'resnet18',
    num_classes: int = 2,
    pretrained: bool = True,
    dropout_rate: float = 0.5
) -> nn.Module:
    
    if model_type == 'basic_cnn':
        return BasicCNN(num_classes=num_classes)
    elif model_type == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes)
    else:
        return FloodClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            backbone=model_type,
            dropout_rate=dropout_rate
        )


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

