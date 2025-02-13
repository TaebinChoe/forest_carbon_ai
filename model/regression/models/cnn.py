import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection (skip connection)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        return F.relu(out)

class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        
        # CNN feature extractor with residual blocks
        self.conv_layers = nn.Sequential(
            ResidualBlock(108, 64),
            ResidualBlock(64, 32),
            ResidualBlock(32, 16),
            nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        )
        
        # Fully Connected Regression Head
        self.fc = nn.Sequential(
            nn.Flatten(),  # (batch, 16*1*1) -> (batch, 16)
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)  # Output a single value for regression
        )
    
    def forward(self, x):
        x = self.conv_layers(x)  # Feature extraction
        x = self.fc(x)  # Regression head
        return x
