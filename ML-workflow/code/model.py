import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=9):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(
            128 * 3 * 3, 256
        )  # assuming 28x28 input -> after pooling x3 -> 3x3
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # 28->14
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 14->7
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # 7->3
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
