import torch
import math
from torch import nn
from typing import Optional

class FF(nn.Module):
    def __init__(self, in_features: int, width: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.width = width
        self.out_features = out_features
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(in_features, width)
        self.fc2 = nn.Linear(width, out_features)

    def forward(self, x):
        x = x.view(len(x), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

    def get_config(self):
        return {
                'in_features': self.in_features,
                'width': self.width,
                'out_features': self.out_features,
                }
