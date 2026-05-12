import torch.nn as nn

# CNN 
class CNN(nn.Module):
    def __init__(self, in_channels: int, in_features: int, out_features: int, 
                 width: int, scale: int = 8):
        super(CNN, self).__init__()
        self.width = width
        self.in_channels = in_channels
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, scale, kernel_size=3, stride=1, padding=1), # Input: RGB images
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(scale, scale * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(scale * 2 * (in_features // 16), width), # Adjust in_features accordingly
            nn.ReLU(),
            nn.Linear(width, out_features),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
