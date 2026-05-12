import torch.nn as nn
from .infaware_fff import InfAwareFFF

# FFF with CNN backbone
class CNNIFFF(nn.Module):
    def __init__(self, in_channels: int, in_features: int, leaf_width: int, 
                 out_features: int, depth: int, sig_alpha: float, 
                 inf_sig_alpha: float, scale: int = 8):
        super(CNNIFFF, self).__init__()
        self.leaf_width = leaf_width
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
        self.classifier = InfAwareFFF(scale * 2 * (in_features // 16), leaf_width, 
                                      out_features, depth, sig_alpha, inf_sig_alpha) # Adjust in_features accordingly

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def get_config(self):
        return self.classifier.get_config() | {"scale": self.scale}
