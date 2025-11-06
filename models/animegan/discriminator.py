import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN

class Discriminator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        
        # Convolutional blocks
        layers = [
            SN(nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ]
        
        layers += [
            SN(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            SN(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)),
            nn.GroupNorm(num_groups=1, num_channels=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ]
        
        layers += [
            SN(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            SN(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)),
            nn.GroupNorm(num_groups=1, num_channels=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ]
        
        layers += [
            SN(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)),
            nn.GroupNorm(num_groups=1, num_channels=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ]
        
        # FCN classification layer
        layers += [
            SN(nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1)),
        ]
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)