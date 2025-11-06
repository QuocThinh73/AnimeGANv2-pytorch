import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()            
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding='same', bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
    def forward(self, x):
        return self.block(x)

class DSConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding='same', groups=channels, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv_block = ConvBlock(channels, channels, kernel_size=1, stride=1)
        
    def forward(self, x):
        x = self.block(x)
        x = self.conv_block(x)
        return x
    
class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.dsconv1 = DSConv(in_channels)
        
    def forward(self, x):
        pass

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        return x
