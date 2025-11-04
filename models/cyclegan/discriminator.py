import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        # Convolutional blocks
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=64,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ]

        layers += [
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ]

        layers += [
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ]

        layers += [
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=4, padding=1),
            nn.InstanceNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ]

        # FCN classification layer
        layers += [nn.Conv2d(in_channels=512, out_channels=1,
                             kernel_size=4, padding=1)]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
