import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=in_features,
                      out_channels=in_features, kernel_size=3),
            nn.InstanceNorm2d(num_features=in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=in_features,
                      out_channels=in_features, kernel_size=3),
            nn.InstanceNorm2d(num_features=in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, num_residual_blocks: int = 9):
        super().__init__()

        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            layers += [
                nn.Conv2d(in_channels=in_features, out_channels=out_features,
                          kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(num_features=out_features),
                nn.ReLU(inplace=True),
            ]

            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(num_residual_blocks):
            layers += [ResnetBlock(in_features=in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features,
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(num_features=out_features),
                nn.ReLU(inplace=True),
            ]

            in_features = out_features
            out_features = in_features // 2

        # Output layer
        layers += [
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)
