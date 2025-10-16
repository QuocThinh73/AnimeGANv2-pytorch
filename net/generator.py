import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.ops import *


def _reflect_pad_for_sep(kernel, stride):
    if kernel == 3 and stride == 1:
        return nn.ReflectionPad2d(1)

    if stride == 2:
        return nn.ReflectionPad2d((0, 1, 0, 1))  # (l, r, t, b)

    return nn.Identity()


class DepthwiseConv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            kernel_size=3,
            stride=1,
            multiplier=1,
            bias=False
    ):
        super().__init__()
        self.pad = _reflect_pad_for_sep(kernel_size, stride)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * multiplier,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=in_channels,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        return self.conv(x)


class SeperableConv2d(nn.Module):
    """
    Depthwise Conv -> Pointwise Conv -> Instance Norm -> LeakyReLU
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            bias=True,
            affine_norm=True
    ):
        super().__init__()
        self.depthwise = DepthwiseConv2d(
            in_channels, kernel_size, stride, multiplier=1, bias=False)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.instanceNorm = InstanceNorm(out_channels, affine=affine_norm)
        self.activateFunc = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.instanceNorm(x)
        x = self.activateFunc(x)
        return x


class ConvBlock(nn.Module):
    """
    Conv -> Instance Norm -> LeakyReLU
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            bias=True
    ):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1) if kernel_size == 3 else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, stride=stride,
                              kernel_size=kernel_size, padding=0, bias=bias)
        self.instanceNorm = InstanceNorm(out_channels, affine=True)
        self.activateFunc = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.conv(x)
        x = self.instanceNorm(x)
        x = self.activateFunc(x)
        return x


class InvertedResidualBlock(nn.Module):
    """
    I -> Conv -> DepthwiseConv -> Instance Norm -> LeakyReLU -> Conv -> Instance Norm -> add with I
    """

    def __init__(
            self,
            in_channels,
            expansion_ratio,
            out_channels,
            stride=1
    ):
        super().__init__()
        bottleNeck = int(round(expansion_ratio * in_channels))

        # Conv-Block (K1, S1, Cmid)
        self.expand = ConvBlock(in_channels, bottleNeck)

        # depthwise 3x3 + Instance Norm + LeakyReLU
        self.depthwise = nn.Sequential(
            DepthwiseConv2d(bottleNeck, kernel_size=3,
                            stride=1, multiplier=1, bias=False),
            InstanceNorm(bottleNeck, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # project 1x1 + Instance Norm
        self.project = nn.Sequential(
            nn.Conv2d(bottleNeck, out_channels, kernel_size=1,
                      stride=1, padding=0, bias=True),
            InstanceNorm(out_channels, affine=True)
        )

        self.use_residual = (in_channels == out_channels and stride == 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        return x + identity if self.use_residual else x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.sep = SeperableConv2d(
            in_channels, out_channels, kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0,
                          mode="bilinear", align_corners=False)
        x = self.sep(x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.sep = SeperableConv2d(
            in_channels, out_channels, kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=0.5,
                          mode="bilinear", align_corners=False)
        x = self.sep(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.b1_c1 = ConvBlock(in_channels, 64)
        self.b1_c2 = ConvBlock(64, 64)
        self.b1_sep_s2 = SeperableConv2d(64, 128, kernel_size=3, stride=2)
        self.b1_skip = DownSample(64, 128)

        self.b2_c1 = ConvBlock(128, 128)
        self.b2_sep = SeperableConv2d(128, 128, kernel_size=3, stride=1)
        self.b2_sep_s2 = SeperableConv2d(128, 256, kernel_size=3, stride=2)
        self.b2_skip = DownSample(128, 256)

        self.mid_in = ConvBlock(256, 256)
        self.mid_blocks = nn.Sequential(
            InvertedResidualBlock(256, 2.0, 256, stride=1),
            InvertedResidualBlock(256, 2.0, 256, stride=1),
            InvertedResidualBlock(256, 2.0, 256, stride=1),
            InvertedResidualBlock(256, 2.0, 256, stride=1),
            InvertedResidualBlock(256, 2.0, 256, stride=1),
            InvertedResidualBlock(256, 2.0, 256, stride=1),
            InvertedResidualBlock(256, 2.0, 256, stride=1),
            InvertedResidualBlock(256, 2.0, 256, stride=1)
        )
        self.mid_out = ConvBlock(256, 256)

        self.u2_up = UpSample(256, 128)
        self.u2_sep = SeperableConv2d(128, 128, kernel_size=3, stride=1)
        self.u2_c1 = ConvBlock(128, 128)

        self.u1_up = UpSample(128, 128)
        self.u1_c1 = ConvBlock(128, 64)
        self.u1_c2 = ConvBlock(64, 64)

        self.out_conv = nn.Conv2d(
            64, 3, kernel_size=1, stride=1, padding=0, bias=True)

        init_weight(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b1
        x = self.b1_c1(x)
        x = self.b1_c2(x)
        skip = self.b1_skip(x)
        x = self.b1_sep_s2(x) + skip  # residual downsample add

        # b2
        x = self.b2_c1(x)
        x = self.b2_sep(x)
        skip = self.b2_skip(x)
        x = self.b2_sep_s2(x) + skip

        # mid
        x = self.mid_in(x)
        x = self.mid_blocks(x)
        x = self.mid_out(x)

        # up path
        x = self.u2_up(x)
        x = self.u2_sep(x)
        x = self.u2_c1(x)

        x = self.u1_up(x)
        x = self.u1_c1(x)
        x = self.u1_c2(x)

        x = self.out_conv(x)
        x = torch.tanh(x)
        return x
