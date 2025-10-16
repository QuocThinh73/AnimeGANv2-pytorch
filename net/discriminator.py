import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.ops import *


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, ch=64, n_dis=3, sn=True):
        """
        Args:
            in_channels: number of input channels
            ch: base channels
            n_dis: numer of stage
            sn: turn on/off spectral normalization
        """
        super().__init__()
        assert n_dis >= 2, "n_dis must >= 2 to have conv_0 and at least 1 block"

        base = ch // 2
        prev_c = base

        layers = nn.ModuleList()

        # conv_0
        layers.append(Convolution(
            in_channels, base, kernel=3, stride=1, pad=1,
            use_bias=False, spectral_normalization=sn
        ))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        for i in range(1, n_dis):
            out_c1 = base * (2 ** i)
            layers.append(Convolution(
                prev_c, out_c1, kernel=3, stride=2, pad=1,
                use_bias=False, spectral_normalization=sn
            ))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_c = out_c1

            out_c2 = base * (2 ** (i + 1))
            layers.append(Convolution(
                prev_c, out_c2, kernel=3, stride=1, pad=1,
                use_bias=False, spectral_normalization=sn
            ))
            layers.append(InstanceNorm(out_c2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_c = out_c2

        # last conv
        layers.append(Convolution(
            prev_c, prev_c, kernel=3, stride=1, pad=1,
            use_bias=False, spectral_normalization=sn
        ))
        layers.append(InstanceNorm(prev_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.features = nn.Sequential(*layers)

        self.logit = Convolution(
            prev_c, 1, kernel=3, stride=1, pad=1,
            use_bias=False, spectral_normalization=sn
        )

        init_weight(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.logit(x)
        return x
