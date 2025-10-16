import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

#############################
# Weight init
#############################


def init_weight(module: nn.Module, mean=0.0, std=0.02):
    """Apply DCGAN-style normal init to Conv/ConvT/Linear + zeros to bias"""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, mean=mean, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

#############################
# Activation Functions
#############################


def flatten(x: torch.Tensor) -> torch.Tensor:
    return torch.flatten(x, start_dim=1)


def lrelu(x: torch.Tensor, alpha=0.2) -> torch.Tensor:
    return F.leaky_relu(x, negative_slope=alpha)


def relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x, inplace=True)


def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)

#############################
# Normalization Function
#############################


def l2_norm(v: torch.Tensor, eps=1e-12) -> torch.Tensor:
    return v / (torch.sqrt(torch.sum(v * v, dim=-1, keepdim=True)) + eps)


class InstanceNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class LayerNormChannels(nn.Module):
    """
    LayerNorm over channel dimension only
    Input expected: NCHW
    This normalizes over C with learned affine params.
    """

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x_hat = x_hat * self.weight + self.bias
        return x_hat


class BatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5, affine=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps=eps, momentum=(1.0 - momentum), affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)

#############################
# Helper: padding calculator
#############################


def _compute_pads(kernel, stride, pad):
    if (kernel - stride) % 2 == 0:
        pad_top = pad_bottom = pad_left = pad_right = pad
    else:
        pad_top = pad_left = pad
        pad_bottom = pad_right = kernel - stride - pad
    return pad_left, pad_right, pad_top, pad_bottom


def _apply_pad(x: torch.Tensor, pads, mode) -> torch.Tensor:
    pl, pr, pt, pb = pads
    if (pl, pr, pt, pb) == (0, 0, 0, 0):
        return x

    if mode == "reflect":
        return F.pad(x, (pl, pr, pt, pb), mode=mode)

    return F.pad(x, (pl, pr, pt, pb), mode="constant", value=0.0)

#############################
# Layers
#############################


class Convolution(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel=4,
            stride=2,
            pad=0,
            pad_type="zero",
            use_bias=True,
            spectral_normalization=False
    ):
        super().__init__()
        assert pad_type in {"zero", "reflect"}
        self.pads = _compute_pads(kernel, stride, pad)
        self.pad_type = pad_type
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                         stride=stride, padding=0, bias=use_bias)
        self.conv = spectral_norm(conv) if spectral_normalization else conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _apply_pad(
            x, self.pads, mode="reflect" if self.pad_type == "reflect" else "constant")
        return self.conv(x)


class Deconvolution(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel=4,
            stride=2,
            use_bias=True,
            spectral_normalization=False
    ):
        super().__init__()
        padding = max((kernel - stride) // 2, 0)
        deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=use_bias)
        self.deconv = spectral_norm(
            deconv) if spectral_normalization else deconv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)

#############################
# Residual Block
#############################


class ResBlock(nn.Module):
    def __init__(self, channels, use_bias=True):
        super().__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.in1 = nn.InstanceNorm2d(channels, affine=True, eps=1e-5)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.in2 = nn.InstanceNorm2d(channels, affine=True, eps=1e-5)

    def forward(self, x_init: torch.Tensor) -> torch.Tensor:
        x = self.pad1(x_init)
        x = self.conv1(x)
        x = self.in1(1)
        x = F.relu(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.in2(x)
        return x + x_init

#############################
# Loss Function
#############################


def L1_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(x - y))


def L2_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((x - y) ** 2)


def Huber_loss(x: torch.Tensor, y: torch.Tensor, delta=1.0) -> torch.Tensor:
    return F.smooth_l1_loss(x, y, beta=delta, reduction="mean")


def discriminator_loss(loss_func, real: torch.Tensor, gray: torch.Tensor, fake: torch.Tensor, real_blur: torch.Tensor) -> torch.Tensor:
    loss_func = loss_func.lower()
    if loss_func in {"wgan-gp", "wgan-lp"}:
        real_loss = -torch.mean(real)
        gray_loss = -torch.mean(gray)
        fake_loss = -torch.mean(fake)
        real_blur_loss = -torch.mean(real_blur)
    elif loss_func == "lsgan":
        real_loss = torch.mean((real - 1.0) ** 2)
        gray_loss = torch.mean(gray ** 2)
        fake_loss = torch.mean(fake ** 2)
        real_blur_loss = torch.mean(real_blur ** 2)
    elif loss_func in {"gan", "dragan"}:
        real_loss = F.binary_cross_entropy_with_logits(
            real, torch.ones_like(real))
        gray_loss = F.binary_cross_entropy_with_logits(
            gray, torch.ones_like(gray))
        fake_loss = F.binary_cross_entropy_with_logits(
            fake, torch.ones_like(fake))
        real_blur_loss = F.binary_cross_entropy_with_logits(
            real_blur, torch.ones_like(real_blur))
    elif loss_func == "hinge":
        real_loss = torch.mean(F.relu(1.0 - real))
        gray_loss = torch.mean(F.relu(1.0 + gray))
        fake_loss = torch.mean(F.relu(1.0 + fake))
        real_blur_loss = torch.mean(F.relu(1.0 + real_blur))
    else:
        raise ValueError(f"Unknown loss_func: {loss_func}")

    return real_loss + gray_loss + fake_loss + 0.1 * real_blur_loss


def generator_loss(loss_func, fake: torch.Tensor) -> torch.Tensor:
    loss_func = loss_func.lower()
    if loss_func in {"wgan-gp", "wgan-lp", "hinge"}:
        return -torch.mean(fake)

    if loss_func == "lsgan":
        return torch.mean((fake - 1.0) ** 2)

    if loss_func in {"gan", "dragan"}:
        return F.binary_cross_entropy_with_logits(fake, torch.ones_like(fake))

    raise ValueError(f"Unknown loss_func: {loss_func}")

#############################
# Style/Content Utilities
#############################


def gram(x: torch.Tensor) -> torch.Tensor:
    """x: (N, C, H, W) -> (N, C, C)"""
    n, c, h, w = x.shape
    x_flat = x.view(n, c, -1)  # (N, C, HW)
    gram_mat = torch.bnm(x_flat, x_flat.transpose(1, 2))  # (N, C, C)
    denom = (h * w * c)
    return gram_mat / float(denom)


def content_loss(vgg, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    real_feature_map = vgg(real)
    real_feature_map = getattr(
        real_feature_map, "conv4_3_no_activation", real_feature_map)

    fake_feature_map = vgg(fake)
    fake_feature_map = getattr(
        fake_feature_map, "conv4_3_no_activation", fake_feature_map)

    return L1_loss(real_feature_map, fake_feature_map)


def style_loss(style: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    return L1_loss(gram(style), gram(fake))


def content_style_loss(vgg, real: torch.Tensor, anime: torch.Tensor, fake: torch.Tensor):
    real_feature_map = vgg(real)
    real_feature_map = getattr(
        real_feature_map, "conv4_3_no_activation", real_feature_map)

    fake_feature_map = vgg(fake)
    fake_feature_map = getattr(
        fake_feature_map, "conv4_3_no_activation", fake_feature_map)

    anime = anime[: fake_feature_map.shape[0]]
    anime_feature_map = vgg(anime)
    anime_feature_map = getattr(
        anime_feature_map, "conv4_3_no_activation", anime_feature_map)

    c_loss = L1_loss(real_feature_map, fake_feature_map)
    s_loss = style_loss(anime_feature_map, fake_feature_map)
    return c_loss, s_loss

#############################
# Color Space Utilities
#############################


def rgb2yuv(rgb: torch.Tensor) -> torch.Tensor:
    """
    rgb in [-1, 1] -> yuv (channels first)
    Expect input is (N, 3, H, W)
    """
    x = (rgb + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    return torch.cat([y, u, v], dim=1)


def color_loss(content: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    content_yuv = rgb2yuv(content)
    fake_yuv = rgb2yuv(fake)
    y_loss = L1_loss(content_yuv[:, 0:1], fake_yuv[:, 0:1])
    u_loss = Huber_loss(content_yuv[:, 1:2], fake_yuv[:, 1:2], delta=1.0)
    v_loss = Huber_loss(content_yuv[:, 2:3], fake_yuv[:, 2:3], delta=1.0)
    return y_loss + u_loss + v_loss
