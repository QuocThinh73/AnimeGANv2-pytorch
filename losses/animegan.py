import torch
import torch.nn as nn
from typing import Literal
from models import VGG16Features, VGG19Features


class AdversarialLoss(nn.Module):
    def __init__(self, lambda_adv: float = 300.0):
        super().__init__()
        self.lambda_adv = float(lambda_adv)
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target_label: float) -> torch.Tensor:
        target_label = torch.full_like(pred, target_label)
        return self.lambda_adv * self.mse(pred, target_label)


class ContentLoss(nn.Module):
    def __init__(self, lambda_con: float = 1.5, backbone: Literal["vgg16", "vgg19"] = "vgg16"):
        super().__init__()
        self.lambda_con = float(lambda_con)
        self.l1_loss = nn.L1Loss()
        if backbone == "vgg16":
            self.vgg = VGG16Features()
        elif backbone == "vgg19":
            self.vgg = VGG19Features()

    def forward(self, fake_anime: torch.Tensor, real_photo: torch.Tensor) -> torch.Tensor:
        fake_anime_features = self.vgg(fake_anime)
        real_photo_features = self.vgg(real_photo)
        return self.lambda_con * self.l1_loss(fake_anime_features, real_photo_features)


class GrayscaleStyleLoss(nn.Module):
    def __init__(self, lambda_gra: float = 10.0, backbone: Literal["vgg16", "vgg19"] = "vgg16"):
        super().__init__()
        self.lambda_gra = float(lambda_gra)
        self.l1_loss = nn.L1Loss()
        if backbone == "vgg16":
            self.vgg = VGG16Features()
        elif backbone == "vgg19":
            self.vgg = VGG19Features()

    def forward(self, fake_anime: torch.Tensor, real_anime: torch.Tensor) -> torch.Tensor:
        fake_anime_gray = self._rgb_to_gray(fake_anime)
        real_anime_gray = self._rgb_to_gray(real_anime)

        fake_anime_gray_features = self.vgg(fake_anime_gray)
        real_anime_gray_features = self.vgg(real_anime_gray)

        return self.lambda_gra * self.l1_loss(
            self._gram_matrix(fake_anime_gray_features),
            self._gram_matrix(real_anime_gray_features)
        )

    def _rgb_to_gray(self, image_rgb: torch.Tensor) -> torch.Tensor:
        r = image_rgb[:, 0:1, :, :]
        g = image_rgb[:, 1:2, :, :]
        b = image_rgb[:, 2:3, :, :]

        gray = 0.299 * r + 0.587 * g + 0.114 * b
        gray = gray.expand(-1, 3, -1, -1).contiguous()
        return gray

    def _gram_matrix(self, image: torch.Tensor) -> torch.Tensor:
        # https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html#style-loss
        n, c, h, w = image.size()
        features = image.view(n, c, h * w)
        G = torch.bmm(features, features.transpose(1, 2)) / (c * h * w)
        return G


class ColorReconstructionLoss(nn.Module):
    def __init__(self, lambda_col: float = 10.0):
        super().__init__()
        self.lambda_col = float(lambda_col)
        self.l1_loss = nn.L1Loss()
        self.huber_loss = nn.HuberLoss()

    def forward(self, fake_anime_rgb: torch.Tensor, real_photo_rgb: torch.Tensor) -> torch.Tensor:
        fake_anime_y, fake_anime_u, fake_anime_v = self._rgb_to_yuv(
            fake_anime_rgb)
        real_photo_y, real_photo_u, real_photo_v = self._rgb_to_yuv(
            real_photo_rgb)
        loss_y = self.l1_loss(fake_anime_y, real_photo_y)
        loss_u = self.huber_loss(fake_anime_u, real_photo_u)
        loss_v = self.huber_loss(fake_anime_v, real_photo_v)
        return self.lambda_col * (loss_y + loss_u + loss_v)

    def _rgb_to_yuv(self, image_rgb: torch.Tensor) -> torch.Tensor:
        r, g, b = image_rgb[:, 0:1], image_rgb[:, 1:2], image_rgb[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v = 0.615 * r - 0.51499 * g - 0.10001 * b
        return y, u, v
