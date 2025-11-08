import torch
import torch.nn as nn
from typing import Literal
from models import VGG16Features, VGG19Features
from utils.image_processing import rgb_to_gray, rgb_to_yuv, gram_matrix


class ContentLoss(nn.Module):
    def __init__(self, lambda_con: float, backbone: Literal["vgg16", "vgg19"], device=None):
        super().__init__()
        self.lambda_con = float(lambda_con)
        self.l1_loss = nn.L1Loss()
        if backbone == "vgg16":
            self.vgg = VGG16Features()
        elif backbone == "vgg19":
            self.vgg = VGG19Features()
        if device is not None:
            self.vgg = self.vgg.to(device)

    def forward(self, fake_anime: torch.Tensor, real_photo: torch.Tensor) -> torch.Tensor:
        fake_anime_features = self.vgg(fake_anime)
        real_photo_features = self.vgg(real_photo)
        return self.lambda_con * self.l1_loss(fake_anime_features, real_photo_features)


class GrayscaleStyleLoss(nn.Module):
    def __init__(self, lambda_gra: float, backbone: Literal["vgg16", "vgg19"], device=None):
        super().__init__()
        self.lambda_gra = float(lambda_gra)
        self.l1_loss = nn.L1Loss()
        if backbone == "vgg16":
            self.vgg = VGG16Features()
        elif backbone == "vgg19":
            self.vgg = VGG19Features()
        if device is not None:
            self.vgg = self.vgg.to(device)

    def forward(self, fake_anime: torch.Tensor, real_anime: torch.Tensor) -> torch.Tensor:
        fake_anime_gray = rgb_to_gray(fake_anime)
        real_anime_gray = rgb_to_gray(real_anime)

        fake_anime_gray_features = self.vgg(fake_anime_gray)
        real_anime_gray_features = self.vgg(real_anime_gray)

        return self.lambda_gra * self.l1_loss(
            gram_matrix(fake_anime_gray_features),
            gram_matrix(real_anime_gray_features)
        )


class ColorReconstructionLoss(nn.Module):
    def __init__(self, lambda_col: float):
        super().__init__()
        self.lambda_col = float(lambda_col)
        self.l1_loss = nn.L1Loss()
        self.huber_loss = nn.HuberLoss()

    def forward(self, fake_anime_rgb: torch.Tensor, real_photo_rgb: torch.Tensor) -> torch.Tensor:
        fake_anime_y, fake_anime_u, fake_anime_v = rgb_to_yuv(
            fake_anime_rgb)
        real_photo_y, real_photo_u, real_photo_v = rgb_to_yuv(
            real_photo_rgb)
        loss_y = self.l1_loss(fake_anime_y, real_photo_y)
        loss_u = self.huber_loss(fake_anime_u, real_photo_u)
        loss_v = self.huber_loss(fake_anime_v, real_photo_v)
        return self.lambda_col * (loss_y + loss_u + loss_v)
