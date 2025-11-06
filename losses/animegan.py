import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    def __init__(self, lambda_adv: float = 300.0):
        super().__init__()
        self.lambda_adv = float(lambda_adv)
        self.mse = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target_label: float) -> torch.Tensor:
        target_label = torch.full_like(pred, target_label)
        return self.lambda_adv * self.mse(pred, target_label)
    
    
class ContentLoss(nn.Module):
    def __init__(self, lambda_con: float = 1.5):
        super().__init__()
        self.lambda_con = float(lambda_con)
    
    def forward(self):
        pass
    
class GrayscaleStyleLoss(nn.Module):
    def __init__(self, lambda_gra: float = 10.0):
        super().__init__()
        self.lambda_gra = float(lambda_gra)
    
    
    def forward(self):
        pass
    
class ColorReconstructionLoss(nn.Module):
    def __init__(self, lambda_col: float = 10.0):
        super().__init__()
        self.lambda_col = float(lambda_col)
        self.l1_loss = nn.L1Loss()
        self.huber_loss = nn.HuberLoss()
    
    def forward(self, generated_photo_rgb: torch.Tensor, real_photo_rgb: torch.Tensor) -> torch.Tensor:
        generated_yuv = self._rgb_to_yuv(generated_photo_rgb)
        real_yuv = self._rgb_to_yuv(real_photo_rgb)
        loss_y = self.l1_loss(generated_yuv[:, 0:1], real_yuv[:, 0:1])
        loss_u = self.huber_loss(generated_yuv[:, 1:2], real_yuv[:, 1:2], delta=1.0)
        loss_v = self.huber_loss(generated_yuv[:, 2:3], real_yuv[:, 2:3], delta=1.0)
        return self.lambda_col * (loss_y + loss_u + loss_v)
    
    def _rgb_to_yuv(self, image_rgb: torch.Tensor) -> torch.Tensor:
        r, g, b = image_rgb[:, 0:1], image_rgb[:, 1:2], image_rgb[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v = 0.615 * r - 0.51499 * g - 0.10001 * b
        return y, u, v
    
