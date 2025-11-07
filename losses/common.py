import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    def __init__(self, lambda_adv: float):
        super().__init__()
        self.lambda_adv = float(lambda_adv)
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target_label: float) -> torch.Tensor:
        target_label = torch.full_like(pred, target_label)
        return self.lambda_adv * self.mse(pred, target_label)


class TotalVariationLoss(nn.Module):
    def __init__(self, lambda_tv: float):
        super().__init__()
        self.lambda_tv = float(lambda_tv)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        dx = image[:, :, 1:, :] - image[:, :, :-1, :]
        dy = image[:, :, :, 1:] - image[:, :, :, :-1]
        dx = dx[:, :, :, :-1]
        dy = dy[:, :, :-1, :]
        tv = torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-6).mean()
        return self.lambda_tv * tv
