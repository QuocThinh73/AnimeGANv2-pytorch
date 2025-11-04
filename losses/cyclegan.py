import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target_label: float) -> torch.Tensor:
        target_label = torch.full_like(pred, target_label)
        return self.mse(pred, target_label)


class CycleConsistencyLoss(nn.Module):
    def __init__(self, lambda_cyc: float = 10.0):
        super().__init__()
        self.lambda_cyc = float(lambda_cyc)
        self.l1_loss = nn.L1Loss()

    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        return self.lambda_cyc * self.l1_loss(reconstructed, original)


class IdentityLoss(nn.Module):
    def __init__(self, lambda_idt: float = 5.0):
        super().__init__()
        self.lambda_idt = float(lambda_idt)
        self.l1_loss = nn.L1Loss()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.lambda_idt * self.l1_loss(output, target)
