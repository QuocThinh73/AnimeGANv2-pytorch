import torch
import torch.nn as nn


class CycleConsistencyLoss(nn.Module):
    def __init__(self, lambda_cyc: float):
        super().__init__()
        self.lambda_cyc = float(lambda_cyc)
        self.l1_loss = nn.L1Loss()

    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        return self.lambda_cyc * self.l1_loss(reconstructed, original)


class IdentityLoss(nn.Module):
    def __init__(self, lambda_idt: float):
        super().__init__()
        self.lambda_idt = float(lambda_idt)
        self.l1_loss = nn.L1Loss()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.lambda_idt * self.l1_loss(output, target)
