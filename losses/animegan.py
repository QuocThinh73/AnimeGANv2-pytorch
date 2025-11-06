import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    def __init__(self, lambda_adv: float = 300.0):
        super().__init__()
        self.lambda_adv = float(lambda_adv)
    
    def forward(self):
        pass
    
    
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
    
    def forward(self):
        pass
    
