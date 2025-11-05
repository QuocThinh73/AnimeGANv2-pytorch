import torch
import torch.nn as nn


def weights_init_normal(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
