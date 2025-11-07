import torch


def rgb_to_gray(image_rgb: torch.Tensor) -> torch.Tensor:
    r = image_rgb[:, 0:1, :, :]
    g = image_rgb[:, 1:2, :, :]
    b = image_rgb[:, 2:3, :, :]

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray = gray.expand(-1, 3, -1, -1).contiguous()
    return gray
