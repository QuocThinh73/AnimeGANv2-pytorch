import torch


def rgb_to_gray(image_rgb: torch.Tensor) -> torch.Tensor:
    r = image_rgb[:, 0:1, :, :]
    g = image_rgb[:, 1:2, :, :]
    b = image_rgb[:, 2:3, :, :]

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray = gray.expand(-1, 3, -1, -1).contiguous()
    return gray

def rgb_to_yuv(image_rgb: torch.Tensor) -> torch.Tensor:
        r, g, b = image_rgb[:, 0:1], image_rgb[:, 1:2], image_rgb[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v = 0.615 * r - 0.51499 * g - 0.10001 * b
        return y, u, v
    
def gram_matrix(image: torch.Tensor) -> torch.Tensor:
        # https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html#style-loss
        n, c, h, w = image.size()
        features = image.view(n, c, h * w)
        G = torch.bmm(features, features.transpose(1, 2)) / (c * h * w)
        return G