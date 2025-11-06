import torch
import torch.nn as nn
import torchvision.models as models


class VGG19Features(nn.Module):
    def __init__(self, last_layer: int = 26):
        super().__init__()

        # VGG19 features
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        layers = list(vgg.features.children())[:last_layer]

        for name, layer in vgg.features.named_children():
            print(name, layer)
        self.features = nn.Sequential(*layers).eval()
        for param in self.features.parameters():
            param.requires_grad = False

        # ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406]).float()
        std = torch.tensor([0.229, 0.224, 0.225]).float()
        self.mean = mean.view(1, 3, 1, 1)
        self.std = std.view(1, 3, 1, 1)

    def forward(self, input: torch.Tensor):
        return self.features(self._normalize(input))

    @torch.no_grad()
    def _normalize(self, image: torch.Tensor):
        image = (image + 1.0) / 2.0
        return (image - self.mean) / self.std


if __name__ == "__main__":
    vgg = VGG19Features()
