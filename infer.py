import torch
from torchvision.utils import save_image
from datasets.dataset import Photo2AnimeDataset
from torch.utils.data import DataLoader
from models import AnimeGANGenerator
from datasets.transforms import build_transforms

# 1) Data (giống train: Resize/Crop/Normalize [-1,1])
ds = Photo2AnimeDataset(
    model="animegan",
    photo_root="data/train_photo",
    anime_style_root="data/Shinkai/style",
    anime_smooth_root="data/Shinkai/smooth",
    train=True, image_size=256
)
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

# 2) Model
device = "cuda" if torch.cuda.is_available() else "cpu"
G = AnimeGANGenerator().to(device).eval()

# 3) One batch
with torch.no_grad():
    batch = next(iter(loader))
    photo = batch["photo_image"].to(device)
    fake = G(photo)

# 4) Denorm [-1,1] -> [0,1]


def denorm(x): return (x * 0.5 + 0.5).clamp(0, 1)


save_image(denorm(photo)[:4], "debug_photo.png", nrow=2)
save_image(denorm(fake)[:4],  "debug_fake.png",  nrow=2)
print("Saved debug_photo.png & debug_fake.png")
