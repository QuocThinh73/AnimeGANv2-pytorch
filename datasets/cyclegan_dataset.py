import os
import glob
import random
from PIL import Image
from torch.utils.data import Dataset
from .transforms import build_transforms


class CycleGANDataset(Dataset):
    def __init__(self, photo_root: str, anime_root: str, train: bool = True, image_size: int = 256):
        self.photo_paths = self._list_images(photo_root)
        if not self.photo_paths:
            raise RuntimeError(f"No images found in {photo_root}")
        
        self.anime_paths = self._list_images(anime_root)
        if not self.anime_paths:
            raise RuntimeError(f"No images found in {anime_root}")
        
        self.transforms = build_transforms(train, image_size)
        self.length = max(len(self.photo_paths), len(self.anime_paths))
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        photo_path = self.photo_paths[index % len(self.photo_paths)]
        anime_path = random.choice(self.anime_paths)
        
        photo_image = Image.open(photo_path).convert("RGB")
        anime_image = Image.open(anime_path).convert("RGB")
        
        photo_image = self.transforms(photo_image)
        anime_image = self.transforms(anime_image)
        
        return {
            "photo_image": photo_image,
            "anime_image": anime_image,
            "photo_path": photo_path,
            "anime_path": anime_path
        }
    
    def _list_images(self, root: str):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Invalid directory: {root}")
        
        files = glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)
        return sorted(files)
    