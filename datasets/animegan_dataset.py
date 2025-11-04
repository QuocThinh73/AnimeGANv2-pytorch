import os
import glob
import random
from PIL import Image
from torch.utils.data import Dataset
from .transforms import build_transforms


class AnimeGANDataset(Dataset):
    def __init__(self, photo_root: str, anime_style_root: str, anime_smooth_root: str, train: bool = True, image_size: int = 256):
        self.photo_paths = self._list_images(photo_root)
        if not self.photo_paths:
            raise RuntimeError(f"No images found in {photo_root}")
        
        self.anime_style_paths = self._list_images(anime_style_root)
        if not self.anime_style_paths:
            raise RuntimeError(f"No images found in {anime_style_root}")
        
        self.anime_smooth_paths = self._list_images(anime_smooth_root)
        if not self.anime_smooth_paths:
            raise RuntimeError(f"No images found in {anime_smooth_root}")
        
        self.transforms = build_transforms(train, image_size)
        self.length = len(self.photo_paths)
    
    def __length__(self):
        return self.length
    
    def __getitem__(self, index: int):
        photo_path = self.photo_paths[index % len(self.photo_paths)]
        anime_style_path = random.choice(self.anime_style_paths)
        anime_smooth_path = random.choice(self.anime_smooth_paths)
        
        photo_image = Image.open(photo_path).convert("RGB")
        anime_style_image = Image.open(anime_style_path).convert("RGB")
        anime_smooth_image = Image.open(anime_smooth_path).convert("RGB")
        
        photo_image = self.transforms(photo_image)
        anime_style_image = self.transforms(anime_style_image)
        anime_smooth_image = self.transforms(anime_smooth_image)
        
        return {
            "photo_image": photo_image,
            "anime_style_image": anime_style_image,
            "anime_smooth_image": anime_smooth_image,
            "photo_path": photo_path,
            "anime_style_path": anime_style_path,
            "anime_smooth_path": anime_smooth_path
        }
    
    def _list_images(self, root: str):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Invalid directory: {root}")
        
        files = glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)
        return sorted(files)