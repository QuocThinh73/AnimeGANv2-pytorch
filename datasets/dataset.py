import os
import glob
import random
from typing import Literal, List
from PIL import Image
from torch.utils.data import Dataset
from .transforms import build_transforms


class Photo2AnimeDataset(Dataset):
    def __init__(self, mode: Literal["cyclegan", "animegan"], photo_root: str, anime_style_root: str, anime_smooth_root: str | None = None, train: bool = True, image_size: int = 256):
        self.mode = mode
        if self.mode not in ["cyclegan", "animegan"]:
            raise ValueError(
                f"Không hỗ trợ mô hình {self.mode}. Chỉ hỗ trợ mô hình CycleGAN và AnimeGAN.")

        self.photo_paths = self._list_images(photo_root)
        if not self.photo_paths:
            raise RuntimeError(f"Không tìm thấy ảnh trong {photo_root}")

        self.anime_style_paths = self._list_images(anime_style_root)
        if not self.anime_style_paths:
            raise RuntimeError(f"Không tìm thấy ảnh trong {anime_style_root}")

        if self.mode == "animegan":
            if anime_smooth_root is None:
                raise ValueError(
                    "Không được để trống anime_smooth_root khi mode là AnimeGAN.")
            self.anime_smooth_paths = self._list_images(anime_smooth_root)
            if not self.anime_smooth_paths:
                raise RuntimeError(
                    f"Không tìm thấy ảnh trong {anime_smooth_root}")

        self.transforms = build_transforms(train, image_size)

        if self.mode == "cyclegan":
            self.length = max(len(self.photo_paths),
                              len(self.anime_style_paths))
        elif self.mode == "animegan":
            self.length = len(self.photo_paths)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        # Photo
        photo_path = self.photo_paths[index % len(self.photo_paths)]
        photo_image = Image.open(photo_path).convert("RGB")
        photo_image = self.transforms(photo_image)

        # Anime Style
        anime_style_path = random.choice(self.anime_style_paths)
        anime_style_image = Image.open(anime_style_path).convert("RGB")
        anime_style_image = self.transforms(anime_style_image)

        if self.mode == "cyclegan":
            return {
                "photo_image": photo_image,
                "anime_style_image": anime_style_image,
                "photo_path": photo_path,
                "anime_style_path": anime_style_path
            }

        elif self.mode == "animegan":
            # Anime Smooth
            anime_smooth_path = random.choice(self.anime_smooth_paths)
            anime_smooth_image = Image.open(anime_smooth_path).convert("RGB")
            anime_smooth_image = self.transforms(anime_smooth_image)

            return {
                "photo_image": photo_image,
                "anime_style_image": anime_style_image,
                "anime_smooth_image": anime_smooth_image,
                "photo_path": photo_path,
                "anime_style_path": anime_style_path,
                "anime_smooth_path": anime_smooth_path
            }

    def _list_images(self, root: str) -> List[str]:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Invalid directory: {root}")

        files = glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)
        return sorted(files)
