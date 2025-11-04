from typing import Literal
from .cyclegan_trainer import CycleGANTrainer
from .animegan_trainer import AnimeGANTrainer


def build_trainer(args, loader):
    if args.model == "cyclegan":
        return CycleGANTrainer(args, loader)
    elif args.model == "animegan":
        return AnimeGANTrainer(args, loader)
    else:
        raise ValueError(
            f"Không hỗ trợ mô hình {args.model}. Chỉ hỗ trợ mô hình CycleGAN và AnimeGAN.")
