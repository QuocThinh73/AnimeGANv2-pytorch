import argparse
import torch
from trainers import build_trainer
from datasets import Photo2AnimeDataset
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True,
                   choices=["cyclegan", "animegan"])

    # Data
    p.add_argument("--photo_root", type=str, required=True)
    p.add_argument("--anime_style_root", type=str, required=True)
    p.add_argument("--anime_smooth_root", type=str, default=None)
    p.add_argument("--image_size", type=int, default=256)

    # Train
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=200)

    # Optimizer
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--decay_epoch", type=int, default=100)

    # Losses
    p.add_argument("--lambda_cyc", type=float, default=10.0)
    p.add_argument("--lambda_idt", type=float, default=5.0)

    # IO
    p.add_argument("--out_dir", type=str, default="output")
    p.add_argument("--save_every", type=int, default=10)

    # Random seed
    p.add_argument("--seed", type=int, default=42)

    # Resume
    p.add_argument("--resume", action="store_true")
    p.add_argument("--start_epoch", type=int, default=0)
    p.add_argument("--ckpt_dir", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dataset = Photo2AnimeDataset(
        model=args.model,
        photo_root=args.photo_root,
        anime_style_root=args.anime_style_root,
        anime_smooth_root=args.anime_smooth_root,
        train=True,
        image_size=args.image_size
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    trainer = build_trainer(args, loader)
    trainer.run()


if __name__ == "__main__":
    main()
