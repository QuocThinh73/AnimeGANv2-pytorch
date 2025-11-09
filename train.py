import torch
from trainers import build_trainer
from datasets import Photo2AnimeDataset
from torch.utils.data import DataLoader
from utils.arg_parser import ArgsParser


def main():
    args = ArgsParser().parse()

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
