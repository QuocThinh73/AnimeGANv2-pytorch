import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from models import AnimeGANGenerator, CycleGANGenerator


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str,
                        choices=["single", "directory"], required=True)
    parser.add_argument("--image_file", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--ckpt_file", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="output/infer")

    args = parser.parse_args()

    if args.mode == "single" and not args.image_file:
        parser.error("--image_file is required when mode is 'single'.")
    if args.mode == "directory" and not args.image_dir:
        parser.error("--image_dir is required when mode is 'directory'.")

    return args


def load_generator(model_type, ckpt_path, device):
    if model_type == "animegan":
        generator = AnimeGANGenerator().to(device)
    elif model_type == "cyclegan":
        generator = CycleGANGenerator().to(device)
    else:
        raise ValueError()

    generator.load_state_dict(torch.load(ckpt_path, map_location=device))
    generator.eval()
    return generator


def preprocess_image(image_path, image_size=256):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def denormalize(tensor):
    return tensor * 0.5 + 0.5


def detect_model_type(ckpt_path):
    filename = os.path.basename(ckpt_path).lower()
    if "animegan" in filename:
        return "animegan"
    if "cyclegan" in filename:
        return "cyclegan"
    raise ValueError()


def infer(image_file, ckpt_file, image_size=256, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = detect_model_type(ckpt_file)
    generator = load_generator(model_type, ckpt_file, device)
    input_tensor = preprocess_image(image_file, image_size).to(device)

    with torch.no_grad():
        output_tensor = generator(input_tensor)
        output_tensor = denormalize(output_tensor)

    return output_tensor.cpu()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_stem = os.path.splitext(os.path.basename(args.ckpt_file))[0]

    if args.mode == "single":
        output_image = infer(
            image_file=args.image_file,
            ckpt_file=args.ckpt_file,
            image_size=args.image_size,
        )
        image_stem = os.path.splitext(os.path.basename(args.image_file))[0]
        output_image_name = f"{image_stem}_{ckpt_stem}.png"
        save_image(output_image, os.path.join(
            args.output_dir, output_image_name))
    else:
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted(
            [
                os.path.join(args.image_dir, f)
                for f in os.listdir(args.image_dir)
                if os.path.splitext(f.lower())[1] in supported_exts
            ]
        )

        if not image_paths:
            raise ValueError(
                f"No supported images found in directory: {args.image_dir}")

        for image_path in image_paths:
            output_image = infer(
                image_file=image_path,
                ckpt_file=args.ckpt_file,
                image_size=args.image_size,
            )
            image_stem = os.path.splitext(os.path.basename(image_path))[0]
            output_image_name = f"{image_stem}_{ckpt_stem}.png"
            save_image(output_image, os.path.join(
                args.output_dir, output_image_name))


if __name__ == "__main__":
    main()