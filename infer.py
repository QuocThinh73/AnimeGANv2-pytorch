import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from models import AnimeGANGenerator, CycleGANGenerator


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_file", type=str)
    parser.add_argument("ckpt_file", type=str)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="output/infer")

    return parser.parse_args()


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


def infer(image_file, ckpt_file, output_dir, image_size=256, device=None):
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

    output_image = infer(
        image_file=args.image_file,
        ckpt_file=args.ckpt_file,
        output_dir=args.output_dir,
        image_size=args.image_size,
    )
    output_image_name = os.path.basename(args.image_file) + "_" + os.path.basename(
        args.ckpt_file).split(".")[0].split("_")[1:] + ".png"
    save_image(output_image, os.path.join(args.output_dir, output_image_name))


if __name__ == "__main__":
    main()
