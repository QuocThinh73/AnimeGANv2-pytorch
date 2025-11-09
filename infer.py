import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from models import AnimeGANGenerator, CycleGANGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference script for AnimeGANv2 and CycleGAN")

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True,
                        choices=["animegan", "cyclegan"])
    parser.add_argument("--image_path", type=str, required=True,)
    parser.add_argument("--output_dir", type=str, required=True,)
    parser.add_argument("--image_size", type=int, default=256,)

    return parser.parse_args()


def load_generator(model_type, ckpt_path, device):
    if model_type == "animegan":
        generator = AnimeGANGenerator().to(device)
    elif model_type == "cyclegan":
        generator = CycleGANGenerator().to(device)

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


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = load_generator(args.model, args.ckpt_path, device)
    input_tensor = preprocess_image(
        args.image_path, args.image_size).to(device)
    with torch.no_grad():
        output_tensor = generator(input_tensor)
        output_tensor = denormalize(output_tensor)

    os.makedirs(args.output_dir, exist_ok=True)
    input_filename = os.path.basename(args.image_path)
    output_filename = f"{os.path.splitext(input_filename)[0]}_anime{os.path.splitext(input_filename)[1]}"
    output_path = os.path.join(args.output_dir, output_filename)

    save_image(output_tensor, output_path)


if __name__ == "__main__":
    main()
