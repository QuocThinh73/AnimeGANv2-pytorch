import os
import argparse
from typing import Tuple

import cv2
import numpy as np
import torch

#############################
# I/O helpers (RGB tensors)
#############################

def read_img(path: str, device: torch.device) -> torch.Tensor:
    """Read image with OpenCV, convert BGR->RGB, return float32 tensor (C,H,W) on device in range [0,255]."""
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img_rgb.astype(np.float32))  # (H,W,3)
    img = img.permute(2, 0, 1).to(device)               # (3,H,W)
    return img


def tensor_to_uint8_hwc(img: torch.Tensor) -> np.ndarray:
    """(C,H,W) float tensor in [0,255] -> uint8 (H,W,C) RGB numpy array"""
    x = img.detach().clamp(0, 255).to(torch.uint8)
    x = x.permute(1, 2, 0).cpu().numpy()
    return x

#############################
# Brightness utilities
#############################

def calculate_average_brightness(img_rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute average brightness and per-channel means for an RGB tensor (C,H,W) in [0,255].
    Uses luminance: 0.299*R + 0.587*G + 0.114*B
    Returns: (brightness, B_mean, G_mean, R_mean) to mirror original return order.
    """
    assert img_rgb.ndim == 3 and img_rgb.shape[0] == 3, "Expected (3,H,W) RGB tensor"
    R = img_rgb[0].mean()
    G = img_rgb[1].mean()
    B = img_rgb[2].mean()
    brightness = 0.299 * R + 0.587 * G + 0.114 * B
    return brightness, B, G, R


def adjust_brightness_from_src_to_dst(
    dst_rgb: torch.Tensor,
    src_rgb: torch.Tensor,
    save_path: str | None = None,
    show: bool | None = None,
    verbose: bool | None = None,
    mode: str = "luma",
) -> torch.Tensor:
    """Match brightness of dst to src using PyTorch.

    Args:
        dst_rgb: (3,H,W) float32 in [0,255]
        src_rgb: (3,H,W) float32 in [0,255]
        save_path: if provided, saves a side-by-side RGB comparison (src | dst | adjusted)
        show: if True, displays the comparison window via OpenCV
        verbose: if True, prints mean stats
        mode: 'luma' to scale by luminance ratio; 'channels' to scale each channel by per-channel ratios

    Returns:
        adjusted tensor (3,H,W) float32 in [0,255]
    """
    device = dst_rgb.device
    brightness_src, B1, G1, R1 = calculate_average_brightness(src_rgb)
    brightness_dst, B2, G2, R2 = calculate_average_brightness(dst_rgb)

    if verbose:
        print(f"Average brightness of source:  {brightness_src.item():.4f}")
        print(f"Average brightness of target:  {brightness_dst.item():.4f}")
        print(f"Brightness ratio (src/dst):    {(brightness_src/brightness_dst).item():.6f}")

    if mode == "luma":
        ratio = (brightness_src / (brightness_dst + 1e-12)).to(device)
        adjusted = dst_rgb * ratio
    elif mode == "channels":
        # Per-channel scaling to match channel-wise means
        r_ratio = (R1 / (R2 + 1e-12)).to(device)
        g_ratio = (G1 / (G2 + 1e-12)).to(device)
        b_ratio = (B1 / (B2 + 1e-12)).to(device)
        adjusted = torch.stack([
            dst_rgb[0] * r_ratio,  # R
            dst_rgb[1] * g_ratio,  # G
            dst_rgb[2] * b_ratio,  # B
        ], dim=0)
    else:
        raise ValueError("mode must be 'luma' or 'channels'")

    adjusted = adjusted.clamp(0.0, 255.0)

    # Optional visualization: [dst | src | adjusted]
    if save_path is not None or show:
        src_np = tensor_to_uint8_hwc(src_rgb)
        dst_np = tensor_to_uint8_hwc(dst_rgb)
        adj_np = tensor_to_uint8_hwc(adjusted)

        H = max(src_np.shape[0], dst_np.shape[0], adj_np.shape[0])
        W = max(src_np.shape[1], dst_np.shape[1], adj_np.shape[1])

        def fit(img: np.ndarray) -> np.ndarray:
            return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        vis = np.concatenate([fit(dst_np), fit(src_np), fit(adj_np)], axis=1)  # RGB strip

        if save_path is not None:
            # OpenCV expects BGR for saving; convert from RGB
            cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        if show:
            cv2.imshow("dst | src | adjusted", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return adjusted


#############################
# CLI
#############################

def parse_args():
    p = argparse.ArgumentParser(description="Brightness match (PyTorch 2.8.0)")
    p.add_argument("src", type=str, help="Path to source (reference) image")
    p.add_argument("dst", type=str, help="Path to target image to be adjusted")
    p.add_argument("--out", type=str, default=None, help="Optional save path for side-by-side preview")
    p.add_argument("--mode", type=str, default="luma", choices=["luma", "channels"], help="Scaling mode")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Torch device")
    p.add_argument("--show", action="store_true", help="Show preview window")
    p.add_argument("--verbose", action="store_true", help="Print stats")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    src = read_img(args.src, device)
    dst = read_img(args.dst, device)

    with torch.no_grad():
        adjusted = adjust_brightness_from_src_to_dst(dst, src, save_path=args.out, show=args.show, verbose=args.verbose, mode=args.mode)

    out_single = args.out
    if out_single is None:
        root, ext = os.path.splitext(args.dst)
        out_single = f"{root}_adjusted{ext}"
    adj_np = tensor_to_uint8_hwc(adjusted)
    cv2.imwrite(out_single, cv2.cvtColor(adj_np, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()