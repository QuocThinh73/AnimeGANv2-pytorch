import os, cv2
import argparse
from glob import glob
import numpy as np
import torch
import torch.nn.functional as F
from tools.utils import check_folder

def parse_args():
    parser = argparse.ArgumentParser(description="Edge smoothed (Pytorch)")
    parser.add_argument('--dataset', type=str, default='Paprika', help='dataset_name')
    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to run blur on')
    return parser.parse_args()

def make_gaussian_kernel_5x5() -> torch.Tensor:
    # Create a normalized 5x5 kernel (sum=1)
    g = cv2.getGaussianKernel(5, 0) # (5, 1)
    k = (g @ g.T).astype(np.float32) # (5, 5)
    k /= k.sum()
    k_t = torch.from_numpy(k)
    k3 = torch.stack([k_t, k_t, k_t], dim=0).unsqueeze(1) # (3, 1, 5, 5)
    return k3

def process_image_torch(bgr_img: np.ndarray, mask: np.ndarray, device: torch.device) -> np.ndarray:
    H, W, _ = bgr_img.shape
    x = torch.from_numpy(bgr_img.astype(np.float32)) # (H, W, 3)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device) # (1, 3, H, W)
    x_pad = F.pad(x, (2, 2, 2, 2), mode='reflect')
    weight = make_gaussian_kernel_5x5().to(device) # (3, 1, 5, 5)
    blurred = F.conv2d(x_pad, weight, bias=None, stride=1, padding=0, groups=3)
    m = torch.from_numpy((mask != 0)).to(device) # (H, W) bool
    m = m.unsqueeze(0).unsqueeze(0).expand(1, 3, H, W) # (1, 3, H, W)
    out = torch.where(m, blurred, x) # (1, 3, H, W)
    out = out.squeeze(0).permute(1, 2, 0).clamp(0, 255).to('cpu').numpy().astype(np.uint8) # (H, W, 3)
    return out

def process_image_opencv_fast(bgr_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(bgr_img, (5, 5), 0, borderType=cv2.BORDER_REFLECT_101)
    out = bgr_img.copy()
    out[mask != 0] = blurred[mask != 0]
    return out

def make_edge_smooth(dataset_name, img_size, device_pref='auto'):
    root = os.path.dirname(os.path.dirname(__file__))
    in_dir = os.path.join(root, 'dataset', dataset_name, 'style')
    out_dir = os.path.join(root, 'dataset', dataset_name, 'smooth')
    check_folder(out_dir)
    file_list = sorted(glob(os.path.join(in_dir, '*.*')))

    if device_pref == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_pref == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    use_torch = (device.type == 'cuda')
    kernel = np.ones((5, 5), np.uint8) # for dilation

    for src_path in file_list:
        file_name = os.path.basename(src_path)
        bgr_img = cv2.imread(src_path)
        gray_img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if bgr_img is None or gray_img is None:
            print(f"[WARN] Failed to read {src_path}, skipping.")
            continue

        bgr_img = cv2.resize(bgr_img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        gray_img = cv2.resize(gray_img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        edges = cv2.Canny(gray_img, 100, 200)
        dilation = cv2.dilate(edges, kernel)

        if use_torch:
            out_img = process_image_torch(bgr_img, dilation, device)
        else:
            out_img = process_image_opencv_fast(bgr_img, dilation)

        cv2.imwrite(os.path.join(out_dir, file_name), out_img)

def main():
    args = parse_args()
    make_edge_smooth(args.dataset, args.img_size, device_pref=args.device)

if __name__ == '__main__':
    main()