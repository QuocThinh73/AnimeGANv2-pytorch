import os
import cv2
import numpy as np
import torch
from tools.adjust_brightness import adjust_brightness_from_src_to_dst, read_img

def preprocessing(img: np.ndarray, size) -> np.ndarray:
    h, w = img.shape[:2]
    h_new = size[0] if h <= size[0] else h - (h % 32)
    w_new = size[1] if w < size[1] else w - (w % 32)
    img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 127.5 - 1.0
    return img

def load_test_data(image_path, size, channels_first=True, as_tensor=True, dtype=torch.float32) -> torch.Tensor:
    """
    Read image -> RGB -> preprocessing ([-1,1]) -> return Tensor NCHW (default).
    """
    img = cv2.imread(image_path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing(img, size)

    if not as_tensor:
        return np.expand_dims(img, axis=0)  # 1xHxWx3 (like TF)
    
    # HWC -> CHW -> NCHW
    if channels_first:
        t = torch.from_numpy(img).to(dtype)  # HWC
        t = t.permute(2, 0, 1).unsqueeze(0).contiguous()  # 1x3xHxW
    else:
        t = torch.from_numpy(img).to(dtype).unsqueeze(0)  # 1xHxWx3
    
    return t

def inverse_transform(images: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    [-1,1] -> [0,255] uint8, clip to avoid float errors causing artifacts.
    Accepts numpy or torch.
    """
    if torch.is_tensor(images):
        images = images.detach().cpu().numpy()

    # If tensor has batch/channel, peel back to HWC
    # Support (N,C,H,W), (C,H,W), (H,W,3), (H,W)
    if images.ndim == 4: # NCHW or NHWC
        if images.shape[1] in (1, 3): # NCHW
            images = np.transpose(images[0], (1, 2, 0))
        else: # NHWC
            images = images[0]
    elif images.ndim == 3 and images.shape[0] in (1, 3): # CHW
        images = np.transpose(images, (1, 2, 0))

    images = (images + 1.0) / 2.0 * 255.0
    images = np.clip(images, 0, 255).astype(np.uint8)
    return images

def imsave(images: np.ndarray, path):
    return cv2.imwrite(path, cv2.cvtColor(images, cv2.COLOR_RGB2BGR))

def save_images(images: np.ndarray | torch.Tensor, image_path, photo_path=None):
    fake = inverse_transform(images)

    if photo_path:
        ref = read_img(photo_path) # RGB
        fake = adjust_brightness_from_src_to_dst(fake, ref)

    return imsave(fake, image_path)

def crop_image(img: np.ndarray, x0, y0, w, h) -> np.ndarray:
    return img[y0:y0 + h, x0:x0 + w]

def random_crop(img1: np.ndarray, img2: np.ndarray, crop_H, crop_W):
    assert img1.shape == img2.shape, "img1 and img2 must be the same size"
    h, w = img1.shape[:2]
    crop_W = w if crop_W > w else crop_W
    crop_H = h if crop_H > h else crop_H
    x0 = np.random.randint(0, w - crop_W + 1)
    y0 = np.random.randint(0, h - crop_H + 1)
    crop_1 = crop_image(img1, x0, y0, crop_W, crop_H)
    crop_2 = crop_image(img2, x0, y0, crop_W, crop_H)
    return crop_1, crop_2

def show_all_variables(model=None, prefix=None):
    if model is None:
        print("[show_all_variables] Pass in nn.Module (model) to list parameters.")
        return

    total = 0
    print("All trainable parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            total += p.numel()
            print(f"{name:60s} | shape={tuple(p.shape)} | params={p.numel()}")

    print(f"Total trainable params: {total:,}")
    if prefix is not None:
        print(f"\nFiltered by prefix='{prefix}':")
        sub_total = 0
        for name, p in model.named_parameters():
            if p.requires_grad and name.startswith(prefix):
                sub_total += p.numel()
                print(f"{name:60s} | shape={tuple(p.shape)} | params={p.numel()}")
        print(f"Subtotal trainable params ({prefix}): {sub_total:,}")

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ("true", "1", "yes", "y")