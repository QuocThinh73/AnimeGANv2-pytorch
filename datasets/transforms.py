from torchvision import transforms


def build_transforms(train: bool, image_size: int = 256, norm_mean=(0.5, 0.5, 0.5), norm_std=(0.5, 0.5, 0.5)):
    if train:
        return transforms.Compose([
            transforms.Resize(int(image_size * 286 / 256)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
    
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])