# Anime-style Transfer

PyTorch implementations of AnimeGANv2 and CycleGAN.

## Dataset
- Download: [Link](https://github.com/TachibanaYoshino/AnimeGAN/releases/tag/dataset-1)
- Arrange data:
  ```
  data/
  ├── train_photo/
  ├── Hayao/{style,smooth}/
  ├── Shinkai/{style,smooth}/
  └── ...
  ```
- Use `scripts/edge_smooth.py` to create extra smooth images if needed.

## Installation
Requirements: Python 3.11, CUDA 11.8+ (optional).

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Train
Configs in `args/` capture all paths and hyper-parameters. Pass the YAML via `--config_file` and override anything (e.g. `--photo_root`, `--batch_size`) as needed.

```bash
# AnimeGANv2
python train.py --config_file args/animegan.yaml

# CycleGAN
python train.py --config_file args/cyclegan.yaml
```

## Inference
Single image:
```bash
python infer.py \
    --mode single \
    --image_file demo/photo.jpg \
    --ckpt_file checkpoints/generator_animegan_shinkai.pth \
    --output_dir output/infer_single
```

Entire directory:
```bash
python infer.py \
    --mode directory \
    --image_dir demo/photos \
    --ckpt_file checkpoints/generator_animegan_shinkai.pth \
    --output_dir output/photos_anime \
```


## Demo
```bash
streamlit run demo.py
```

## Acknowledgements
- [AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)
- [animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch)
- [pytorch-animeGAN](https://github.com/ptran1203/pytorch-animeGAN)
- [PyTorch-CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN)

Thanks to the authors and contributors of these repositories for releasing their code and documentation.
