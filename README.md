# Anime-style Transfer

Hiện thực mô hình AnimeGANv2 và CycleGAN để chuyển đổi ảnh thực thành phong cách anime.

## Cài đặt

### Yêu cầu

- Python 3.13
- CUDA 11.8+

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

Nếu sử dụng GPU, cài đặt PyTorch với CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Cấu trúc dự án

```
.
├── args/                  
│   ├── animegan.yaml
│   └── cyclegan.yaml
├── data/                 
│   ├── train_photo/        
│   ├── {Style}/            
│   │   ├── style/          
│   │   └── smooth/         
│   └── val/                 
├── datasets/               
├── losses/                 
├── models/                 
│   ├── animegan/
│   └── cyclegan/
├── trainers/                
├── utils/                   
├── scripts/                 
└── output/                  
```

## Sử dụng

### Training

#### AnimeGANv2

```bash
python train.py \
    --model animegan \
    --args_root args \
    --photo_root data/train_photo \
    --anime_style_root data/{Style}/style \
    --anime_smooth_root data/{Style}/smooth \
    --out_dir output/{Style}/animegan
```

#### CycleGAN

```bash
python train.py \
    --model cyclegan \
    --args_root args \
    --photo_root data/train_photo \
    --anime_style_root data/{Style}/style \
    --out_dir output/{Style}/cyclegan
```

### Sử dụng Docker

#### Build Docker image

```bash
docker build -t cyclegan-trainer:cu118 .
```

#### Training với Docker (AnimeGAN)

```bash
bash scripts/train/train_animegan.sh Shinkai 0 0 0
```

#### Training với Docker (CycleGAN)

```bash
bash scripts/train/train_cyclegan.sh Shinkai 0 0 0
```

### Cấu hình

Bạn có thể chỉnh sửa các file YAML trong thư mục `args/` để thay đổi:
- Learning rates
- Loss weights
- Training parameters
- Model settings

Hoặc override trực tiếp qua command line:

```bash
python train.py --model animegan --args_root args --batch_size 8 --num_epochs 150
```

## Dataset

### Dataset gốc từ tác giả AnimeGANv2

Bạn có thể tải dataset từ repository gốc của tác giả:

- **AnimeGANv2 Dataset**: [https://github.com/TachibanaYoshino/AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)

Dataset bao gồm:
- Ảnh thực (train_photo)
- Ảnh anime style cho các phong cách: Hayao, Shinkai, Paprika, SummerWar
- Ảnh anime smooth (cho AnimeGAN)

### Chuẩn bị dữ liệu

1. Tải dataset từ link trên
2. Tổ chức thư mục theo cấu trúc:
   ```
   data/
   ├── train_photo/          # Ảnh thực
   ├── Hayao/
   │   ├── style/            # Ảnh anime style
   │   └── smooth/           # Ảnh anime smooth
   ├── Shinkai/
   │   ├── style/
   │   └── smooth/
   └── ...
   ```

3. Sử dụng script `scripts/edge_smooth.py` để tạo smooth images nếu cần

## Kết quả

Sau khi training, kết quả sẽ được lưu trong thư mục `output/`:
- Checkpoints: `output/{Style}/{model}/checkpoints/`
- Generated images: `output/{Style}/{model}/images/`

## Acknowledgements

Dự án này được xây dựng dựa trên và tham khảo từ các repository sau:

- **[AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)** - Repository gốc của tác giả TachibanaYoshino, cung cấp implementation TensorFlow và dataset
- **[animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch)** - PyTorch implementation của bryandlee, cung cấp nhiều pretrained models
- **[pytorch-animeGAN](https://github.com/ptran1203/pytorch-animeGAN)** - PyTorch implementation của ptran1203 với nhiều phong cách anime
- **[PyTorch-CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN)** - PyTorch implementation của CycleGAN từ aitorzip

Chúng tôi xin gửi lời cảm ơn chân thành đến các tác giả và contributors của các repository trên vì đã cung cấp mã nguồn và tài liệu tham khảo quý giá.

## Citation

Nếu bạn sử dụng code này trong nghiên cứu, vui lòng cite các paper gốc:

**AnimeGANv2:**
```bibtex
@inproceedings{chen2021animeganv2,
  title={AnimeGANv2: Learning to Generate High-Quality Anime Images from Photos},
  author={Chen, Yunjey and others},
  booktitle={arXiv preprint arXiv:2102.06239},
  year={2021}
}
```

**CycleGAN:**
```bibtex
@inproceedings{zhu2017unpaired,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={ICCV},
  year={2017}
}
```

## License

MIT License - Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## Liên hệ

Nếu có câu hỏi hoặc góp ý, vui lòng tạo issue trên GitHub repository.
