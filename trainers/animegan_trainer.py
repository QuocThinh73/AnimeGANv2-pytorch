import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from .base_trainer import BaseTrainer
from models import AnimeGANGenerator, AnimeGANDiscriminator
from losses import AnimeGANAdversarialLoss, AnimeGANContentLoss, AnimeGANGrayscaleStyleLoss, AnimeGANColorReconstructionLoss


class AnimeGANTrainer(BaseTrainer):
    def __init__(self, args, loader):
        super().__init__(args, loader)
        self.stage = "pretrain" if args.resume else "train"

    def build_models(self):
        # Networks
        self.G = AnimeGANGenerator().to(self.device)
        self.D = AnimeGANDiscriminator().to(self.device)

        # Load checkpoints
        if self.args.resume:
            ckpt_path = os.path.join(self.args.ckpt_dir, "ckpt.pth")
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.G.load_state_dict(state_dict["G"])
            self.D.load_state_dict(state_dict["D"])

        # Loss functions
        self.criterion_GAN = AnimeGANAdversarialLoss(self.args.lambda_adv)
        self.criterion_content = AnimeGANContentLoss(self.args.lambda_con)
        self.criterion_grayscale_style = AnimeGANGrayscaleStyleLoss(
            self.args.lambda_gra)
        self.criterion_color_reconstruction = AnimeGANColorReconstructionLoss(
            self.args.lambda_col)

        # Output directories
        self.sample_dir = os.path.join(
            self.args.out_dir, "animegan", "samples")
        self.ckpt_dir = os.path.join(
            self.args.out_dir, "animegan", "checkpoints")
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Logger
        self.logger = {
            "G_total": 0.0,
            "D_total": 0.0,
            "G_adv": 0.0,
            "G_content": 0.0,
            "G_gray": 0.0,
            "G_color": 0.0,
            "D_real": 0.0,
            "D_fake": 0.0,
            "n": 0,
        }

        self._last_fake_anime = None

    def build_optim(self):
        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            self.G.parameters(), lr=self.args.g_lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(
            self.D.parameters(), lr=self.args.d_lr, betas=(0.5, 0.999))

        # Load checkpoints
        if self.args.resume:
            ckpt_path = os.path.join(self.args.ckpt_dir, "ckpt.pth")
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.optimizer_G.load_state_dict(state_dict["opt_G"])
            self.optimizer_D.load_state_dict(state_dict["opt_D"])

    @torch.no_grad()
    def _save_samples(self, fake_anime: torch.Tensor, epoch: int):
        save_image(self._denorm(fake_anime)[:4], os.path.join(
            self.sample_dir, f"fake_anime_epoch_{epoch:03d}.png"), nrow=2)

    @torch.no_grad()
    def _save_checkpoints(self, epoch: int):
        ckpt_epoch_dir = os.path.join(self.ckpt_dir, f"epoch_{epoch:03d}")
        os.makedirs(ckpt_epoch_dir, exist_ok=True)
        torch.save(self.G.state_dict(), os.path.join(
            ckpt_epoch_dir, "G.pth"))
        torch.save({
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
            "opt_G": self.optimizer_G.state_dict(),
            "opt_D": self.optimizer_D.state_dict(),
            "epoch": epoch,
        }, os.path.join(ckpt_epoch_dir, "ckpt.pth"))

    def train_one_step(self, batch: dict, step: int):
        real_photo = batch["photo_image"].to(self.device)
        real_anime_style = batch["anime_style_image"].to(self.device)
        real_anime_smooth = batch["anime_smooth_image"].to(self.device)

        ######### Generators #########
        self.optimizer_G.zero_grad()

        fake_anime = self.G(real_photo)
        pred_fake_anime = self.D(fake_anime)

        loss_adversarial = self.criterion_GAN(pred_fake_anime, 1.0)
        loss_content = self.criterion_content(fake_anime, real_photo)
        loss_grayscale_style = self.criterion_grayscale_style(
            fake_anime, real_anime_style)
        loss_color_reconstruction = self.criterion_color_reconstruction(
            fake_anime, real_photo)

        # Total loss
        loss_G = (
            loss_adversarial + loss_content
            + loss_grayscale_style + loss_color_reconstruction
        )
        loss_G.backward()

        self.optimizer_G.step()
        #########################################################

        # Save samples for logging
        with torch.no_grad():
            self._last_fake_anime = fake_anime.detach().cpu()

        ######### Discriminator #########
        self.optimizer_D.zero_grad()

        # Real loss

    def on_epoch_end(self, epoch: int): pass
