import os
import torch
from torchvision.utils import save_image
from .base_trainer import BaseTrainer
from models import AnimeGANGenerator, AnimeGANDiscriminator
from losses import AdversarialLoss, AnimeGANContentLoss, AnimeGANGrayscaleStyleLoss, AnimeGANColorReconstructionLoss
from utils.image_processing import rgb_to_gray


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
        self.criterion_GAN = AdversarialLoss(lambda_adv=self.args.lambda_adv)
        self.criterion_content = AnimeGANContentLoss(
            lambda_con=self.args.lambda_con, backbone=self.args.backbone)
        self.criterion_grayscale_style = AnimeGANGrayscaleStyleLoss(
            lambda_gra=self.args.lambda_gra, backbone=self.args.backbone)
        self.criterion_color_reconstruction = AnimeGANColorReconstructionLoss(
            lambda_col=self.args.lambda_col)

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

        fake_anime_style = self.G(real_photo)
        pred_fake_anime_style = self.D(fake_anime_style)

        loss_adversarial = self.criterion_GAN(pred_fake_anime_style, 1.0)
        loss_content = self.criterion_content(fake_anime_style, real_photo)
        loss_grayscale_style = self.criterion_grayscale_style(
            fake_anime_style, real_anime_style)
        loss_color_reconstruction = self.criterion_color_reconstruction(
            fake_anime_style, real_photo)

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
            self._last_fake_anime_style = fake_anime_style.detach().cpu()

        ######### Discriminator #########
        self.optimizer_D.zero_grad()

        real_anime_style_gray = rgb_to_gray(real_anime_style)

        # Logits
        pred_real_anime_style = self.D(real_anime_style)
        pred_real_anime_style_gray = self.D(real_anime_style_gray)
        pred_fake_anime_style = self.D(fake_anime_style.detach())
        pred_fake_anime_smooth = self.D(real_anime_smooth)

        # Real loss
        loss_D_real_anime_style = self.criterion_GAN(
            pred_real_anime_style, 1.0)
        loss_D_real_anime_style_gray = self.criterion_GAN(
            pred_real_anime_style_gray, 1.0)

        # Fake loss
        loss_D_fake_anime_style = self.criterion_GAN(
            pred_fake_anime_style, 0.0)
        loss_D_fake_anime_smooth = self.criterion_GAN(
            pred_fake_anime_smooth, 0.0)

        # Total loss
        loss_D_real = 0.5 * (loss_D_real_anime_style +
                             loss_D_real_anime_style_gray)
        loss_D_fake = 0.5 * (loss_D_fake_anime_style +
                             loss_D_fake_anime_smooth)
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()

        self.optimizer_D.step()
        #########################################################

        # Logging
        self.logger["G_total"] += float(loss_G.item())
        self.logger["D_total"] += float(loss_D.item())
        self.logger["G_adv"] += float(loss_adversarial.item())
        self.logger["G_content"] += float(loss_content.item())
        self.logger["G_gray"] += float(loss_grayscale_style.item())
        self.logger["G_color"] += float(loss_color_reconstruction.item())
        self.logger["D_real"] += float(loss_D_real.item())
        self.logger["D_fake"] += float(loss_D_fake.item())
        self.logger["n"] += 1

    def on_epoch_end(self, epoch: int):
        # Logging
        n = max(self.logger["n"], 1)
        avg_G_total = self.logger["G_total"] / n
        avg_D_total = self.logger["D_total"] / n
        avg_G_adv = self.logger["G_adv"] / n
        avg_G_content = self.logger["G_content"] / n
        avg_G_gray = self.logger["G_gray"] / n
        avg_G_color = self.logger["G_color"] / n
        avg_D_real = self.logger["D_real"] / n
        avg_D_fake = self.logger["D_fake"] / n
        print(f"[Epoch {epoch}] | G_total: {avg_G_total:.3f} | D_total: {avg_D_total:.3f} | G_adv: {avg_G_adv:.3f} | G_content: {avg_G_content:.3f} | G_gray: {avg_G_gray:.3f} | G_color: {avg_G_color:.3f} | D_real: {avg_D_real:.3f} | D_fake: {avg_D_fake:.3f}")

        # Reset logger
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

        # Save
        if epoch % self.args.save_every == 0 and self._last_fake_anime_style is not None:
            self._save_checkpoints(epoch)
            self._save_samples(self._last_fake_anime_style, epoch)
