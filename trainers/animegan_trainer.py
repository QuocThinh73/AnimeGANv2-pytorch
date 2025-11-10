import os
import torch
from torchvision.utils import save_image
from .base_trainer import BaseTrainer
from models import AnimeGANGenerator, AnimeGANDiscriminator
from losses import AdversarialLoss, AnimeGANContentLoss, AnimeGANGrayscaleStyleLoss, AnimeGANColorReconstructionLoss, TotalVariationLoss
from utils.image_processing import rgb_to_gray
from tqdm import tqdm
from typing import Literal


class AnimeGANTrainer(BaseTrainer):
    def __init__(self, args, loader):
        super().__init__(args, loader)

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
        self.criterion_GAN_G = AdversarialLoss(
            lambda_adv=self.args.lambda_adv_g)
        self.criterion_GAN_D = AdversarialLoss(
            lambda_adv=self.args.lambda_adv_d)
        self.criterion_content = AnimeGANContentLoss(
            lambda_con=self.args.lambda_con, backbone=self.args.backbone)
        self.criterion_grayscale_style = AnimeGANGrayscaleStyleLoss(
            lambda_gra=self.args.lambda_gra, backbone=self.args.backbone)
        self.criterion_color_reconstruction = AnimeGANColorReconstructionLoss(
            lambda_col=self.args.lambda_col)
        self.criterion_total_variation = TotalVariationLoss(
            lambda_tv=self.args.lambda_tv)

        # Output directories
        self.pretrain_dir = os.path.join(
            self.args.out_dir, "pretrain")
        self.sample_dir = os.path.join(
            self.args.out_dir, "samples")
        self.ckpt_dir = os.path.join(
            self.args.out_dir, "checkpoints")
        os.makedirs(self.pretrain_dir, exist_ok=True)
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
            "G_tv": 0.0,
            "D_real": 0.0,
            "D_fake": 0.0,
            "D_gray": 0.0,
            "D_smooth": 0.0,
            "n": 0,
        }

        self._last_fake_anime = None

    def build_optim(self):
        # Optimizers
        if self.args.pretrain_epochs > 0 and not self.args.resume:
            self.optimizer_G_pretrain = torch.optim.Adam(
                self.G.parameters(), lr=self.args.g_lr_pretrain, betas=(0.5, 0.999))

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
    def _save_samples(self, fake_anime: torch.Tensor, mode: Literal["pretrain", "train"], epoch: int):
        if mode == "pretrain":
            save_image(self._denorm(fake_anime)[:4], os.path.join(
                self.pretrain_dir, f"fake_anime_pretrain_epoch_{epoch:03d}.png"), nrow=2)
        elif mode == "train":
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

        loss_adversarial = self.criterion_GAN_G(pred_fake_anime_style, 1.0)
        loss_content = self.criterion_content(fake_anime_style, real_photo)
        loss_grayscale_style = self.criterion_grayscale_style(
            fake_anime_style, real_anime_style)
        loss_color_reconstruction = self.criterion_color_reconstruction(
            fake_anime_style, real_photo)
        loss_total_variation = self.criterion_total_variation(fake_anime_style)

        # Total loss
        loss_G = (
            loss_adversarial + loss_content
            + loss_grayscale_style + loss_color_reconstruction
            + loss_total_variation
        )
        loss_G.backward()

        self.optimizer_G.step()
        #########################################################

        # Save samples for logging
        with torch.no_grad():
            self._last_fake_anime_style = fake_anime_style.detach().cpu()

        ######### Discriminator #########
        self.optimizer_D.zero_grad()

        # Predictions
        pred_real_anime = self.D(real_anime_style)
        pred_fake_anime = self.D(fake_anime_style.detach())
        pred_gray_anime = self.D(rgb_to_gray(real_anime_style))
        pred_smooth_gray_anime = self.D(rgb_to_gray(real_anime_smooth))

        # Real loss
        # Classify real anime as real
        loss_real = self.criterion_GAN_D(pred_real_anime, 1.0)
        # Classify generated as fake
        loss_fake = self.criterion_GAN_D(pred_fake_anime, 0.0)
        # Classify real anime gray as fake
        loss_gray = self.criterion_GAN_D(pred_gray_anime, 0.0)
        # Classify real anime smooth gray as fake
        loss_smooth = self.args.lambda_smo * \
            self.criterion_GAN_D(pred_smooth_gray_anime, 0.0)

        # Total loss
        loss_D = loss_real + loss_fake + loss_gray + loss_smooth
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
        self.logger["G_tv"] += float(loss_total_variation.item())
        self.logger["D_real"] += float(loss_real.item())
        self.logger["D_fake"] += float(loss_fake.item())
        self.logger["D_gray"] += float(loss_gray.item())
        self.logger["D_smooth"] += float(loss_smooth.item())
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
        avg_G_tv = self.logger["G_tv"] / n
        avg_D_real = self.logger["D_real"] / n
        avg_D_fake = self.logger["D_fake"] / n
        avg_D_gray = self.logger["D_gray"] / n
        avg_D_smooth = self.logger["D_smooth"] / n
        print(f"[Epoch {epoch}] | "
              f"G_total: {avg_G_total:.5f} | "
              f"D_total: {avg_D_total:.5f} | "
              f"G_adv: {avg_G_adv:.5f} | "
              f"G_content: {avg_G_content:.5f} | "
              f"G_gray: {avg_G_gray:.5f} | "
              f"G_color: {avg_G_color:.5f} | "
              f"G_tv: {avg_G_tv:.5f} | "
              f"D_real: {avg_D_real:.5f} | "
              f"D_fake: {avg_D_fake:.5f} | "
              f"D_gray: {avg_D_gray:.5f} | "
              f"D_smooth: {avg_D_smooth:.5f}")

        # Reset logger
        self.logger = {
            "G_total": 0.0,
            "D_total": 0.0,
            "G_adv": 0.0,
            "G_content": 0.0,
            "G_gray": 0.0,
            "G_color": 0.0,
            "G_tv": 0.0,
            "D_real": 0.0,
            "D_fake": 0.0,
            "D_gray": 0.0,
            "D_smooth": 0.0,
            "n": 0,
        }

        # Save
        if epoch % self.args.save_every == 0 and self._last_fake_anime_style is not None:
            self._save_checkpoints(epoch)
            self._save_samples(self._last_fake_anime_style, "train", epoch)

    def run(self):
        self.build_models()
        self.build_optim()

        if self.args.pretrain_epochs > 0 and not self.args.resume:
            self.pretrain_generator()

        step = 0
        first_epoch = self.args.start_epoch + 1 if self.args.resume else 1

        for epoch in tqdm(range(first_epoch, self.args.num_epochs + 1),
                          desc=f"Training {self.args.model} from epoch {first_epoch} to {self.args.num_epochs}"):
            for batch in self.loader:
                step += 1
                self.train_one_step(batch, step)
            self.on_epoch_end(epoch)

    def pretrain_generator(self):
        for epoch in tqdm(range(1, self.args.pretrain_epochs + 1), desc=f"Pretraining generator from epoch 1 to {self.args.pretrain_epochs}"):
            total, n, last_fake_anime = 0.0, 0, None
            for batch in self.loader:
                real_photo = batch["photo_image"].to(self.device)

                self.optimizer_G_pretrain.zero_grad()
                fake_anime = self.G(real_photo)
                loss_con = self.criterion_content(fake_anime, real_photo)
                loss_con.backward()
                self.optimizer_G_pretrain.step()

                total += float(loss_con.item())
                n += 1
                last_fake_anime = fake_anime.detach().cpu()

            print(f"[Pretrain][Epoch {epoch}/{self.args.pretrain_epochs}] | "
                  f"Content loss: {total / n:.5f}")

            if last_fake_anime is not None:
                self._save_samples(last_fake_anime, "pretrain", epoch)

        torch.save(self.G.state_dict(), os.path.join(
            self.pretrain_dir, "G_pretrain.pth"))

        del self.optimizer_G_pretrain
        torch.cuda.empty_cache()
