import os
import itertools
import torch
from torchvision.utils import save_image
from .base_trainer import BaseTrainer
from models import CycleGANGenerator, CycleGANDiscriminator
from losses import AdversarialLoss, CycleConsistencyLoss, IdentityLoss


class CycleGANTrainer(BaseTrainer):
    def build_models(self):
        args = self.args

        # Networks
        self.G_photo2anime = CycleGANGenerator(
            in_channels=3, out_channels=3).to(self.device)
        self.G_anime2photo = CycleGANGenerator(
            in_channels=3, out_channels=3).to(self.device)
        self.D_photo = CycleGANDiscriminator(in_channels=3).to(self.device)
        self.D_anime = CycleGANDiscriminator(in_channels=3).to(self.device)

        # Loss functions
        self.criterion_GAN = AdversarialLoss()
        self.criterion_cycle = CycleConsistencyLoss(lambda_cyc=args.lambda_cyc)
        self.criterion_identity = IdentityLoss(lambda_idt=args.lambda_idt)

        # Output directories
        self.sample_dir = os.path.join(args.out_dir, "cyclegan", "samples")
        self.ckpt_dir = os.path.join(args.out_dir, "cyclegan", "checkpoints")
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def build_optim(self):
        args = self.args

        # Optimizers
        self.optimizer_G = torch.optim.Adam(itertools.chain(
            self.G_photo2anime.parameters(), self.G_anime2photo.parameters()), lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_D_photo = torch.optim.Adam(
            self.D_photo.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_D_anime = torch.optim.Adam(
            self.D_anime.parameters(), lr=args.lr, betas=(0.5, 0.999))

        # Learning rate schedulers
        def lambda_rule(epoch):
            e = epoch + 1
            return 1.0 - max(0, e - args.decay_epoch) / (args.epochs - args.decay_epoch)
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lambda_rule)
        self.lr_scheduler_D_photo = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_photo, lambda_rule)
        self.lr_scheduler_D_anime = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_anime, lambda_rule)

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 + 0.5

    @torch.no_grad()
    def _save_samples(self, fake_photo: torch.Tensor, fake_anime: torch.Tensor, step: int):
        save_image(self._denorm(fake_photo)[:4], os.path.join(
            self.sample_dir, f"fake_photo_{step}.png"), nrow=2)
        save_image(self._denorm(fake_anime)[:4], os.path.join(
            self.sample_dir, f"fake_anime_{step}.png"), nrow=2)

    def train_one_step(self, batch: dict, step: int):
        device = self.device
        real_photo = batch["photo_image"].to(device)
        real_anime = batch["anime_style_image"].to(device)

        ######### Generatos #########
        self.optimizer_G.zero_grad()

        # Identity loss
        same_anime = self.G_photo2anime(real_anime)
        loss_idt_anime = self.criterion_identity(same_anime, real_anime)

        same_photo = self.G_anime2photo(real_photo)
        loss_idt_photo = self.criterion_identity(same_photo, real_photo)

        # Adversarial loss
        fake_anime = self.G_photo2anime(real_photo)
        loss_adv_photo2anime = self.criterion_GAN(
            self.D_anime(fake_anime), 1.0)

        fake_photo = self.G_anime2photo(real_anime)
        loss_adv_anime2photo = self.criterion_GAN(
            self.D_photo(fake_photo), 1.0)

        # Cycle consistency loss
        reconstructed_photo = self.G_anime2photo(fake_anime)
        loss_cycle_photo = self.criterion_cycle(
            reconstructed_photo, real_photo)

        reconstructed_anime = self.G_photo2anime(fake_photo)
        loss_cycle_anime = self.criterion_cycle(
            reconstructed_anime, real_anime)

        # Total loss
        loss_G = (
            loss_idt_photo + loss_idt_anime
            + loss_adv_photo2anime + loss_adv_anime2photo
            + loss_cycle_photo + loss_cycle_anime
        )
        loss_G.backward()

        self.optimizer_G.step()
        #########################################################

        ######### Photo Discriminator #########
        self.optimizer_D_photo.zero_grad()

        # Real loss
        pred_real_photo = self.D_photo(real_photo)
        loss_D_photo_real = self.criterion_GAN(pred_real_photo, 1.0)

        # Fake loss
        pred_fake_photo = self.D_photo(fake_photo.detach())
        loss_D_photo_fake = self.criterion_GAN(pred_fake_photo, 0.0)

        # Total discriminator loss
        loss_D_photo = 0.5 * (loss_D_photo_real + loss_D_photo_fake)
        loss_D_photo.backward()

        self.optimizer_D_photo.step()

        #########################################################

        ######### Anime Discriminator #########
        self.optimizer_D_anime.zero_grad()

        # Real loss
        pred_real_anime = self.D_anime(real_anime)
        loss_D_anime_real = self.criterion_GAN(pred_real_anime, 1.0)

        # Fake loss
        pred_fake_anime = self.D_anime(fake_anime.detach())
        loss_D_anime_fake = self.criterion_GAN(pred_fake_anime, 0.0)

        # Total discriminator loss
        loss_D_anime = 0.5 * (loss_D_anime_real + loss_D_anime_fake)
        loss_D_anime.backward()

        self.optimizer_D_anime.step()

        #########################################################

        if step % 100 == 0:
            print(
                f"[{step}] "
                f"G: {loss_G.item():.3f} "
                f"| D_photo: {loss_D_photo.item():.3f} | D_anime: {loss_D_anime.item():.3f}"
            )

        if step % self.args.sample_every == 0:
            self._save_samples(fake_photo, fake_anime, step)

    def on_epoch_end(self, epoch: int):
        # Update learning rates
        self.lr_scheduler_G.step()
        self.lr_scheduler_D_photo.step()
        self.lr_scheduler_D_anime.step()

        # Save models checkpoints
        if epoch % self.args.save_every == 0:
            ckpt_path = os.path.join(self.ckpt_dir, f"epoch_{epoch:03d}.pth")
            torch.save({
                "epoch": epoch,
                "G_photo2anime": self.G_photo2anime.state_dict(),
                "G_anime2photo": self.G_anime2photo.state_dict(),
                "D_photo": self.D_photo.state_dict(),
                "D_anime": self.D_anime.state_dict(),
                "opt_G": self.optimizer_G.state_dict(),
                "opt_D_photo": self.optimizer_D_photo.state_dict(),
                "opt_D_anime": self.optimizer_D_anime.state_dict(),
            }, ckpt_path)
