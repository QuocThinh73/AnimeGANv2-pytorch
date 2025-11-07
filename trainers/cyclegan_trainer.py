import os
import itertools
import torch
from torchvision.utils import save_image
from .base_trainer import BaseTrainer
from models import CycleGANGenerator, CycleGANDiscriminator
from losses import AdversarialLoss, CycleGANCycleConsistencyLoss, CycleGANIdentityLoss
from utils.lr_scheduler import LambdaLR
from utils.replay_buffer import ReplayBuffer


class CycleGANTrainer(BaseTrainer):
    def build_models(self):
        # Networks
        self.G_photo2anime = CycleGANGenerator().to(self.device)
        self.G_anime2photo = CycleGANGenerator().to(self.device)
        self.D_photo = CycleGANDiscriminator().to(self.device)
        self.D_anime = CycleGANDiscriminator().to(self.device)

        # Initialize weights
        if not self.args.resume:
            from utils.weights_init_normal import weights_init_normal
            self.G_photo2anime.apply(weights_init_normal)
            self.G_anime2photo.apply(weights_init_normal)
            self.D_photo.apply(weights_init_normal)
            self.D_anime.apply(weights_init_normal)
        # Load checkpoints
        else:
            ckpt_path = os.path.join(self.args.ckpt_dir, "ckpt.pth")
            state_dict = torch.load(ckpt_path, map_location=self.device)

            self.G_photo2anime.load_state_dict(state_dict["G_photo2anime"])
            self.G_anime2photo.load_state_dict(state_dict["G_anime2photo"])
            self.D_photo.load_state_dict(state_dict["D_photo"])
            self.D_anime.load_state_dict(state_dict["D_anime"])

        # Loss functions
        self.criterion_GAN = AdversarialLoss(lambda_adv=self.args.lambda_adv)
        self.criterion_cycle = CycleGANCycleConsistencyLoss(
            lambda_cyc=self.args.lambda_cyc)
        self.criterion_identity = CycleGANIdentityLoss(
            lambda_idt=self.args.lambda_idt)

        # Output directories
        self.sample_dir = os.path.join(
            self.args.out_dir, "cyclegan", "samples")
        self.ckpt_dir = os.path.join(
            self.args.out_dir, "cyclegan", "checkpoints")
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Logger
        self.logger = {
            "G": 0.0,
            "D_photo": 0.0,
            "D_anime": 0.0,
            "n": 0
        }

        self._last_fake_photo = None
        self._last_fake_anime = None

        # Replay buffers
        self.photo_buffer = ReplayBuffer()
        self.anime_buffer = ReplayBuffer()

    def build_optim(self):
        # Optimizers
        self.optimizer_G = torch.optim.Adam(itertools.chain(
            self.G_photo2anime.parameters(), self.G_anime2photo.parameters()), lr=self.args.lr, betas=(0.5, 0.999))
        self.optimizer_D_photo = torch.optim.Adam(
            self.D_photo.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        self.optimizer_D_anime = torch.optim.Adam(
            self.D_anime.parameters(), lr=self.args.lr, betas=(0.5, 0.999))

        # Load checkpoints
        if self.args.resume:
            ckpt_path = os.path.join(self.args.ckpt_dir, "ckpt.pth")
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.optimizer_G.load_state_dict(state_dict["opt_G"])
            self.optimizer_D_photo.load_state_dict(state_dict["opt_D_photo"])
            self.optimizer_D_anime.load_state_dict(state_dict["opt_D_anime"])

        # Learning rate schedulers
        lr_lambda = LambdaLR(
            n_epochs=self.args.num_epochs, offset=self.args.start_epoch, decay_start_epoch=self.args.decay_epoch)
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=lr_lambda.step)
        self.lr_scheduler_D_photo = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_photo, lr_lambda=lr_lambda.step)
        self.lr_scheduler_D_anime = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_anime, lr_lambda=lr_lambda.step)

    @torch.no_grad()
    def _save_samples(self, fake_photo: torch.Tensor, fake_anime: torch.Tensor, epoch: int):
        save_image(self._denorm(fake_photo)[:4], os.path.join(
            self.sample_dir, f"fake_photo_epoch_{epoch:03d}.png"), nrow=2)
        save_image(self._denorm(fake_anime)[:4], os.path.join(
            self.sample_dir, f"fake_anime_epoch_{epoch:03d}.png"), nrow=2)

    @torch.no_grad()
    def _save_checkpoints(self, epoch: int):
        ckpt_epoch_dir = os.path.join(self.ckpt_dir, f"epoch_{epoch:03d}")
        os.makedirs(ckpt_epoch_dir, exist_ok=True)
        torch.save(self.G_photo2anime.state_dict(), os.path.join(
            ckpt_epoch_dir, "G_photo2anime.pth"))
        torch.save({
            "G_photo2anime": self.G_photo2anime.state_dict(),
            "G_anime2photo": self.G_anime2photo.state_dict(),
            "D_photo": self.D_photo.state_dict(),
            "D_anime": self.D_anime.state_dict(),
            "opt_G": self.optimizer_G.state_dict(),
            "opt_D_photo": self.optimizer_D_photo.state_dict(),
            "opt_D_anime": self.optimizer_D_anime.state_dict(),
            "epoch": epoch,
        }, os.path.join(ckpt_epoch_dir, "ckpt.pth"))

    def train_one_step(self, batch: dict, step: int):
        real_photo = batch["photo_image"].to(self.device)
        real_anime = batch["anime_style_image"].to(self.device)

        ######### Generators #########
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

        # Save samples for logging
        with torch.no_grad():
            self._last_fake_photo = fake_photo.detach().cpu()
            self._last_fake_anime = fake_anime.detach().cpu()

        ######### Photo Discriminator #########
        self.optimizer_D_photo.zero_grad()

        # Real loss
        pred_real_photo = self.D_photo(real_photo)
        loss_D_photo_real = self.criterion_GAN(pred_real_photo, 1.0)

        # Fake loss
        fake_photo = fake_photo.detach()
        fake_photo = self.photo_buffer.push_and_pop(fake_photo)
        pred_fake_photo = self.D_photo(fake_photo)
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
        fake_anime = fake_anime.detach()
        fake_anime = self.anime_buffer.push_and_pop(fake_anime)
        pred_fake_anime = self.D_anime(fake_anime)
        loss_D_anime_fake = self.criterion_GAN(pred_fake_anime, 0.0)

        # Total discriminator loss
        loss_D_anime = 0.5 * (loss_D_anime_real + loss_D_anime_fake)
        loss_D_anime.backward()

        self.optimizer_D_anime.step()

        #########################################################

        # Logging
        self.logger["G"] += float(loss_G.item())
        self.logger["D_photo"] += float(loss_D_photo.item())
        self.logger["D_anime"] += float(loss_D_anime.item())
        self.logger["n"] += 1

    def on_epoch_end(self, epoch: int):
        # Update learning rates
        self.lr_scheduler_G.step()
        self.lr_scheduler_D_photo.step()
        self.lr_scheduler_D_anime.step()

        # Logging
        n = max(self.logger["n"], 1)
        avg_G = self.logger["G"] / n
        avg_D_photo = self.logger["D_photo"] / n
        avg_D_anime = self.logger["D_anime"] / n
        print(
            f"[Epoch {epoch}] "
            f"G: {avg_G:.3f} "
            f"D_photo: {avg_D_photo:.3f} "
            f"D_anime: {avg_D_anime:.3f}"
        )

        # Reset logger
        self.logger = {"G": 0.0, "D_photo": 0.0, "D_anime": 0.0, "n": 0}

        # Save
        if epoch % self.args.save_every == 0 and self._last_fake_photo is not None and self._last_fake_anime is not None:
            self._save_checkpoints(epoch)
            self._save_samples(self._last_fake_photo,
                               self._last_fake_anime, epoch)
