from .base_trainer import BaseTrainer
from models import AnimeGANGenerator, AnimeGANDiscriminator
from losses import AnimeGANAdversarialLoss, AnimeGANContentLoss, AnimeGANGrayscaleStyleLoss, AnimeGANColorReconstructionLoss


class AnimeGANTrainer(BaseTrainer):
    def build_models(self): pass
    def build_optim(self): pass
    def train_one_step(self, batch: dict, step: int): pass
    def on_epoch_end(self, epoch: int): pass
