import torch
import torch_xla.core.xla_model as xm
import torch_xla
from tqdm import tqdm
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(self, args, loader):
        self.args = args
        self.loader = loader
        # Sử dụng TPU device (cách mới để tránh deprecation warning)
        self.device = torch_xla.device()

    @abstractmethod
    def build_models(self): pass

    @abstractmethod
    def build_optim(self): pass

    @abstractmethod
    def train_one_step(self, batch, step): pass

    @abstractmethod
    def on_epoch_end(self, epoch): pass

    def run(self):
        self.build_models()
        self.build_optim()
        step = 0
        first_epoch = self.args.start_epoch + 1 if self.args.resume else 1

        for epoch in tqdm(range(first_epoch, self.args.num_epochs + 1),
                          desc=f"Training {self.args.model} from epoch {first_epoch} to {self.args.num_epochs}"):
            for batch in self.loader:
                step += 1
                self.train_one_step(batch, step)
            self.on_epoch_end(epoch)

    @torch.no_grad()
    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 + 0.5
