import torch
from tqdm import tqdm
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(self, args, loader):
        self.args = args
        self.loader = loader
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

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
        for epoch in tqdm(range(1, self.args.num_epochs + 1), desc="Training"):
            for batch in self.loader:
                step += 1
                self.train_one_step(batch, step)
            self.on_epoch_end(epoch)
