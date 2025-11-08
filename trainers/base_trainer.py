import torch
from tqdm import tqdm
from abc import ABC, abstractmethod

# TPU imports
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.distributed.parallel_loader import MpDeviceLoader


class BaseTrainer(ABC):
    def __init__(self, args, loader):
        self.args = args
        self.loader = loader
        if args.device == "tpu":
            self.device = xm.xla_device()
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def build_models(self): pass

    @abstractmethod
    def build_optim(self): pass

    @abstractmethod
    def train_one_step(self, batch, step): pass

    @abstractmethod
    def on_epoch_end(self, epoch): pass

    def _wrap_loader_if_tpu(self):
        """For TPU"""
        if isinstance(self.device, torch_xla.device.XLADevice):
            self.loader = MpDeviceLoader(self.loader, self.device)

    def run(self):
        self.build_models()
        self.build_optim()

        self._wrap_loader_if_tpu() # For TPU
        
        step = 0
        first_epoch = self.args.start_epoch + 1 if self.args.resume else 1

        for epoch in tqdm(range(first_epoch, self.args.num_epochs + 1),
                          desc=f"Training {self.args.model} from epoch {first_epoch} to {self.args.num_epochs}"):
            for batch in self.loader:
                step += 1
                self.train_one_step(batch, step)

            if isinstance(self.device, torch_xla.device.XLADevice): # For TPU
                xm.mark_step()
                              
            self.on_epoch_end(epoch)

    @torch.no_grad()
    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 + 0.5
