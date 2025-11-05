class LambdaLR:
    def __init__(self, n_epochs: int, offset: int, decay_start_epoch: int):
        assert n_epochs - decay_start_epoch > 0, "Decay phải bắt đầu trước khi training kết thúc!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch: int) -> float:
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
