import pytorch_lightning as pl
import torch


class EMA(pl.Callback):
    def __init__(self, decay: float):
        super().__init__()
        self.decay = decay
        self.ema_weights = None

    def on_train_start(self, trainer, pl_module):
        self.ema_weights = [w.clone().detach() for w in pl_module.parameters()]
        # print(111)

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        alpha = self.decay
        for ema_w, w in zip(self.ema_weights, pl_module.parameters()):
            ema_w.data.mul_(alpha).add_((1 - alpha) * w.data)

    def on_validation_start(self, trainer, pl_module):
        # replace model weights with EMA weights for validation
        if self.ema_weights is None:
            self.ema_weights = [w.clone().detach() for w in pl_module.parameters()]
        self.original_weights = [w.clone().detach() for w in pl_module.parameters()]
        for ema_w, w in zip(self.ema_weights, pl_module.parameters()):
            w.data.copy_(ema_w)

    def on_validation_end(self, trainer, pl_module):
        # restore model weights after validation
        for original_w, w in zip(self.original_weights, pl_module.parameters()):
            w.data.copy_(original_w)
