import pdb
from typing import Any, Dict

import pytorch_lightning as pl
import torch


class EMA(pl.Callback):
    def __init__(self, decay: float, use_ema_weights: bool = True):
        super().__init__()
        self.decay = decay
        self.ema_weights = None
        self.use_ema_weights = use_ema_weights

    def on_train_start(self, trainer, pl_module):
        self.ema_weights = [w.clone().detach() for w in pl_module.parameters()]
        # print(111)

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        alpha = self.decay
        for ema_w, w in zip(self.ema_weights, pl_module.parameters()):
            ema_w.data.mul_(alpha).add_((1 - alpha) * w.data)

    def on_validation_start(self, trainer, pl_module):
        # replace model weights with EMA weights for validation
        # print("on_val_start: oula!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if self.ema_weights is None:
            self.ema_weights = [w.clone().detach() for w in pl_module.parameters()]
        self.original_weights = [w.clone().detach() for w in pl_module.parameters()]
        for ema_w, w in zip(self.ema_weights, pl_module.parameters()):
            w.data.copy_(ema_w)

    def on_validation_end(self, trainer, pl_module):
        # restore model weights after validation
        # print("on_val_end: oula!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        for original_w, w in zip(self.original_weights, pl_module.parameters()):
            w.data.copy_(original_w)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # print("on_test_start: oula!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if self.ema_weights is None:
            self.ema_weights = [w.clone().detach() for w in pl_module.parameters()]
        self.original_weights = [w.clone().detach() for w in pl_module.parameters()]
        for ema_w, w in zip(self.ema_weights, pl_module.parameters()):
            w.data.copy_(ema_w)
    
    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # print("on_test_end: oula!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        for original_w, w in zip(self.original_weights, pl_module.parameters()):
            w.data.copy_(original_w)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: Dict[str, Any]) -> dict:
        # save the ema weights in the checkpoint
        checkpoint["state_dict_ema"] = self.ema_weights
        return checkpoint
    
    # won't work as a hook when validatinga and testing in the pytorch_lightning 1.5.0, you need to call mannully
    # So use it as user-defined function
    def on_load_checkpoint(self, checkpoint_path: str) -> None:
        # load the ema weights from the checkpoint
        # pdb.set_trace()
        print("on_load: oula!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        ckpt = torch.load(checkpoint_path)
        if self.use_ema_weights:
            self.ema_weights = ckpt["state_dict_ema"]
            # self.original_weights = [w.clone().detach() for w in pl_module.parameters()]
            # for ema_w, w in zip(self.ema_weights, pl_module.parameters()):
            #     w.data.copy_(ema_w)
