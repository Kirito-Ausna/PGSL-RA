import os
import pdb

import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn import functional as F

from task_framework.property_prediction import MultiBinaryClassifyHead
from lightning_module._base import register_task
from models._base import get_model
from modules import losses, metrics
from utils.lr_scheduler import AlphaFoldLRScheduler


@register_task("mbclassify")
class MultiBClassifyWrapper(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = get_model(config.downstream.encoder)(config)
        self.heads = MultiBinaryClassifyHead(**config.downstream.head)
        if config.downstream.encoder_fixed:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.tasks = self.config.downstream.head.task_num
        self.last_lr_step = -1
        self.metrics = config.downstream.metric
        self.define_metrics()
        self.preprocess()
        self.train_config = config.train

    def define_metrics(self):
        modes = []
        if self.config.data.dataset.training_mode:
            modes.extend(["train", "val"])
        if self.config.data.dataset.test:
            modes.extend(["tm_test", "plddt_test"])
        if self.config.data.dataset.test.ground_truth:
            modes.append("ground_truth_test")
        for _metric in self.metrics:
            if _metric == "auroc_micro":
                for mode in modes:
                    setattr(self, f"{mode}_auroc_micro", metrics.classification_metric("area_under_roc")) 
            elif _metric == "f1_max":
                for mode in modes:
                    setattr(self, f"{mode}_f1_max", metrics.classification_metric("f1_max"))
            elif _metric == "auprc_micro":
                for mode in modes:
                    setattr(self, f"{mode}_auprc_micro", metrics.classification_metric("area_under_prc"))
            elif _metric == "acc":
                for mode in modes:
                    setattr(self, f"{mode}_acc", metrics.classification_metric("accuracy"))
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)
    
    def preprocess(self, train_set=None):
        """
        Compute the weight for each task on the training set.
        """
        save_path = self.config.data.dataset.root_dir
        task_weight_path = save_path + "task_weight.pt"
        pos_weight_path = save_path + "pos_weight.pt"
        
        if self.config.downstream.reweight:
            if not os.path.exists(task_weight_path):
                values = []
                train_size = len(train_set)
                for data in train_set:
                    values.append(data["targets"])
                values = torch.stack(values, dim=0)
                num_pos = values.sum(dim=0)
                task_weight = (num_pos.mean() / num_pos).clamp(1, 60)
                pos_weight = ( train_size - num_pos / num_pos).clamp(1, 60)
                torch.save(task_weight, task_weight_path)
                torch.save(pos_weight, pos_weight_path)
                # pdb.set_trace()
            else:
                task_weight = torch.load(task_weight_path)
                pos_weight = torch.load(pos_weight_path)
        else:
            task_weight = torch.ones(self.tasks, dtype=torch.float)
            pos_weight = torch.ones(self.tasks, dtype=torch.float)
        
        self.register_buffer("weight", task_weight)
        self.register_buffer("pos_weight", pos_weight)



    def forward(self, batch):
        # return self.heads(batch)
        node_repr = self.encoder(batch)
        return self.heads(batch, node_repr)
    
    def training_step(self, batch, batch_idx):
        # pdb.set_trace()
        pred = self.forward(batch)
        target = batch["targets"]
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none", pos_weight=self.pos_weight)
        loss = loss.mean(dim=0)
        loss = (loss * self.weight).sum() / self.weight.sum()
        # loss = losses.sigmoid_focal_loss(pred, target, reduction="mean")
        self._log(loss, target, pred, train=True)
        return loss
        # print("Training step")
    
    def validation_step(self, batch, batch_idx):
        
        pred = self.forward(batch)
        target = batch["targets"]
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none", pos_weight=self.pos_weight)
        loss = loss.mean(dim=0)
        loss = (loss * self.weight).sum() / self.weight.sum()
        # loss = losses.sigmoid_focal_loss(pred, target, reduction="mean")
        # pdb.set_trace()
        self._log(loss, target, pred, train=False)
        # print("Validation step")

    def test_step(self, batch, batch_idx, dataloader_idx):
        # Test step is the same as validation step
        # self.validation_step(batch, batch_idx)
        pred = self.forward(batch)
        target = batch["targets"]
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none", pos_weight=self.pos_weight)
        loss = loss.mean(dim=0)
        loss = (loss * self.weight).sum() / self.weight.sum()
        # loss = losses.sigmoid_focal_loss(pred, target, reduction="mean")
        # pdb.set_trace()
        test_set = ["tm_test", "plddt_test", "ground_truth_test"]
        self._log(loss, target, pred, train=False, test=True, test_set=test_set[dataloader_idx])

    def _log(self, loss, target, pred, train=True, test=False, test_set=None):
        # phase = "train" if train else "val"
        if train:
            phase="train"
        elif test:
            phase=test_set
        else:
            phase="val"
        self.log(f"{phase}/bce", loss, on_step=train or test, on_epoch=(not train), logger=True, add_dataloader_idx=False)
        if(train):
            self.log(
                f"{phase}/bce_epoch",
                loss,
                on_step=False, on_epoch=True, logger=True,
            )
        #Compute metrics in one epoch
        with torch.no_grad():
           self.evaluate(pred, target, phase, test)
 
    def evaluate(self, pred, target, phase, test):
        for _metric in self.metrics:
            getattr(self, f"{phase}_{_metric}").update(pred, target)
            # if test:
            self.log(f"{phase}/{_metric}", getattr(self, f"{phase}_{_metric}"), on_step=False, on_epoch=True, logger=True, prog_bar=True, add_dataloader_idx=False)
            # else:
            #     self.log(f"{phase}/{_metric}", getattr(self, f"{phase}_{_metric}"), on_step=False, on_epoch=True, logger=True, prog_bar=True)
        

    def configure_optimizers(self, 
        learning_rate: float = 1e-3,
        eps: float = 1e-8,
        ) -> torch.optim.Adam:

        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=learning_rate, 
            eps=eps
        )

        if self.last_lr_step != -1:
            for group in optimizer.param_groups:
                if 'initial_lr' not in group:
                    group['initial_lr'] = learning_rate

        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10)
        lr_scheduler = AlphaFoldLRScheduler(
            optimizer,
            **self.train_config,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "AlphaFoldLRScheduler",
            }
        }

    def resume_last_lr_step(self, lr_step):
        self.last_lr_step = lr_step




        

        
