import pytorch_lightning as pl
import torch

# from models.denoise_module import DenoiseModule
from lightning_module._base import register_task
from models._base import get_model
from utils import residue_constants
from utils.loss import RefineDiffLoss, lddt_ca
from utils.lr_scheduler import AlphaFoldLRScheduler
from utils.superimposition import superimpose
from utils.validation_metrics import drmsd, gdt_ha, gdt_ts


@register_task("RefineDiff")
class RefineDiffWrapper(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = get_model(config.globals.model_class)(config)
        self.loss = RefineDiffLoss(config.loss)
        self.last_lr_step = -1
        self.train_config = config.train
    
    def forward(self, batch):
        return self.model(batch)

    def _log(self, loss_breakdown, batch, outputs, train=True, test=False):
        if train:
            phase="train"
        elif test:
            phase="test"
        else:
            phase="val"

        for loss_name, indiv_loss in loss_breakdown.items():
            self.log(
                f"{phase}/{loss_name}", 
                indiv_loss, 
                on_step=train, on_epoch=(not train), logger=True,
            )

            if(train):
                self.log(
                    f"{phase}/{loss_name}_epoch",
                    indiv_loss,
                    on_step=False, on_epoch=True, logger=True,
                )

        with torch.no_grad():
            other_metrics = self._compute_validation_metrics(
                batch, 
                outputs,
                superimposition_metrics=(not train)
            )

        for k,v in other_metrics.items():
            self.log(
                f"{phase}/{k}", 
                v, 
                on_step=test, on_epoch=True, logger=True
            )
    
    def training_step(self, batch, batch_idx):

        # Run the model
        outputs = self(batch)

        # Compute loss
        loss, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        # Log it
        self._log(loss_breakdown, batch, outputs)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Run the model
        outputs = self(batch)
        _, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        self._log(loss_breakdown, batch, outputs, train=False)

    def test_step(self, batch, batch_idx):
        # Test the model
        outputs = self(batch)
        _, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        self._log(loss_breakdown, batch, outputs, train=False, test=True)


    def _compute_validation_metrics(self, 
        batch, 
        outputs, 
        superimposition_metrics=False
    ):
        metrics = {}
        
        gt_coords = batch["label_all_atom_positions"]
        pred_coords = outputs["final_atom_positions"]
        all_atom_mask = batch["label_all_atom_mask"]
        # Starting Model
        decoy_coords = batch["decoy_all_atom_positions"]

    
        # This is super janky for superimposition. Fix later
        gt_coords_masked = gt_coords * all_atom_mask[..., None]
        pred_coords_masked = pred_coords * all_atom_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = all_atom_mask[..., ca_pos]
        # Starting Model
        decoy_coords_masked = decoy_coords * all_atom_mask[..., None]
        decoy_coords_masked_ca = decoy_coords_masked[..., ca_pos, :]
    
        lddt_ca_score = lddt_ca(
            pred_coords,
            gt_coords,
            all_atom_mask,
            eps=self.config.globals.eps,
            per_residue=False,
        )

        starting_lddt = lddt_ca(
            decoy_coords,
            gt_coords,
            all_atom_mask,
            eps=self.config.globals.eps,
            per_residue=False,
        )
   
        metrics["lddt_ca"] = lddt_ca_score
        metrics["delta_lddt_ca"] = lddt_ca_score - starting_lddt
   
        drmsd_ca_score = drmsd(
            pred_coords_masked_ca,
            gt_coords_masked_ca,
            mask=all_atom_mask_ca, # still required here to compute n
        )
        
        starting_drmsd = drmsd(
            decoy_coords_masked_ca,
            gt_coords_masked_ca,
            mask=all_atom_mask_ca, # still required here to compute n
        )
   
        metrics["drmsd_ca"] = drmsd_ca_score
        metrics["delta_drmsd_ca"] = drmsd_ca_score - starting_drmsd
    
        if(superimposition_metrics):
            superimposed_pred, alignment_rmsd = superimpose(
                gt_coords_masked_ca, pred_coords_masked_ca, all_atom_mask_ca,
            )
            gdt_ts_score = gdt_ts(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )
            gdt_ha_score = gdt_ha(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )

            metrics["alignment_rmsd"] = alignment_rmsd
            metrics["gdt_ts"] = gdt_ts_score
            metrics["gdt_ha"] = gdt_ha_score

            superimposed_pred, alignment_rmsd = superimpose(
                gt_coords_masked_ca, decoy_coords_masked_ca, all_atom_mask_ca,
            )
            starting_gdt_ts = gdt_ts(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )
            starting_gdt_ha = gdt_ha(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )

            metrics["delta_gdt_ts"] = gdt_ts_score - starting_gdt_ts
            metrics["delta_gdt_ha"] = gdt_ha_score - starting_gdt_ha

    
        return metrics

    def configure_optimizers(self, 
        learning_rate: float = 1e-3,
        eps: float = 1e-5,
        ) -> torch.optim.Adam:

        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
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