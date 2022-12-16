# Filter out the DEBUG messages
import logging
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)
import argparse
import os
import random
import numpy as np
import pytorch_lightning as pl
import torch
from modules.config._base import get_config
from pytorch_lightning.utilities.seed import seed_everything
from data.data_module import UnifiedDataModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.training_type import DeepSpeedPlugin, DDPPlugin
# from pytorch_lightning.callbacks import StochasticWeightAveraging
import pdb
from lightningmodule._base import get_task


def seed_globally(seed=None):
    if("PL_GLOBAL_SEED" not in os.environ):
        if(seed is None):
            seed = random.randint(0, np.iinfo(np.uint32).max)
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        logging.info(f'os.environ["PL_GLOBAL_SEED"] set to {seed}')

    # seed_everything is a bit log-happy
    seed_everything(seed)

def main(args):
    # initialize datamodule and lightning module
    if(args.seed is not None):
        seed_everything(args.seed, workers=True)

    # model config details now give the users to specify
    config = get_config(args.task)()

    ## The Dataset interfaces are not the same and lack a unified model initialize interface
    #TODO: Make the interfaces the same √
    model_module = get_task(args.task)(config)
    # data_module.setup()
    data_module = UnifiedDataModule(
        config=config.data,
        **vars(args)
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # data_module.setup()
    # the most commonly used plugins
    callbacks = []
    if(args.checkpoint_best_val):
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        mc = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="RefineDiff-epoch{epoch:02d}-f1_max{val/f1_max:.2f}",
            auto_insert_metric_name=False,
            monitor="val/f1_max",
            save_top_k=3,
            mode="max",
            save_last=True,
            save_on_train_epoch_end=False
        )
        callbacks.append(mc)

    if(args.log_lr):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
    
    # swa = StochasticWeightAveraging(swa_lrs=1e-2)   
        
    loggers = []
    if(args.wandb):
        wdb_logger = WandbLogger(
            save_dir=args.output_dir,
            name=args.experiment_name,
            id=args.wandb_id,
            version=args.wandb_id,
            project=args.wandb_project,
            **{"entity": args.wandb_entity,
               "group":args.wandb_group,
               "resume": "allow"},
        )
        loggers.append(wdb_logger)
    
    if (args.gpus is not None and args.gpus > 1) or args.num_nodes > 1 :
        strategy = DDPPlugin(find_unused_parameters=False)
    else:
        strategy = None

    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir=args.output_dir,
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
        accelerator="gpu"
    )

    if(args.resume_model_weights_only):
        ckpt_path = None
        # ckpt = torch.load(args.resume_from_ckpt)
        # pdb.set_trace()
        # model_module.load_from_checkpoint(args.resume_from_ckpt, config)
        model_module = model_module.load_from_checkpoint(args.resume_from_ckpt, config)
    else:
        ckpt_path = args.resume_from_ckpt
    
    if(args.encoder_model_checkpoint is not None):
        encoder_model_state_dict = torch.load(args.encoder_model_checkpoint)["state_dict"]
        pdb.set_trace()
        model_module.heads.load_state_dict(encoder_model_state_dict, strict=False)

    # model_module.preprocess()
    trainer.fit(
        model_module, 
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )

    trainer.save_checkpoint(
        os.path.join(args.output_dir, "checkpoints", "final.ckpt")
    )

def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Keep the same with the config file
    parser.add_argument("--task", type=str, default="refine_diff")
    parser.add_argument("--config_name", type=str, default="AlphaRefine-Local")
    # parser.add_argument("--data_class", type=str, default="EnzymeCommission")
    # Training config arguments, and these should be the most common arguments
    parser.add_argument(
        "--output_dir", type=str,
        help='''Directory in which to output checkpoints, logs, etc. Ignored
                if not on rank 0'''
    )
    parser.add_argument(
        "--seed", type=int, default=145,
        help="Random seed"
    )
    parser.add_argument(
        "--debug", type=bool_type, default=False
    )
    parser.add_argument(
        "--checkpoint_best_val", type=bool_type, default=True,
        help="""Whether to save the model parameters that perform best during
                validation"""
    )
    parser.add_argument(
        "--resume_from_ckpt", type=str, default=None,
        help="Path to a model checkpoint from which to restore training state"
    )
    parser.add_argument(
        "--encoder_model_checkpoint", type=str, default=None,
        help="Path to a pretrained Encoder model checkpoint"
    )
    parser.add_argument(
        "--resume_model_weights_only", type=bool_type, default=False,
        help="Whether to load just model weights as opposed to training state"
    )
    parser.add_argument(
        "--log_lr", action="store_true", default=False,
    )
    parser.add_argument(
        "--wandb", action="store_true", default=False,
    )
    parser.add_argument(
        "--experiment_name", type=str, default="RefineDiff_Debug",
    )
    parser.add_argument(
        "--wandb_id", type=str, default=None,
    )
    parser.add_argument(
        "--wandb_group", type=str, default=None,
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None,
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None,
    )
    # Datset config arguments, #TODO: Make the dataset config interface more unified
    # parser.add_argument(
    #     "--train_targets_path", type=str, default="/root/HibikeFold/RefineDiff-FullAtom/DecoyDataset/train_proteins.npy",
    #     help="the path to the list of training protein names"
    # )
    # parser.add_argument(
    #     "--val_targets_path", type=str, default="/root/HibikeFold/RefineDiff-FullAtom/DecoyDataset/valid_proteins.npy",
    #     help="the path to the list of validating protein names"
    # )
    # parser.add_argument(
    #     "--predict_targets_path", type=str, default=None,
    #     help="the path to the list of predicting protein names"
    # )
    # parser.add_argument(
    #     "--root_dir", type=str, default="/root/HibikeFold/RefineDiff-FullAtom/DecoyDataset/pdbs",
    #     help="the path to the list of data directory"
    # )
    # parser.add_argument(
    #     "--include_native", type=bool_type, default=False,
    #     help="whether to include native structure"
    # )


    parser = pl.Trainer.add_argparse_args(parser)

    # Disable the initial validation pass
    # parser.set_defaults(
    #     num_sanity_val_steps=0,
    # )
    
    args = parser.parse_args()

    main(args)
