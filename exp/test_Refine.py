import logging

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)
import argparse
import os
import pdb
import random

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import WandbLogger
from config._base import get_config
from data.data_module import UnifiedDataModule
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
    config = get_config(args.config_name)()
    model_moule = get_task(args.task)(config)
    data_module = UnifiedDataModule(
        config=config.data,
        **vars(args)
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
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
    
    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir=args.output_dir,
        logger=loggers,
        accelerator="gpu"
    )

    if args.resume_from_checkpoint is not None:
        model_moule = model_moule.load_from_checkpoint(args.resume_from_checkpoint, config)
    trainer.test(model=model_moule, datamodule=data_module)

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
    parser.add_argument("--task", type=str, default="refine_diff")
    parser.add_argument("--config_name", type=str, default="AlphaRefineTest-Cluster")
    parser.add_argument(
        "--output_dir", type=str, default="/huangyufei/DiffSE/test_result/AlphaRefine/DeepACCNet/",
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
        "--resume_from_ckpt", type=str, default="/huangyufei/RefineDiff/RefineDiff-SM/train_result/ScoreMatching/RefineDiff_Debug/RefineDiff-epoch95-val_gdt0.66.ckpt",
        help="Path to a model checkpoint from which to restore training state"
    )
    #Logger
    parser.add_argument(
        "--wandb", action="store_true", default=False,
    )
    parser.add_argument(
        "--experiment_name", type=str, default="AlphaRefineTest-DeepACCNet",
    )
    parser.add_argument(
        "--wandb_id", type=str, default="ARTest-DeepACCNet",
    )
    parser.add_argument(
        "--wandb_group", type=str, default="AlphaRefineTest",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="RefineDiff",
    )
    parser.add_argument(
        "--wandb_entity", type=str, default="kirito_asuna",
    )
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)

