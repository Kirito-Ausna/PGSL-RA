import pdb
from functools import partial
from typing import Any, Dict, List, Mapping, Optional

import ml_collections as mlc
import numpy as np
import pytorch_lightning as pl
import torch

from data.feature_pipeline import process_features
from utils.tensor_utils import dict_multimap

FeatureDict = Mapping[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]
from data.dataset._base import get_dataset


class BatchCollator:
    def __init__(self, config, stage="train") -> None:
        self.config = config
        self.stage = stage

    def __call__(self, raw_prots) -> FeatureDict:
        processed_prots = []
        # get the max sequence length in a batch
        max_seq_len = max([prot["decoy_seq_mask"].shape[0] for prot in raw_prots])

        for prot in raw_prots:
            features = process_features(prot, self.stage, self.config, max_seq_len)
            processed_prots.append(features)

        stack_fn = partial(torch.stack, dim=0)
        return dict_multimap(stack_fn, processed_prots) 


class UnifiedDataModule(pl.LightningDataModule):
    def __init__(self,
        config: mlc.ConfigDict,
        debug: bool = False,
        **kwargs
    ):
        self.config = config # data config
        self.debug = debug
    #TODO: Make a the dataset can be registered
    def setup(self, stage=None):
        Data = get_dataset(self.config.dataset.name)
        dataset_gen = partial(Data,
            config = self.config,
            debug = self.debug)
        self.training_mode = self.config.dataset.training_mode

        if self.training_mode:
            self.train_dataset = dataset_gen(mode = "train")
            if self.config.dataset.eval:
                self.val_dataset = dataset_gen(mode = "eval")
        else:
            self.test_dataset = dataset_gen(mode = "test")
    
    def _gen_batch_collator(self, stage):
        collate_fn = BatchCollator(self.config, stage)
        return collate_fn

    def train_dataloader(self):
        if(self.train_dataset is not None):
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.config.data_module.train_dataloader.batch_size,
                collate_fn=self._gen_batch_collator("train"),
                pin_memory=True,
            )

    def val_dataloader(self):
        if(self.val_dataset is not None):
            # pdb.set_trace()
            return torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.config.data_module.val_dataloader.batch_size,
                collate_fn=self._gen_batch_collator("eval"),
                pin_memory=True,
            )
        else:
            return None
    
    def test_dataloader(self):
        if (self.test_dataset is not None):
            return torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.config.data_module.predict_dataloader.batch_size,
                collate_fn=self._gen_batch_collator("predict"),
                pin_memory=True,
            )
    
            

        

        
            
        
        