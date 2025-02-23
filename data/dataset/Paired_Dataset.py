import csv
import glob
import os
import pdb
import pickle
from typing import Dict, Mapping, Optional

import lmdb
import ml_collections as mlc
import numpy as np
import torch
from torch.utils.data import Dataset
from torchdrug import utils

from data.data_transform import random_crop_to_size
from data.dataset._base import register_dataset

FeatureDict = Mapping[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]

@register_dataset("Paired")
class PairedDataset(Dataset):
    """
    Dataset for paired (or single) protein strcuture data based on preprocessed lmdb files of GO and EC. This can satisfy all tasks included in PGSL-RPA

    Basic config is as follows:
        - pdb_ids: list of pdb ids to be loaded
        - data_dir: directories to the lmdb files (Predicted or GroundTruth)    
        - (optional) labels: list of labels to be loaded
    All the configs will be saved in config.py 
    Focous on the implementatiion of __getitem__ to acheive the paired data loading 
    and generation to single source data. Not the data preprocess which is done in another script.
    """
    MAP_SIZE = 512*(1024*1024*1024) # 512 GB
    def __init__(self,
                 config: mlc.ConfigDict,
                 mode: str="train",
                 debug: bool=False,
                 **kwargs):
        super().__init__()
        self.config = config
        self.mode = mode
        self.debug = debug
        
        self.task = config.dataset.task
        self.paired = config.dataset.paired
        self.pred = config.dataset.pred # Use when paired is False
        self.root_dir = config.dataset.root_dir
        self.framework = config.dataset.framework
        if self.framework == "SAO":
            mask_setting = config.dataset.mask_setting
            self.mask_prob = mask_setting.mask_prob
            self.leave_unmasked_prob = mask_setting.leave_unmasked_prob
            self.random_token_prob = mask_setting.random_token_prob
            # convert to tensor
            self.mask_index = torch.tensor(21) # 21 is the index of mask token in the vocabulary

        self.exp_lmdb_path = None
        self.pred_lmdb_path = None
        self.exp_db = None
        self.pred_db = None
        self.pos_targets = None
        self.error_targets = None

        self.stage = mode

        if self.task == "GO":
            # should load labels and help find the lmdb files
            self.branch = config.dataset.branch
            self.branches = ["MF", "BP", "CC"]
        
        align_processed_path = os.path.join(self.root_dir, "align_processed")
        processed_path = os.path.join(self.root_dir, "processed")
        paired_eval_frameworks = ["SAO", "PGSL"]
        if mode == "test":
            self.exp_lmdb_path = os.path.join(processed_path,"exp_struct", f"{self.task}_{self.mode}")
            index_file = os.path.join(self.exp_lmdb_path, "structures.lmdb-ids")
            self.stage = "predict"
        elif mode in ["tm_test", "plddt_test"]:
            self.pred_lmdb_path = os.path.join(processed_path,"pred_struct", f"{self.task}_{self.mode}")
            index_file = os.path.join(self.pred_lmdb_path, "structures.lmdb-ids")
            self.stage = "predict"
        elif self.paired and mode == "train" or (self.paired and mode == "eval" and self.framework in paired_eval_frameworks):
            # pdb.set_trace()
            self.exp_lmdb_path = os.path.join(align_processed_path,"exp_struct", f"{self.task}_{self.mode}")
            self.pred_lmdb_path = os.path.join(align_processed_path, "pred_struct", f"{self.task}_{self.mode}")
            index_file = os.path.join(self.pred_lmdb_path, "structures.lmdb-ids")
            self.error_targets = os.path.join(self.pred_lmdb_path, "error_pdb_id.pkl")
        elif self.pred:
            self.pred_lmdb_path = os.path.join(processed_path, "pred_struct", f"{self.task}_{self.mode}")
            index_file = os.path.join(self.pred_lmdb_path, "structures.lmdb-ids")
        else:
            self.exp_lmdb_path = os.path.join(processed_path, "exp_struct", f"{self.task}_{self.mode}")
            index_file = os.path.join(self.exp_lmdb_path, "structures.lmdb-ids")


        with open(index_file, "rb") as fin:
            self.pdb_ids = pickle.load(fin)
        if self.error_targets is not None: # remove the error targets with wrong alignment
            with open(self.error_targets, "rb") as fin:
                self.error_targets = pickle.load(fin)
            new_pdb_ids = []
            for pdb_id in self.pdb_ids:
                if pdb_id not in self.error_targets:
                    new_pdb_ids.append(pdb_id)
            self.pdb_ids = new_pdb_ids
        
        self.crop_size = config[self.stage].crop_size
        self.crop_feats = dict(config.common.feat)
        self.crop_transform = random_crop_to_size(self.crop_size, self.crop_feats)

        if self.debug:
            self.pdb_ids = self.pdb_ids[:100]
        
        self._connect_db()
        if self.framework not in paired_eval_frameworks:
            self.tsv_file = os.path.join(self.root_dir, f"nrPDB-{self.task}_annot.tsv")
            self.load_anontation(self.tsv_file, self.pdb_ids)

    def _connect_db(self):
        if self.exp_lmdb_path and self.exp_db is None:
            self.exp_db = lmdb.open(os.path.join(self.exp_lmdb_path, 'structures.lmdb'),
                                    map_size=self.MAP_SIZE,
                                    create=False,
                                    subdir=False, 
                                    readonly=True, 
                                    lock=False, 
                                    readahead=False, 
                                    meminit=False)
        if self.pred_lmdb_path and self.pred_db is None:
            self.pred_db = lmdb.open(os.path.join(self.pred_lmdb_path, 'structures.lmdb'),
                                    map_size=self.MAP_SIZE,
                                    create=False,
                                    subdir=False,  
                                    readonly=True, 
                                    lock=False, 
                                    readahead=False, 
                                    meminit=False)      
    
    def load_anontation(self, tsv_file, pdb_ids):
        if self.task == 'EC':
            with open(tsv_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                _ = next(reader)
                tasks = next(reader)
                task2id = {task: i for i, task in enumerate(tasks)}
                _ = next(reader)
                pos_targets = {}
                for pdb_id, pos_target in reader:
                    pos_target = [task2id[t] for t in pos_target.split(",")]
                    pos_target = torch.tensor(pos_target)
                    pos_targets[pdb_id] = pos_target
        else:
            idx = self.branches.index(self.branch)
            with open(tsv_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                for i in range(12):
                    _ = next(reader)
                    if i == idx * 4 + 1:
                        tasks = _
                task2id = {task: i for i, task in enumerate(tasks)}
                _ = next(reader)
                pos_targets = {}
                for line in reader:
                    pdb_id, pos_target = line[0], line[idx + 1] if idx + 1 < len(line) else None
                    pos_target = [task2id[t] for t in pos_target.split(",")] if pos_target else []
                    pos_target = torch.LongTensor(pos_target)
                    pos_targets[pdb_id] = pos_target

        # fake targets to enable the property self.tasks
        self.targets = task2id
        self.pos_targets = []
        for pdb_id in pdb_ids:
            self.pos_targets.append(pos_targets[pdb_id])
        
    def __len__(self):
        return len(self.pdb_ids)
    
    @property
    def tasks(self):
        return list(self.targets.keys())
    
    def _get_structure(self, pname):
        exp_data, pred_data = None, None
        if self.exp_db:
            with self.exp_db.begin(write=False) as txn:
                exp_data = pickle.loads(txn.get(pname.encode()))
        
        if self.pred_db:
            with self.pred_db.begin(write=False) as txn:
                pred_data = pickle.loads(txn.get(pname.encode()))

        return exp_data, pred_data
    
    def _create_mask_view(self, feat):
            aatype = feat["decoy_aatype"].numpy()
            exp_angle_feats = feat["exp_angle_feats"]
            sz = len(aatype)
            num_mask = int(self.mask_prob * sz + np.random.rand()) # add a random number for probabilistic rounding 
            mask_idc = np.random.choice(sz, num_mask, replace=False)
            mask = np.full(sz, False)
            mask[mask_idc] = True
            feat["mask_targets"] = np.full(sz, -1) # -1 is the ignore index value for masked tokens
            feat["mask_targets"][mask] = aatype[mask]
            # decide unmasking and random replacement
            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
            if rand_or_unmask_prob > 0.0:
                rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
                if self.random_token_prob == 0.0:
                    unmask = rand_or_unmask
                    rand_mask = None
                elif self.leave_unmasked_prob == 0.0:
                    unmask = None
                    rand_mask = rand_or_unmask
                else:
                    unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                    decision = np.random.rand(sz) < unmask_prob
                    unmask = rand_or_unmask & decision
                    rand_mask = rand_or_unmask & (~decision)
            else:
                unmask = rand_mask = None

            if unmask is not None:
                mask = mask ^ unmask
            
            mask_aatype = np.copy(aatype)
            mask_aatype[mask] = self.mask_index
            # copy the exp_angle_feats( in tensor)
            mask_angle_feats = torch.clone(exp_angle_feats)
            # pdb.set_trace()
            mask_angle_feats[mask] = 0.0 # clear the sidechain angle features for masked residues
            mask_angle_feats[mask][:,:22] = torch.nn.functional.one_hot(self.mask_index, num_classes=22) # set the one-hot encoding of the mask token

            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    mask_aatype[rand_mask] = np.random.randint(0, 20, num_rand)
                    rand_aatype = torch.from_numpy(mask_aatype[rand_mask])
                    mask_angle_feats[rand_mask][:,:22] = torch.nn.functional.one_hot(rand_aatype, num_classes=22)


            feat["mask_aatype"] = torch.from_numpy(mask_aatype)
            feat["mask_angle_feats"] = mask_angle_feats
            feat["mask_targets"] = torch.from_numpy(feat["mask_targets"]).long()

            return feat
            

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        pname = self.pdb_ids[index]
        exp_data, pred_data = self._get_structure(pname)
        # sequence_length = len(exp_data["decoy_seq_mask"])
        
        feats = {}

        if exp_data is not None and pred_data is not None:
            # feats = {**exp_data, **pred_data}
            if self.framework == "PGSL":
                # randomly choose one structure as the input
                if np.random.rand() > 0.5: # mix up the experimental and predicted data
                    feats = pred_data # Align the predicted data and experimental data, refinement task
                else:
                    feats = exp_data # Autoencoder task for experimental data to guide the encoder learn correct experimental data's embeddings
                feats['label_bb_rigid_tensors'] = exp_data["bb_rigid_tensors"] # The goal is to align the predicted data to the experimental data(directional) 
                                                                            # and maintain the representive power of experimental data for downstream tasks learning 
            elif self.framework == "Noisy_Training":
                # randomly choose one structure as the input
                if np.random.rand() > 0.5:
                    feats = pred_data
                else:
                    feats = exp_data
            elif self.framework == "SAO":
                # feats["predicted"] = pred_data
                # feats["experimental"] = exp_data
                # Create new feature to generate the input not the concatenation of two inputs
                feats = pred_data
                # Only the neccessary features for SAO for the sake of simplicity
                feats["exp_all_atom_positions"] = exp_data["decoy_all_atom_positions"]
                feats["exp_angle_feats"] = exp_data["decoy_angle_feats"]
                feats["label_bb_rigid_tensors"] = exp_data["bb_rigid_tensors"]
                sequence_length = feats["decoy_seq_mask"].shape[0]
                if self.crop_size is not None and sequence_length > self.crop_size:
                    feats = self.crop_transform(feats)
                # add mask after cropping
                feats = self._create_mask_view(feats)

        elif exp_data is None:
            feats = pred_data
        else:
            feats = exp_data

        # crop the features
        sequence_length = feats["decoy_seq_mask"].shape[0]
        if self.framework != "SAO" and self.crop_size is not None and sequence_length > self.crop_size:
            feats = self.crop_transform(feats)
        # not PGSL pretraining step
        if self.pos_targets is not None:
            indices = self.pos_targets[index].unsqueeze(0)
            values = torch.ones(len(self.pos_targets[index]))
            feats["targets"] = utils.sparse_coo_tensor(indices, values, (len(self.tasks),)).to_dense()

        return feats