import os
import pdb
from functools import partial
from typing import Any, Dict, List, Mapping, Optional

import ml_collections as mlc
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset

from data.dataset._base import register_dataset
from data.feature_pipeline import (process_decoy, process_features,
                                   process_label)
from utils import Utils
from utils.protein import get_seqs_from_pdb
from utils.tensor_utils import dict_multimap

FeatureDict = Mapping[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]

@register_dataset("DeepAccNet")
class Data(Dataset):
    def __init__(self, 
                config: mlc.ConfigDict, # data config
                mode: str = "train", 
                debug: bool = False,
                **kwargs
                ):
        super().__init__()
        self.dataset_config = config.dataset
        self.length = {}
        self.num_decoys = {}
        self.decoys_dict = {}
        self.root_dir = self.dataset_config.root_dir
        self.mode = mode
        self.include_native = self.dataset_config.include_native
        self.config = config
        self.gfeat_save_dir = config.dataset.gfeat_save_dir
        if not os.path.exists(self.gfeat_save_dir):
            os.makedirs(self.gfeat_save_dir)
        self.esm_save_dir = config.dataset.esm_save_dir
        self.feat_class = {'seq': {'node': ['rPosition',], 'edge': ['SepEnc']},
                           'struc': {'node': ['SS3', 'RSA', 'Dihedral'], 
                           'edge': ['Ca1-Ca2', 'Cb1-Cb2', 'N1-O2', 'Ca1-Cb1-Cb2', 'N1-Ca1-Cb1-Cb2', 'Ca1-Cb1-Cb2-Ca2']}}
        targets_attr = mode + "_targets_path"
        targets_path = getattr(self.dataset_config, targets_attr)
        targets = np.load(targets_path)
        if debug: targets = targets[:100]
        target_available = []
        
        # pdb.set_trace()
        for p in targets:
            path = self.root_dir + p
            if os.path.exists(path):
                sample_files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            else:
                continue
            # Removing native if necessasry. This is not a default behavior
            if not self.include_native:
                len_sample_files = len(sample_files)
                # pdb.set_trace()
                sample_files = [s for s in sample_files if s.split("/")[-1] != "native.pdb"]
                # pdb.set_trace()
                len_new = len(sample_files)
                label_available = True if (len_sample_files - len_new) == 1 else False
            np.random.shuffle(sample_files)
            samples = sample_files
            if len(samples) > 0 and label_available:
                target_available.append(p)
                length = len(get_seqs_from_pdb(samples[0]))
                self.decoys_dict[p] = samples
                self.num_decoys[p] = len_new
                self.length[p] = length
        # Make a list of targets which are all corresponding to many decoys for training.
        self.targets = target_available


    def __len__(self):
        return len(self.targets)
        # Not the actual number of training samples
    
    # A Dict contains various features with defined names
    def _process_decoy(self, path, gnn_feature, seq, chain_id: Optional[str] = None):
        data = process_decoy(path, gnn_feature, seq, self.config.decoy)
        return data

    def _process_label(self, path, feats, chain_id: Optional[str] = None):
        """This function will handle the error of path not exists"""
        feats = process_label(pdb_path=path, chain_id=chain_id, feats=feats)
        return feats

    def __get_seq_feature(self, pdb_file, pname):
        seq = Utils.get_seqs_from_pdb(pdb_file)
        # node_feat
        node_feat = {
            # 'onehot': Utils.get_seq_onehot(seq),
            'rPosition': Utils.get_rPos(seq),
            # 'esm':Utils.get_esm_embedding(seq, pname)
        }
        # edge_feat
        edge_feat = {
            'SepEnc': Utils.get_SepEnc(seq),
        }
        return node_feat, edge_feat, len(seq), seq

    def __get_struc_feat(self, pdb_file, seq_len):
        # node feat
        node_feat = Utils.get_DSSP_label(pdb_file, [1, seq_len])
        # atom_emb
        embedding = Utils.get_atom_emb(pdb_file, [1, seq_len])
        node_feat['atom_emb'] = {
            'embedding': embedding.astype(np.float32),
        }
        # edge feat
        edge_feat = Utils.calc_geometry_maps(pdb_file, [1, seq_len], self.feat_class['struc']['edge'])
        return node_feat, edge_feat


    def __getitem__(self, index , pindex=0):
        if torch.is_tensor(index): index = index.tolist()
        # Get target name
        pname = self.targets[index]
        if pindex == -1:
            pindex = np.random.choice(np.arange(self.num_decoys[pname]))
        # Get decoy path
        pdb_file = self.decoys_dict[pname][pindex]
        seq_len = self.length[pname]
        # Add GNNRefine Features, node as single feature, edge as pair feature
        feature = {"node": None, "edge": None}
        file_path = os.path.join(self.gfeat_save_dir, pname + ".pt")
        if not os.path.exists(file_path):
            # seq feature
            seq_node_feat, seq_edge_feat, seq_len, seq = self.__get_seq_feature(pdb_file, pname)
            for _feat in self.feat_class['seq']['node']:
                feature['node'] = seq_node_feat[_feat] if feature['node'] is None else np.concatenate((feature['node'], seq_node_feat[_feat]), axis=-1)
            # print(feature['node'].shape)
            for _feat in self.feat_class['seq']['edge']:
                feature['edge'] = seq_edge_feat[_feat] if feature['edge'] is None else np.concatenate((feature['edge'], seq_edge_feat[_feat]), axis=-1)
            # struc feature
            struc_node_feat, struc_edge_feat = self.__get_struc_feat(pdb_file, seq_len)
            for _feat in self.feat_class['struc']['node']:
                feature['node'] = struc_node_feat[_feat] if feature['node'] is None else np.concatenate((feature['node'], struc_node_feat[_feat]), axis=-1)
            for _feat in self.feat_class['struc']['edge']:
                feature['edge'] = struc_edge_feat[_feat] if feature['edge'] is None else np.concatenate((feature['edge'], struc_edge_feat[_feat]), axis=-1)
            # feature
            feature = np.nan_to_num(feature)
            feature['node'] = feature['node'].astype(np.float32)
            feature['edge'] = feature['edge'].astype(np.float32)
            feature['atom_emb'] = struc_node_feat['atom_emb']['embedding']
            # feature['esm_emb'] = seq_node_feat['esm']
            # Prepare Decoy Feature
            feats = self._process_decoy(pdb_file, feature, seq)
            # Prepare label
            pdb_label_file = os.path.split(pdb_file)[0] + '/' + 'native.pdb'
            # Prepare groundtruth Feature
            feats = self._process_label(pdb_label_file, feats)
            torch.save(feats, file_path)
        else:
            feats = torch.load(file_path)
            seq=None
        # ESM feature
        # feats['esm_emb'] = Utils.get_esm_embedding(seq, pname, self.esm_save_dir)
        

        return feats