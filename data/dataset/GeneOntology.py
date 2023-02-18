import csv
import glob
import os
import pdb
from functools import partial
from typing import Any, Dict, List, Mapping, Optional

import ml_collections as mlc
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torchdrug import utils

from data.dataset._base import register_dataset
from data.feature_pipeline import process_decoy
from utils import Utils

FeatureDict = Mapping[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]

@register_dataset("GO")
class Data(Dataset):
    """
    A set of proteins with their 3D structures and GO terms. These terms classify proteins 
    into hierarchically related functional classes organized into three ontologies: molecular 
    function (MF), biological process (BP) and cellular component (CC).

    Statistics (test_cutoff=0.95):
        - #Train: 27,496
        - #Valid: 3,053
        - #Test: 2,991

    Parameters:
        path (str): the path to store the dataset
        branch (str, optional): the GO branch
        test_cutoff (float, optional): the test cutoff used to split the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    # url = "https://zenodo.org/record/6622158/files/GeneOntology.zip"
    # md5 = "376be1f088cd1fe720e1eaafb701b5cb"
    branches = ["MF", "BP", "CC"]
    test_cutoffs = [0.3, 0.4, 0.5, 0.7, 0.95]
    def __init__(self,
                config: mlc.ConfigDict,
                mode: str="train",
                debug: bool=False,
                **kwargs):
        super().__init__()
        root_dir = config.dataset.root_dir
        test_cutoff = config.dataset.test_cutoff
        branch = config.dataset.branch
        self.gfeat_save_dir = config.dataset.gfeat_save_dir
        self.esm_save_dir = config.dataset.esm_save_dir
        if not os.path.exists(self.gfeat_save_dir):
            os.makedirs(self.gfeat_save_dir)
        root_dir = os.path.expanduser(root_dir)
        self.path = root_dir
        if branch not in self.branches:
            raise ValueError("Unknown branch `%s` for GeneOntology dataset" % branch)
        self.branch = branch
        if test_cutoff not in self.test_cutoffs:
            raise ValueError("Unknown test cutoff `%.2f` for EnzymeCommission dataset" % test_cutoff)
        self.test_cutoff = test_cutoff
        self.mode = mode
        self.config = config
        self.debug = debug
        self.feat_class = {'seq': {'node': ['rPosition',], 'edge': ['SepEnc']}, 
                           'struc': {'node': ['SS3', 'RSA', 'Dihedral'], 
                           'edge': ['Ca1-Ca2', 'Cb1-Cb2', 'N1-O2', 'Ca1-Cb1-Cb2', 'N1-Ca1-Cb1-Cb2', 'Ca1-Cb1-Cb2-Ca2']}}
        exlude_pdb_ids = []
        tsv_file = os.path.join(self.path, "nrPDB-GO_annot.tsv") # MulticlassBinaryClassification label
        if self.mode == "predict":
            csv_file = os.path.join(self.path, "nrPDB-GO_test.csv")
            with open(csv_file, "r") as fin:
                reader = csv.reader(fin, delimiter=",")
                idx = self.test_cutoffs.index(test_cutoff) + 1
                _ = next(reader)
                for line in reader:
                    if line[idx] == "0": #Note: It's proteins that are not included
                        exlude_pdb_ids.append(line[0])
            path = os.path.join(self.path, "test")
        elif self.mode == "train":
            path = os.path.join(self.path, "train")
        else: #TODO: add validation, currently use test as validation
            csv_file = os.path.join(self.path, "nrPDB-GO_test.csv")
            with open(csv_file, "r") as fin:
                reader = csv.reader(fin, delimiter=",")
                idx = self.test_cutoffs.index(test_cutoff) + 1
                _ = next(reader)
                for line in reader:
                    if line[idx] == "0": #Note: It's proteins that are not included
                        exlude_pdb_ids.append(line[0])
            path = os.path.join(self.path, "test")
        self.pdb_files = glob.glob(os.path.join(path, "*.pdb")) #pdb file list in train/valid/test set
        if self.debug:
            self.pdb_files = self.pdb_files[:100]
        if len(exlude_pdb_ids) > 0:
            self.filter_pdb(exlude_pdb_ids)# filter out similar proteins in test set
        self.dicard_nonstandard_pdb()
        pdb_ids = [os.path.basename(pdb_file).split("_")[0] for pdb_file in self.pdb_files]
        self.pdb_ids = pdb_ids
        self.load_annotation(tsv_file, pdb_ids)
        
    def filter_pdb(self, exclude_pdb_ids):
        exclude_pdb_ids = set(exclude_pdb_ids)
        pdb_files = []
        for pdb_file in self.pdb_files:
            if os.path.basename(pdb_file).split("_")[0] in exclude_pdb_ids:
                continue
            pdb_files.append(pdb_file)

        self.pdb_files = pdb_files
    
    def dicard_nonstandard_pdb(self):
        pdb_files = []
        for pdb_file in self.pdb_files:
            pdb_id = os.path.basename(pdb_file).split("_")[0]
            chain_id = pdb_id.split("-")[-1]
            if len(chain_id) != 1:
                continue
            pdb_files.append(pdb_file)
        self.pdb_files = pdb_files

    def load_annotation(self, tsv_file, pdb_ids):
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
        return len(self.pdb_files)
    
    @property
    def tasks(self):
        """List of tasks."""
        return list(self.targets.keys())

    # A Dict contains various features with defined names
    def _process_decoy(self, path, gnn_feature, seq, chain_id: Optional[str] = None):
        data = process_decoy(path, gnn_feature, seq, self.config.decoy)
        return data

    def __get_seq_feature(self, pdb_file, pname):
        seq = Utils.get_seqs_from_pdb(pdb_file)
        seq = seq.replace('X','')
        # pdb.set_trace()
        # node_feat
        # save_path = "/usr/commondata/local_public/protein-datasets/EnzymeCommission/ESMFeature/"
        # save_path = os.path.dirname(self.path) + "/ESMFeature/"
        node_feat = {
            # 'onehot': Utils.get_seq_onehot(seq),
            'rPosition': Utils.get_rPos(seq),
            # 'esm':Utils.get_esm_embedding(seq, pname, save_path)
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
        # Utils.get_atom_emb(pdb_file, [1, seq_len])
        node_feat['atom_emb'] = {
            'embedding': embedding.astype(np.float32),
        }
        # edge feat
        edge_feat = Utils.calc_geometry_maps(pdb_file, [1, seq_len], self.feat_class['struc']['edge'])
        # return None
        return node_feat, edge_feat

    def __getitem__(self, index):
        # return 1
        if torch.is_tensor(index): index = index.tolist()
        # Get target name
        pname = self.pdb_ids[index]
        # Get decoy path
        pdb_file = self.pdb_files[index]
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
            # self.__get_struc_feat(pdb_file, seq_len)
            for _feat in self.feat_class['struc']['node']:
                feature['node'] = struc_node_feat[_feat] if feature['node'] is None else np.concatenate((feature['node'], struc_node_feat[_feat]), axis=-1)
            for _feat in self.feat_class['struc']['edge']:
                feature['edge'] = struc_edge_feat[_feat] if feature['edge'] is None else np.concatenate((feature['edge'], struc_edge_feat[_feat]), axis=-1)
          
            feature = np.nan_to_num(feature)
            feature['node'] = feature['node'].astype(np.float32)
            feature['edge'] = feature['edge'].astype(np.float32)
            feature['atom_emb'] = struc_node_feat['atom_emb']['embedding']
            feats = self._process_decoy(pdb_file, feature, seq)
            torch.save(feats, file_path)
        else:
            feats = torch.load(file_path)
            seq = None
        # ESM feature
        # feats['esm_emb'] = Utils.get_esm_embedding(seq, pname, self.esm_save_dir)
        # Prepare groundtruth
        indices = self.pos_targets[index].unsqueeze(0)
        values = torch.ones(len(self.pos_targets[index]))
        # pdb.set_trace()
        feats["targets"] = utils.sparse_coo_tensor(indices, values, (len(self.tasks),)).to_dense()
        
        
        return feats
