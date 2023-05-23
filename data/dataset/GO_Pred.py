import csv
import pandas as pd
import glob
import logging
import os
import sys
sys.path.append("/root/Generative-Models/PGSL-RA")

import pdb
from typing import Dict, Mapping, Optional

import ml_collections as mlc
import numpy as np
import torch
from torch.utils.data import Dataset
from torchdrug import utils

from data.dataset._base import register_dataset
from data.pipeline_base import get_pipeline
# from utils import Utils
from tqdm.auto import tqdm
import lmdb
import joblib
import pickle
FeatureDict = Mapping[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]

@register_dataset("GO_Pred")
class GOPredData(Dataset):
    """
        A set of proteins with their predicted GO Structures and GO terms
    """
    branches = ["MF", "BP", "CC"]
    test_cutoffs = [0.3, 0.4, 0.5, 0.7, 0.95]
    MAP_SIZE = 512*(1024*1024*1024)  # 512GB
    def __init__(self,
                 config: mlc.ConfigDict,
                 mode: str="train",
                 debug: bool=False,
                 reset: bool=False,
                 **kwargs):
        super().__init__()
        self.config = config
        self.mode = mode
        self.debug = debug
        self.feature_pipeline = config.dataset.feature_pipeline

        self.data_dir = config.dataset.root_dir
        root_dir = os.path.expanduser(self.data_dir)
        self.path = root_dir

        test_cutoff = config.dataset.test_cutoff
        if test_cutoff not in self.test_cutoffs:
            raise ValueError("Unknown test cutoff `%.2f` for GeneOntology dataset" % test_cutoff)
        self.test_cutoff = test_cutoff
        self.tm_cutoff = config.dataset.tm_cutoff
        self.plddt_cutoff = config.dataset.plddt_cutoff
        if self.mode == "tm_test":
            self.tm_cutoff = config.dataset.test.tm_cutoff
            self.plddt_cutoff = None
        elif self.mode == "plddt_test":
            self.plddt_cutoff = config.dataset.test.plddt_cutoff
            self.tm_cutoff = None

        branch = config.dataset.branch
        if branch not in self.branches:
            raise ValueError("Unknown branch `%s` for GeneOntology dataset" % branch)
        self.branch = branch

        if config.dataset.processed_dir is None:
            self.feature_save_mode = False
        else:
            self.feature_save_mode = True
            if self.mode in ["train", "eval"]:
                self.processed_dir = os.path.join(config.dataset.processed_dir, "GO_tm_"+self.mode)#default Filter according to TMScore
                if self.plddt_cutoff is not None:
                    self.processed_dir = os.path.join(config.dataset.processed_dir, "GO_plddt_"+self.mode)
                if not os.path.exists(self.processed_dir):
                    os.makedirs(self.processed_dir)
            else:
                self.processed_dir = os.path.join(config.dataset.processed_dir, "GO_" + self.mode)
                if not os.path.exists(self.processed_dir):
                    os.makedirs(self.processed_dir)
        
        # GO Term labels
        self.label_tsv = "/usr/commondata/local_public/protein-datasets/GeneOntology/nrPDB-GO_annot.tsv"

        # Get list of pdb ids and corresponding pdb paths
        self.exlude_pdb_ids = [] # filter out pdb ids with low resolution or high sequence identity
        self.load_predicted_structure()
        # discard_nonstandard_pdb
        for pdb_id in self.pdb_ids:
            chain_id = pdb_id.split("-")[-1]
            if len(chain_id) != 1:
                self.exlude_pdb_ids.append(pdb_id)

        # Filter out pdb ids and corresponding pdb paths at the same time
        if len(self.exlude_pdb_ids) > 0:
            self.filter_pdb(self.exlude_pdb_ids)
        # pdb.set_trace()
        if self.feature_save_mode:
            self.db_conn = None
            self._load_structure_features(reset)
            self._connect_db()
        
        self.load_annontaion(self.label_tsv, self.pdb_ids)

        

    def load_predicted_structure(self):
        # Predicted-Experimental protein Structure pairing
        if self.mode in ["train", "eval"]:
            data_path = os.path.join(self.path, "nrPDB-GO_" + self.mode)
        else:
            data_path = os.path.join(self.path, "nrPDB-GO_test")
        tm_score_tsv = os.path.join(data_path, "tmscore.tsv")
        df = pd.read_csv(tm_score_tsv, sep="\t", header=0)
        if self.tm_cutoff is not None:
            df = df.query(f'tmscore > {self.tm_cutoff}') # find predicted structure with tmscore > cutoff

        self.pdb_ids = df["pdb_chain"].tolist() # pdb_ids haven't been filtered by plddt and test_cutoff
        uniprot_ids = df["uniprot"].tolist() # conresponding uniprot ids for finding the pdb path
        pdb_path = os.path.join(data_path,"af2")
        # pdb_path = os.path.join(data_path,"aligned_pdb")
        self.pdb_files = [os.path.join(pdb_path, uniprot_id + '.pdb') for uniprot_id in uniprot_ids]
        # self.pdb_files = [os.path.join(pdb_path, pdb_id + '.pdb') for pdb_id in self.pdb_ids]


        if self.debug:
            self.pdb_ids = self.pdb_ids[:100]
            self.pdb_files = self.pdb_files[:100]

        if self.plddt_cutoff is not None:
            # read the reference plddt value for each pdb file
            plddt_tsv = os.path.join(self.path, "af2_plddt.tsv")
            df = pd.read_csv(plddt_tsv, sep="\t", header=0)
            # convert the dataframe to a dictionary whose value is real number
            plddt_dict = df.set_index('name').T.to_dict('list')
            # for every value in plddt_dict, convert it to a float number
            plddt_dict = {k: float(v[0]) for k, v in plddt_dict.items()}
            # filter out pdb ids with low plddt value
            self.exlude_pdb_ids = [pdb_id for id, pdb_id in enumerate(self.pdb_ids) if plddt_dict[uniprot_ids[id]] < self.plddt_cutoff]

        if self.mode != "train":
            csv_file = os.path.join(self.path, "nrPDB-GO_test.csv")
            with open(csv_file, "r") as fin:
                reader = csv.reader(fin, delimiter=",")
                idx = self.test_cutoffs.index(self.test_cutoff) + 1
                _ = next(reader)
                for line in reader:
                    if line[idx] == "0": #Note: It's proteins that are not included
                        self.exlude_pdb_ids.append(line[0])

        error_list_path = os.path.join(data_path, "error.txt")
        # read a list from the error.txt file
        with open(error_list_path, "r") as fin:
            error_list = fin.readlines()
        # strip the '\n' in the end of each line
        error_list = [x.strip().upper() for x in error_list]
        # filter out pdb ids with error
        self.exlude_pdb_ids.extend(error_list)
    
    def filter_pdb(self, exclude_pdb_ids):
        # Filter out pdb ids and corresponding pdb paths at the same time
        exclude_pdb_ids = set(exclude_pdb_ids)
        pdb_files = []
        pdb_ids = []
        for pdb_id, pdb_file in zip(self.pdb_ids, self.pdb_files):
            if pdb_id not in exclude_pdb_ids and pdb_id.split("-")[0] not in exclude_pdb_ids:
                pdb_files.append(pdb_file)
                pdb_ids.append(pdb_id)
        self.pdb_files = pdb_files
        self.pdb_ids = pdb_ids

    @property
    def _structure_cache_path(self):
        return os.path.join(self.processed_dir, 'structures.lmdb')

    def _load_structure_features(self, reset):
        if not os.path.exists(self._structure_cache_path) or reset:
            if os.path.exists(self._structure_cache_path):
                os.unlink(self._structure_cache_path)
            self._preprocess_structures()
    
    def _preprocess_structures(self):
        tasks = []
        for pdb_id, pdb_file in zip(self.pdb_ids, self.pdb_files):
            tasks.append(
                {
                    "pdb_id": pdb_id,
                    "pdb_file": pdb_file,
                }
            )

        data_list = joblib.Parallel(
            n_jobs=max(joblib.cpu_count()//2, 1),
        )(
            joblib.delayed(self._process_protein)(task["pdb_file"], task["pdb_id"])
            for task in tqdm(tasks, dynamic_ncols=True ,desc="Preprocessing structures")
        )

        db_conn = lmdb.open(
            self._structure_cache_path,
            map_size = self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )
        ids = []
        with db_conn.begin(write=True, buffers=True) as txn:
            for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
                if data is None:
                    continue
                ids.append(data['pdb_id'])
                txn.put(data["pdb_id"].encode('utf-8'), pickle.dumps(data))
        with open(self._structure_cache_path + '-ids', 'wb') as f:
            pickle.dump(ids, f)
    
    def load_annontaion(self, tsv_file, pdb_ids):
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
    def _process_protein(self, pdb_path: str, pname: Optional[str] = None, chain_id: Optional[str] = None):
        # data = process_decoy(path, gnn_feature, seq, self.config.decoy)
        data = get_pipeline(self.feature_pipeline)(pdb_path, pname, getattr(self.config, "decoy", None))
        return data

    def _connect_db(self):
        if self.db_conn is not None:
            return
        self.db_conn = lmdb.open(
            self._structure_cache_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        # self.feature_list = {} # store all data in cpu for the sake of speed
        # self.data = pickle.loads(self.db_conn)
    
    def _get_structure(self, pname):
        # data = self.feature_list.get(pname, None)
        # if data is None:
        with self.db_conn.begin() as txn:
            data = pickle.loads(txn.get(pname.encode()))
            # self.feature_list[pname] = data
        return data


    def __getitem__(self, index):
        if torch.is_tensor(index): index = index.tolist()
        # Get target name
        pname = self.pdb_ids[index]
        # Get decoy path
        pdb_file = self.pdb_files[index]

        if self.feature_save_mode:
            feats = self._get_structure(pname)
        else:
            feats = self._process_protein(pdb_file, pname)
        # ESM feature
        # feats['esm_emb'] = Utils.get_esm_embedding(seq, pname, self.esm_save_dir)
        # Prepare groundtruth
        indices = self.pos_targets[index].unsqueeze(0)
        values = torch.ones(len(self.pos_targets[index]))
        # pdb.set_trace()
        feats["targets"] = utils.sparse_coo_tensor(indices, values, (len(self.tasks),)).to_dense()
        
        
        return feats

if __name__ == '__main__':
    import argparse
    from config._base import get_config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="Anontation_Test")
    parser.add_argument("--mode", type=str, default="eval")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--reset", type=bool, default=False)
    args = parser.parse_args()
    if args.reset:
        sure = input('Sure to reset? (y/n): ')
        if sure != 'y':
            exit()
    config = get_config(args.config_name)()
    # pdb.set_trace()
    data_config = config.data
    dataset = GOPredData(data_config, args.mode, debug=args.debug, reset=args.reset)
    data = dataset[0]
    # pdb.set_trace()
    # print(data["edge_type"].shape)
    # print(data["dist"].shape)
    print("pdb_id: ", data["pdb_id"])
    print("length: ", len(data["decoy_seq_mask"]))
    print(data.keys())
    print(len(dataset))
        