import csv
import glob
import logging
import os
import sys
sys.path.append("/root/Generative-Models/PGSL-RA")
# from functools import partial
import pdb
from typing import Dict, Mapping, Optional

import ml_collections as mlc
import numpy as np
# import pytorch_lightning as pl
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
    MAP_SIZE = 512*(1024*1024*1024)  # 512GB
    def __init__(self,
                config: mlc.ConfigDict,
                mode: str="train",
                debug: bool=False,
                reset: bool=False,
                **kwargs):
        super().__init__()
        # self.esm_save_dir = config.dataset.esm_save_dir
        self.feature_pipeline = config.dataset.feature_pipeline
        self.mode = mode
        self.config = config
        self.debug = debug

        root_dir = config.dataset.root_dir
        root_dir = os.path.expanduser(root_dir)
        self.path = root_dir

        test_cutoff = config.dataset.test_cutoff
        if test_cutoff not in self.test_cutoffs:
            raise ValueError("Unknown test cutoff `%.2f` for GeneOntology dataset" % test_cutoff)
        self.test_cutoff = test_cutoff

        branch = config.dataset.branch
        if branch not in self.branches:
            raise ValueError("Unknown branch `%s` for GeneOntology dataset" % branch)
        self.branch = branch

        self.processed_dir = os.path.join(config.dataset.processed_dir, self.mode)
        self.feature_saved_mode = False
        if self.processed_dir is not None:
            self.feature_saved_mode = True
        if self.feature_saved_mode and not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        
        self.exlude_pdb_ids = []
        self.tsv_file = os.path.join(self.path, "nrPDB-GO_annot.tsv") # MulticlassBinaryClassification label
        self.load_data_entries()

        if len(self.exlude_pdb_ids) > 0:
            self.filter_pdb(self.exlude_pdb_ids)# filter out similar proteins in test set
        self.dicard_nonstandard_pdb()
        
        if self.feature_saved_mode:
            self.db_conn = None
            self._load_structures(reset)
            self._connect_db()

        pdb_ids = [os.path.basename(pdb_file).split("_")[0] for pdb_file in self.pdb_files]
        self.pdb_ids = pdb_ids
        self.load_annotation(self.tsv_file, pdb_ids)

    def load_data_entries(self):
        if self.mode == "predict":
            csv_file = os.path.join(self.path, "nrPDB-GO_test.csv")
            with open(csv_file, "r") as fin:
                reader = csv.reader(fin, delimiter=",")
                idx = self.test_cutoffs.index(self.test_cutoff) + 1
                _ = next(reader)
                for line in reader:
                    if line[idx] == "0": #Note: It's proteins that are not included
                        self.exlude_pdb_ids.append(line[0])
            path = os.path.join(self.path, "test")
        elif self.mode == "train":
            path = os.path.join(self.path, "train")
        else: #TODO: add validation, currently use test as validation
            csv_file = os.path.join(self.path, "nrPDB-GO_test.csv")
            with open(csv_file, "r") as fin:
                reader = csv.reader(fin, delimiter=",")
                idx = self.test_cutoffs.index(self.test_cutoff) + 1
                _ = next(reader)
                for line in reader:
                    if line[idx] == "0": #Note: It's proteins that are not included
                        self.exlude_pdb_ids.append(line[0])
            path = os.path.join(self.path, "test")
        self.pdb_files = glob.glob(os.path.join(path, "*.pdb")) #pdb file list in train/valid/test set
        if self.debug:
            self.pdb_files = self.pdb_files[:100]
    
    @property
    def _structure_cache_path(self):
        return os.path.join(self.processed_dir, 'structures.lmdb')
    
    def _load_structures(self, reset):
        if not os.path.exists(self._structure_cache_path) or reset:
            if os.path.exists(self._structure_cache_path):
                os.unlink(self._structure_cache_path)
            self._preprocess_structures()
        
    def _preprocess_structures(self):
        tasks = []
        for entry in self.pdb_files:
            if not os.path.exists(entry):
                logging.warning("PDB file `%s` does not exist" % entry)
                continue
            tasks.append(
                {
                    "pdb_path": entry,
                }
            )
        data_list = joblib.Parallel(
            n_jobs=max(joblib.cpu_count()//2, 1),
        )(
            joblib.delayed(self._process_protein)(task["pdb_path"])
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

        if self.feature_saved_mode:
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
    parser.add_argument("--config_name", type=str, default="GOMF_Graphformer")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--reset", type=bool, default=True)
    args = parser.parse_args()
    if args.reset:
        sure = input('Sure to reset? (y/n): ')
        if sure != 'y':
            exit()
    config = get_config(args.config_name)()
    # pdb.set_trace()
    data_config = config.data
    dataset = Data(data_config, args.mode, debug=args.debug, reset=args.reset)
    data = dataset[0]
    # pdb.set_trace()
    # print(data["edge_type"].shape)
    # print(data["dist"].shape)
    print(data.keys())
    print(len(dataset))

