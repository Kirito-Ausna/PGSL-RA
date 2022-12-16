import copy
from functools import partial
import json
import logging
# logging.basicConfig(level=logging.NOTSET)
import os
import pickle
from typing import Optional, Sequence, Any, List

import ml_collections as mlc
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import RandomSampler, random_split

from openfold.data import (
    data_pipeline,
    feature_pipeline,
    mmcif_parsing,
    templates,
)
from openfold.utils.tensor_utils import tensor_tree_map, dict_multimap
import pdb



class OpenFoldSingleDataset(torch.utils.data.Dataset):
    def __init__(self,
        SeqEmb_dir: str,
        data_dir: str,
        alignment_dir: str, 
        template_mmcif_dir: str,
        max_template_date: str,
        config: mlc.ConfigDict,
        kalign_binary_path: str = '/usr/bin/kalign',
        mapping_path: Optional[str] = None,
        max_template_hits: int = 4,
        template_release_dates_cache_path: Optional[str] = None,
        shuffle_top_k_prefiltered: Optional[int] = None,
        treat_pdb_as_distillation: bool = True,
        mode: str = "train", 
        _output_raw: bool = False,
        _alignment_index: Optional[Any] = None
    ):
        """
            Args:
                data_dir:
                    A path to a directory containing mmCIF files (in train
                    mode) or FASTA files (in inference mode).
                alignment_dir:
                    A path to a directory containing only data in the format 
                    output by an AlignmentRunner 
                    (defined in openfold.features.alignment_runner).
                    I.e. a directory of directories named {PDB_ID}_{CHAIN_ID}
                    or simply {PDB_ID}, each containing .a3m, .sto, and .hhr
                    files.
                template_mmcif_dir:
                    Path to a directory containing template mmCIF files.
                config:
                    A dataset config object. See openfold.config
                kalign_binary_path:
                    Path to kalign binary.
                mapping_path:
                    A json file containing a mapping from consecutive numerical
                    ids to sample names (matching the directories in data_dir).
                    Samples not in this mapping are ignored. Can be used to 
                    implement the various training-time filters described in
                    the AlphaFold supplement.
                max_template_hits:
                    An upper bound on how many templates are considered. During
                    training, the templates ultimately used are subsampled
                    from this total quantity.
                template_release_dates_cache_path:
                    Path to the output of scripts/generate_mmcif_cache.
                shuffle_top_k_prefiltered:
                    Whether to uniformly shuffle the top k template hits before
                    parsing max_template_hits of them. Can be used to
                    approximate DeepMind's training-time template subsampling
                    scheme much more performantly.
                treat_pdb_as_distillation:
                    Whether to assume that .pdb files in the data_dir are from
                    the self-distillation set (and should be subjected to
                    special distillation set preprocessing steps).
                mode:
                    "train", "val", or "predict"
        """
        super(OpenFoldSingleDataset, self).__init__()
        self.data_dir = data_dir
        self.alignment_dir = alignment_dir
        self.config = config
        self.treat_pdb_as_distillation = treat_pdb_as_distillation
        self.mode = mode
        self._output_raw = _output_raw
        self._alignment_index = _alignment_index

        valid_modes = ["train", "eval", "predict"]
        if(mode not in valid_modes):
            raise ValueError(f'mode must be one of {valid_modes}')

        if(mapping_path is None):
            self.mapping = {
                str(i):os.path.splitext(name)[0] 
                for i, name in enumerate(os.listdir(alignment_dir))
            }
        else:
            with open(mapping_path, 'r') as fp:
                self.mapping = json.load(fp)

        if(template_release_dates_cache_path is None):
            logging.warning(
                "Template release dates cache does not exist. Remember to run "
                "scripts/generate_mmcif_cache.py before running OpenFold"
            )

        if(_alignment_index is not None):
            self._chain_ids = list(_alignment_index.keys())
        elif(mapping_path is None):
            self._chain_ids = list(os.listdir(alignment_dir))
        else:
            with open(mapping_path, "r") as f:
                self._chain_ids = [l.strip() for l in f.readlines()]
        
        self._chain_id_to_idx_dict = {
            chain: i for i, chain in enumerate(self._chain_ids)
        }

        template_featurizer = templates.TemplateHitFeaturizer(
            mmcif_dir=template_mmcif_dir,
            max_template_date=max_template_date,
            max_hits=max_template_hits,
            kalign_binary_path=kalign_binary_path,
            release_dates_path=template_release_dates_cache_path,
            obsolete_pdbs_path=None,
            _shuffle_top_k_prefiltered=shuffle_top_k_prefiltered,
        )

        self.data_pipeline = data_pipeline.DataPipeline(
            # SeqEmb_dir=SeqEmb_dir,
            template_featurizer=template_featurizer,
        )

        if(not self._output_raw):
            self.feature_pipeline = feature_pipeline.FeaturePipeline(config) 

    def _parse_mmcif(self, path, file_id, chain_id, alignment_dir, _alignment_index):
        with open(path, 'r') as f:
            mmcif_string = f.read()

        mmcif_object = mmcif_parsing.parse(
            file_id=file_id, mmcif_string=mmcif_string
        )

        # Crash if an error is encountered. Any parsing errors should have
        # been dealt with at the alignment stage.
        if(mmcif_object.mmcif_object is None):
            raise list(mmcif_object.errors.values())[0]

        mmcif_object = mmcif_object.mmcif_object

        data = self.data_pipeline.process_mmcif(
            mmcif=mmcif_object,
            alignment_dir=alignment_dir,
            chain_id=chain_id,
            _alignment_index=_alignment_index
        )

        return data
    
    def __getitem__(self, idx):
        name = self.mapping[str(idx)]
        alignment_dir = os.path.join(self.alignment_dir, name)
        # logging.warning(f"current sample is {name}, please debug")
        _alignment_index = None
        if(self._alignment_index is not None):
            alignment_dir = self.alignment_dir
            _alignment_index = self._alignment_index[name]

        if(self.mode == 'train' or self.mode == 'eval'):
            spl = name.rsplit('_', 1)
            if(len(spl) == 2):
                file_id, chain_id = spl
            else:
                file_id, = spl
                chain_id = None

            path = os.path.join(self.data_dir, file_id)
            if(os.path.exists(path + ".cif")):
                data = self._parse_mmcif(
                    path + ".cif", file_id, chain_id, alignment_dir, _alignment_index,
                )
            elif(os.path.exists(path + ".core")):
                data = self.data_pipeline.process_core(
                    path + ".core", alignment_dir, _alignment_index,
                )
            else:
                path = os.path.join(self.data_dir, name)
                data = self.data_pipeline.process_pdb(
                    pdb_path=path + ".pdb",
                    alignment_dir=alignment_dir,
                    is_distillation=self.treat_pdb_as_distillation,
                    chain_id=chain_id,
                    _alignment_index=_alignment_index,
                )
        else:
            path = os.path.join(name, name + ".fasta")
            data = self.data_pipeline.process_fasta(
                fasta_path=path,
                alignment_dir=alignment_dir,
                _alignment_index=_alignment_index,
            )
            
        if(self._output_raw):
            return data

        feats = self.feature_pipeline.process_features(
            data, self.mode
        )

        return feats

    def __len__(self):
        return len(self._chain_ids) 


def deterministic_train_filter(
    chain_data_cache_entry: Any,
    max_resolution: float = 9.,
    max_single_aa_prop: float = 0.8,
) -> bool:
    # Hard filters
    resolution = chain_data_cache_entry.get("resolution", None)
    if(resolution is not None and resolution > max_resolution):
        return False

    seq = chain_data_cache_entry["seq"]
    counts = {}
    for aa in seq:
        counts.setdefault(aa, 0)
        counts[aa] += 1
    largest_aa_count = max(counts.values())
    largest_single_aa_prop = largest_aa_count / len(seq)
    if(largest_single_aa_prop > max_single_aa_prop):
        return False

    return True


def get_stochastic_train_filter_prob(
    chain_data_cache_entry: Any,
) -> List[float]:
    # Stochastic filters
    probabilities = []
    
    cluster_size = chain_data_cache_entry.get("cluster_size", None)
    if(cluster_size is not None and cluster_size > 0):
        probabilities.append(1 / cluster_size)
    
    chain_length = len(chain_data_cache_entry["seq"])
    probabilities.append((1 / 512) * (max(min(chain_length, 512), 256)))
    
     # Risk of underflow here?
    out = 1
    for p in probabilities:
        out *= p
        
    
    return out

def looped_sequence(sequence):
    while True:
        for x in sequence:
            yield x

class OpenFoldDataset(torch.utils.data.IterableDataset):
    """
        The Dataset is written to accommodate the requirement that proteins are
        sampled from the distillation set with some probability p
        and from the PDB set with probability (1 - p). Proteins are sampled
        from both sets without replacement, and as soon as either set is
        emptied, it is refilled. The Dataset therefore has an arbitrary length.
        Nevertheless, for compatibility with various PyTorch Lightning
        functionalities, it is possible to specify an epoch length. This length
        has no effect on the output of the Dataset.
    """
    def __init__(self,
        datasets: Sequence[OpenFoldSingleDataset],
        probabilities: Sequence[int],
        epoch_len: int,
        chain_data_cache_paths: List[str],
        generator: torch.Generator = None,
        _roll_at_init: bool = True,
    ):
        self.datasets = datasets
        self.samplers = [
            looped_sequence(RandomSampler(d)) for d in datasets
        ]
        self.epoch_len = epoch_len
        self.generator = generator
        
        # self.chain_data_caches = []
        # for path in chain_data_cache_paths:
        #      with open(path, "r") as fp:
        #          self.chain_data_caches.append(json.load(fp))

        def looped_shuffled_dataset_idx(dataset_len):
            while True:
                # Uniformly shuffle each dataset's indices
                weights = [1. for _ in range(dataset_len)]
                shuf = torch.multinomial(
                    torch.tensor(weights),
                    num_samples=dataset_len,
                    replacement=False,
                    generator=self.generator,
                )
                for idx in shuf:
                    yield idx

        def looped_samples(dataset_idx):
            max_cache_len = int(epoch_len * probabilities[dataset_idx])
            dataset = self.datasets[dataset_idx]
            idx_iter = looped_shuffled_dataset_idx(len(dataset))
            chain_data_cache = self.chain_data_caches[dataset_idx]
            while True:
                weights = []
                idx = []
                for _ in range(max_cache_len):
                    candidate_idx = next(idx_iter)
                    chain_id = dataset.idx_to_chain_id(candidate_idx)
                    chain_data_cache_entry = chain_data_cache[chain_id]
                    if(not deterministic_train_filter(chain_data_cache_entry)):
                        continue

                    p = get_stochastic_train_filter_prob(
                        chain_data_cache_entry,
                    )
                    weights.append([1. - p, p])
                    idx.append(candidate_idx)

                samples = torch.multinomial(
                    torch.tensor(weights),
                    num_samples=1,
                    generator=self.generator,
                )
                samples = samples.squeeze()

                cache = [i for i, s in zip(idx, samples) if s]

                for datapoint_idx in cache:
                    yield datapoint_idx

        self.distr = torch.distributions.categorical.Categorical(
            probs=torch.tensor(probabilities),
        )

    def __iter__(self):
        return self

    def __next__(self):
        dataset_idx = self.distr.sample()
        sampler = self.samplers[dataset_idx]
        element_idx = next(sampler)
        return self.datasets[dataset_idx][element_idx] 

    def __len__(self):
        return self.epoch_len


class OpenFoldBatchCollator:
    def __init__(self, config, generator, stage="train"):
        self.config = config
        self.generator = generator
        self.stage = stage
        self.feature_pipeline = feature_pipeline.FeaturePipeline(config)
        self._prep_batch_properties_probs()
        
    def _prep_batch_properties_probs(self):
        keyed_probs = []
        stage_cfg = self.config[self.stage]

        max_iters = self.config.common.max_recycling_iters
        if(stage_cfg.supervised):
            clamp_prob = self.config.supervised.clamp_prob
            keyed_probs.append(
                ("use_clamped_fape", [1 - clamp_prob, clamp_prob])
            ) 
            
        if(stage_cfg.uniform_recycling):
            recycling_probs = [
                1. / (max_iters + 1) for _ in range(max_iters + 1)
            ]
        else:
            recycling_probs = [
                0. for _ in range(max_iters + 1)
            ]
            recycling_probs[-1] = 1.
        
        keyed_probs.append(
            ("no_recycling_iters", recycling_probs)
        )

        keys, probs = zip(*keyed_probs)
        max_len = max([len(p) for p in probs])
        padding = [[0.] * (max_len - len(p)) for p in probs] 
        
        self.prop_keys = keys
        self.prop_probs_tensor = torch.tensor(
            [p + pad for p, pad in zip(probs, padding)],
            dtype=torch.float32,
        )

    def _add_batch_properties(self, raw_prots):
        samples = torch.multinomial(
            self.prop_probs_tensor,
            num_samples=1, # 1 per row
            replacement=True,
            generator=self.generator
        )

        for i, key in enumerate(self.prop_keys):
            sample = samples[i][0]
            for prot in raw_prots:
                prot[key] = np.array(sample, dtype=np.float32)

    def __call__(self, raw_prots):
        # pdb.set_trace()
        self._add_batch_properties(raw_prots)
        processed_prots = []
        # pdb.set_trace()
        for prot in raw_prots:
            features = self.feature_pipeline.process_features(
                prot, self.stage
            )
            processed_prots.append(features)

        stack_fn = partial(torch.stack, dim=0)
        return dict_multimap(stack_fn, processed_prots) 


class OpenFoldDataModule(pl.LightningDataModule):
    def __init__(self,
        config: mlc.ConfigDict,
        SeqEmb_dir: str,
        template_mmcif_dir: str,
        max_template_date: str,
        train_data_dir: Optional[str] = None,
        train_alignment_dir: Optional[str] = None,
        train_chain_data_cache_path: Optional[str] = None,
        distillation_data_dir: Optional[str] = None,
        distillation_alignment_dir: Optional[str] = None,
        distillation_chain_data_cache_path: Optional[str] = None,
        val_data_dir: Optional[str] = None,
        val_alignment_dir: Optional[str] = None,
        predict_data_dir: Optional[str] = None,
        predict_alignment_dir: Optional[str] = None,
        kalign_binary_path: str = '/usr/bin/kalign',
        train_mapping_path: Optional[str] = None,
        distillation_mapping_path: Optional[str] = None,
        obsolete_pdbs_file_path: Optional[str] = None,
        template_release_dates_cache_path: Optional[str] = None,
        batch_seed: Optional[int] = None,
        train_epoch_len: int = 50000, 
        _alignment_index_path: Optional[str] = None,
        **kwargs
    ):
        super(OpenFoldDataModule, self).__init__()

        self.config = config
        self.SeqEmb_dir = SeqEmb_dir
        self.template_mmcif_dir = template_mmcif_dir
        self.max_template_date = max_template_date
        self.train_data_dir = train_data_dir
        self.train_alignment_dir = train_alignment_dir
        self.train_chain_data_cache_path = train_chain_data_cache_path
        self.distillation_data_dir = distillation_data_dir
        self.distillation_alignment_dir = distillation_alignment_dir
        self.distillation_chain_data_cache_path = (
            distillation_chain_data_cache_path
        )
        self.val_data_dir = val_data_dir
        self.val_alignment_dir = val_alignment_dir
        self.predict_data_dir = predict_data_dir
        self.predict_alignment_dir = predict_alignment_dir
        self.kalign_binary_path = kalign_binary_path
        self.train_mapping_path = train_mapping_path
        self.distillation_mapping_path = distillation_mapping_path
        self.template_release_dates_cache_path = (
            template_release_dates_cache_path
        )
        self.obsolete_pdbs_file_path = obsolete_pdbs_file_path
        self.batch_seed = batch_seed
        self.train_epoch_len = train_epoch_len

        if(self.train_data_dir is None and self.predict_data_dir is None):
            raise ValueError(
                'At least one of train_data_dir or predict_data_dir must be '
                'specified'
            )

        self.training_mode = self.train_data_dir is not None

        if(self.training_mode and self.train_alignment_dir is None):
            raise ValueError(
                'In training mode, train_alignment_dir must be specified'
            )
        elif(not self.training_mode and predict_alignment_dir is None):
            raise ValueError(
                'In inference mode, predict_alignment_dir must be specified'
            )      
        elif(val_data_dir is not None and val_alignment_dir is None):
            raise ValueError(
                'If val_data_dir is specified, val_alignment_dir must '
                'be specified as well'
        )

        # An ad-hoc measure for our particular filesystem restrictions
        self._alignment_index = None
        if(_alignment_index_path is not None):
            with open(_alignment_index_path, "r") as fp:
                self._alignment_index = json.load(fp)

    def setup(self):
        # Most of the arguments are the same for the three datasets 
        dataset_gen = partial(OpenFoldSingleDataset,
            SeqEmb_dir = self.SeqEmb_dir,
            template_mmcif_dir=self.template_mmcif_dir,
            max_template_date=self.max_template_date,
            config=self.config,
            kalign_binary_path=self.kalign_binary_path,
            template_release_dates_cache_path=
                self.template_release_dates_cache_path,
        )

        if(self.training_mode):        
            train_dataset = dataset_gen(
                data_dir=self.train_data_dir,
                alignment_dir=self.train_alignment_dir,
                mapping_path=self.train_mapping_path,
                max_template_hits=self.config.train.max_template_hits,
                shuffle_top_k_prefiltered=
                    self.config.train.shuffle_top_k_prefiltered,
                treat_pdb_as_distillation=False,
                mode="train",
                _output_raw=True,
                _alignment_index=self._alignment_index,
            )
            
            distillation_dataset = None
            if(self.distillation_data_dir is not None):
                distillation_dataset = dataset_gen(
                    data_dir=self.distillation_data_dir,
                    alignment_dir=self.distillation_alignment_dir,
                    mapping_path=self.distillation_mapping_path,
                    max_template_hits=self.config.train.max_template_hits,
                    treat_pdb_as_distillation=True,
                    mode="train",
                    _output_raw=True,
                )

                d_prob = self.config.train.distillation_prob
           
            if(self.val_data_dir is None):
               train_size = int(0.99975 * len(train_dataset))
               test_size = len(train_dataset) - train_size
               logging.info(f"Split the train dataset as train/vadition:{train_size}/{test_size}")
               train_dataset, val_dataset = random_split(train_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
               logging.info(f"train_len: {len(train_dataset)}, val_len: {len(val_dataset)}")

            if(distillation_dataset is not None):
                datasets = [train_dataset, distillation_dataset]
                d_prob = self.config.train.distillation_prob
                probabilities = [1 - d_prob, d_prob]
                chain_data_cache_paths = [
                    self.train_chain_data_cache_path,
                    self.distillation_chain_data_cache_path,
                ]
            else:
                logging.info("There is no distillation_dataset")
                datasets = [train_dataset]
                probabilities = [1.]   
                chain_data_cache_paths = [
                    self.train_chain_data_cache_path,
                ]
            
                

            self.train_dataset = OpenFoldDataset(
                datasets=datasets,
                probabilities=probabilities,
                epoch_len=self.train_epoch_len,
                chain_data_cache_paths=chain_data_cache_paths,
                _roll_at_init=False,
            )
    
            if(self.val_data_dir is not None):
                self.val_dataset = dataset_gen(
                    data_dir=self.val_data_dir,
                    alignment_dir=self.val_alignment_dir,
                    mapping_path=None,
                    max_template_hits=self.config.eval.max_template_hits,
                    mode="eval",
                )
            else:
                self.val_dataset = val_dataset
        else:           
            self.predict_dataset = dataset_gen(
                data_dir=self.predict_data_dir,
                alignment_dir=self.predict_alignment_dir,
                mapping_path=None,
                max_template_hits=self.config.predict.max_template_hits,
                mode="predict",
            )

    def _gen_batch_collator(self, stage):
        """ We want each process to use the same batch collation seed """
        generator = torch.Generator()
        if(self.batch_seed is not None):
            generator = generator.manual_seed(self.batch_seed)
        collate_fn = OpenFoldBatchCollator(
            self.config, generator, stage
        )
        return collate_fn

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.data_module.data_loaders.batch_size,
            num_workers=self.config.data_module.data_loaders.num_workers,
            collate_fn=self._gen_batch_collator("train"),
        )

    def val_dataloader(self):
        if(self.val_dataset is not None):
            return torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.config.data_module.data_loaders.batch_size,
                num_workers=self.config.data_module.data_loaders.num_workers,
                collate_fn=self._gen_batch_collator("train")
                # A quick fix for split train dataset into train and eval
            )

        return None

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.config.data_module.data_loaders.batch_size,
            num_workers=self.config.data_module.data_loaders.num_workers,
            collate_fn=self._gen_batch_collator("predict")
        )


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, batch_path):
        with open(batch_path, "rb") as f:
            self.batch = pickle.load(f)

    def __getitem__(self, idx):
        return copy.deepcopy(self.batch)

    def __len__(self):
        return 1000


class DummyDataLoader(pl.LightningDataModule):
    def __init__(self, batch_path):
        super().__init__()
        self.dataset = DummyDataset(batch_path)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset)
