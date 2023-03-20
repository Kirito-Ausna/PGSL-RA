import copy

import ml_collections as mlc

from config._base import register_config

def set_inf(c, inf):
    for k, v in c.items():
        if isinstance(v, mlc.ConfigDict):
            set_inf(v, inf)
        elif k == "inf":
            c[k] = inf

@register_config("GOMF_Graphformer")
def model_cofig(train=False, low=False):
    c = copy.deepcopy(config)
    return c

encoder_embed_dim = mlc.FieldReference(default=512, field_type=int) # description="Encoder embedding dimension.",
encoder_ffn_embed_dim = mlc.FieldReference(default=2048, field_type=int) # description="Encoder feedforward embedding dimension.",
pair_embed_dim = mlc.FieldReference(default=128, field_type=int) # description="Pair embedding dimension.",
num_attention_heads = mlc.FieldReference(default=64, field_type=int) # description="Number of attention heads.",
activation_fn = mlc.FieldReference(default="gelu", field_type=str) # description="Activation function.",
max_seq_len = mlc.FieldReference(default=512, field_type=int) # description="Maximum sequence length.",
eps = mlc.FieldReference(default=1e-5, field_type=float) # description="Epsilon for numerical stability.",

NUM_RES = "num residues placeholder"

config = mlc.ConfigDict(
    {
        "globals":{
            "encoder_embed_dim": encoder_embed_dim,
            "encoder_ffn_embed_dim": encoder_ffn_embed_dim,
            "pretrain": False,
            "metric": "f1_max",
        },
        "data":{
            "dataset": {
                "name": "GO",
                "root_dir": "/usr/commondata/local_public/protein-datasets/GeneOntology/",
                "branch": "MF",
                "test_cutoff": 0.95,
                "training_mode": True,
                "eval": True,
                "feature_pipeline": "Graphformer",
                "processed_dir": "/usr/commondata/local_public/protein-datasets/GeneOntology/reprocessed/",
                "esm_save_dir": None,
            },
            "common":{
                "feat":{
                    "decoy_aatype": [NUM_RES],
                    "decoy_all_atom_mask": [NUM_RES, None],
                    "decoy_all_atom_positions": [NUM_RES, None, None],
                    # build in dataloader
                    "decoy_seq_mask": [NUM_RES],
                    # Build in gaussian encoder
                    "decoy_angle_feats": [NUM_RES, None], 
                    "dist": [NUM_RES, NUM_RES],
                    "edge_type": [NUM_RES, NUM_RES],
                    "targets": [None], #:= 538
                }
            },
            "predict": {
                "fixed_size": True,
                "crop": False,
                "crop_size": None,
                "supervised": False,
                "uniform_recycling": False,
            },
            "eval": {
                "fixed_size": True,
                "crop": False,
                "crop_size": None,
                "supervised": True,
                "uniform_recycling": False,
            },
            "train": {
                "fixed_size": True,
                "crop": True,
                "crop_size": max_seq_len,
                "supervised": True,
                "clamp_prob": 0.9,
                "uniform_recycling": False,
            },
            "data_module":{
                "train_dataloader":{
                    "batch_size": 4,
                    "num_workers": 32,
                },
                "val_dataloader":{
                    "batch_size": 1,
                    "num_workers": 32,
                },
                "predict_dataloader":{
                    "batch_size": 1,
                    "num_workers": 16,
                },
            },
        },
        "downstream":{
            "encoder": "uni_encoder",
            "encoder_checkpoint": None,
            "head": {
                "model_out_dim": encoder_embed_dim,
                "task_num": 489, #EC: 538, GO-CC: 320, GO-MF: 489, GO-BP: 1943
                "num_mlp_layers": 3,
            },
            "metric": ['f1_max'],
            "encoder_fixed": False,
            "reweight": False,
        },
        "model": {
            "embedder": {
                "protein_angle_embedder": {
                    "c_in": 57,
                    "c_m": encoder_embed_dim // 2,
                    "c_out": encoder_embed_dim,
                },
                "gaussian_layer": {
                    "kernel_num": pair_embed_dim,
                },
                "non_linear_head": {
                    "input_dim": pair_embed_dim,
                    "out_dim": num_attention_heads,
                    "activation_fn": activation_fn,
                    "hidden": 2*num_attention_heads,
                }   
            },
            "graphformer": {
                "encoder_layers": 15, # original 15
                "embed_dim": encoder_embed_dim,
                "ffn_embed_dim": encoder_ffn_embed_dim,
                "attention_heads": num_attention_heads,
                "emb_dropout": 0.1,
                "dropout": 0.1,
                "attention_dropout": 0.1,
                "activation_dropout": 0.1,
                "max_seq_len": max_seq_len,
                "activation_fn": activation_fn,
                # "pooler_activation_fn": "tanh",
                "post_ln": False,
                "no_final_head_layer_norm": True,
            }
        },
        "loss": {
    
        },
        "train":{
            "base_lr": 0.,
            "max_lr": 1e-4,
            "warmup_no_steps": 34240,
            "start_decay_after_n_steps": 171300,
            "decay_every_n_steps": 3440
        }

    }
)