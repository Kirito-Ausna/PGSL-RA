import copy

import ml_collections as mlc

from config._base import register_config

def set_inf(c, inf):
    for k, v in c.items():
        if isinstance(v, mlc.ConfigDict):
            set_inf(v, inf)
        elif k == "inf":
            c[k] = inf

@register_config("EC_UniMol")
def model_cofig(train=False, low=False):
    c = copy.deepcopy(config)
    return c

encoder_embed_dim = mlc.FieldReference(default=512, field_type=int) # description="Encoder embedding dimension.",
encoder_ffn_embed_dim = mlc.FieldReference(default=2048, field_type=int) # description="Encoder feedforward embedding dimension.",
pair_embed_dim = mlc.FieldReference(default=400, field_type=int) # description="Pair embedding dimension.",
num_attention_heads = mlc.FieldReference(default=64, field_type=int) # description="Number of attention heads.",
activation_fn = mlc.FieldReference(default="gelu", field_type=str) # description="Activation function.",
max_seq_len = mlc.FieldReference(default=256, field_type=int) # description="Maximum sequence length.",
eps = mlc.FieldReference(default=1e-12, field_type=float) # description="Epsilon for numerical stability.",

NUM_RES = "num residues placeholder"

config = mlc.ConfigDict(
    {
        "globals":{
            "encoder_embed_dim": encoder_embed_dim,
            "encoder_ffn_embed_dim": encoder_ffn_embed_dim,
            "pretrain": False,
            "metric": "f1_max",
            "max_epochs": 75,
        },
        "data":{
            "dataset": {
                "name": "EC",
                "root_dir": "/usr/commondata/local_public/protein-datasets/EnzymeCommission/",
                "test_cutoff": 0.95,
                "training_mode": True,
                "eval": True,
                "feature_pipeline": "Graphformer",
                "processed_dir": "/usr/commondata/local_public/protein-datasets/EnzymeCommission/processed/",
                "esm_save_dir": None,
            },
            "common":{
                "feat":{
                    "decoy_aatype": [NUM_RES],
                    "decoy_all_atom_mask": [NUM_RES, None],
                    "decoy_all_atom_positions": [NUM_RES, None, None],
                    "decoy_angle_feats": [NUM_RES, None], 
                    "targets": [None], #:= 538
                    "bb_rigid_tensors": [NUM_RES, None, None],
                    # build in dataloader
                    "decoy_seq_mask": [NUM_RES],
                    # Build in gaussian encoder
                    "dist": [NUM_RES, NUM_RES, None],
                    "edge_type": [NUM_RES, NUM_RES],
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
                "crop": True,
                "crop_size": 512,
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
                    "batch_size": 16,
                    "num_workers": 32,
                },
                "val_dataloader":{
                    "batch_size": 16,
                    "num_workers": 32,
                },
                "predict_dataloader":{
                    "batch_size": 1,
                    "num_workers": 16,
                },
            },
        },


        "downstream":{
            "encoder": "uni-mol",
            "encoder_checkpoint": None,
            "head": {
                "model_out_dim": encoder_embed_dim,
                "task_num": 538, #EC: 538, GO-CC: 320, GO-MF: 489, GO-BP: 1943
                "num_mlp_layers": 3,
            },
            "metric": ['f1_max','auprc_micro'],
            "encoder_fixed": False,
            "reweight": False,
        },


        "model": {
            "embedder": {
                "protein_angle_embedder": {
                    "c_in": 57,
                    "c_m": encoder_embed_dim*2,
                    "c_out": encoder_embed_dim,
                },
                "gaussian_layer": {
                    "kernel_num": 16,
                    "num_pair_distance": 25
                },
                "bias_proj_layer": {
                    "input_dim": pair_embed_dim,# 25*16 = 400
                    "out_dim": num_attention_heads,
                    "activation_fn": activation_fn,
                    "hidden": 2*num_attention_heads,
                },
                "centrality_proj_layer":{
                    "input_dim": pair_embed_dim,
                    "out_dim": encoder_embed_dim,
                    "activation_fn": activation_fn,
                    "hidden": 2*encoder_embed_dim,
                }   
            },
            "graphformer": {
                "encoder_layers": 6, # original 15
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
            },
        },
        "loss": {
    
        },
        "train":{
            "base_lr": 0.,
            "max_lr": 1e-4,
            "warmup_no_steps": 2650, # 5 epochs
            "start_decay_after_n_steps": 26500, # 50 epochs
            "decay_every_n_steps": 265 # 1 epoch
        }

    }
)