import copy

import ml_collections as mlc

from config._base import register_config


def set_inf(c, inf):
    for k, v in c.items():
        if isinstance(v, mlc.ConfigDict):
            set_inf(v, inf)
        elif k == "inf":
            c[k] = inf

@register_config("GOMF_GraphformerPlus")
def model_cofig(train=False, low=False):
    c = copy.deepcopy(config)
    return c

c_z = mlc.FieldReference(128, field_type=int)
c_m = mlc.FieldReference(256, field_type=int)
c_t = mlc.FieldReference(64, field_type=int)
c_e = mlc.FieldReference(64, field_type=int)
c_s = mlc.FieldReference(384, field_type=int)
blocks_per_ckpt = mlc.FieldReference(1, field_type=int)
chunk_size = mlc.FieldReference(None, field_type=int)
aux_distogram_bins = mlc.FieldReference(64, field_type=int)
tm_enabled = mlc.FieldReference(False, field_type=bool)
eps = mlc.FieldReference(1e-8, field_type=float)
activation_fn = mlc.FieldReference(default="gelu", field_type=str) # description="Activation function.",
NUM_RES = "num residues placeholder"

config = mlc.ConfigDict(
    {
        "globals": {
            "blocks_per_ckpt": blocks_per_ckpt,
            "chunk_size": chunk_size,
            "c_z": c_z,
            "c_m": c_m,
            "c_t": c_t,
            "c_e": c_e,
            "c_s": c_s,
            "eps": eps,
            "pretrain": False,
            "metric": "f1_max",
        },
        "data":{
            "dataset":{
                "name": "GO",
                "root_dir": "/usr/commondata/local_public/protein-datasets/GeneOntology/",
                "branch": "MF",
                "test_cutoff": 0.95,
                "training_mode": True,
                "eval": True,
                "feature_pipeline": "Graphformer",
                "processed_dir": "/usr/commondata/local_public/protein-datasets/GeneOntology/processed/",
            },
            "common": {
                "feat": {
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
                "crop": False,
                "crop_size": None,
                "supervised": True,
                "uniform_recycling": False,
            },
            "train": {
                "fixed_size": True,
                "crop": True,
                "crop_size": 384,
                "supervised": True,
                "uniform_recycling": False,
            },
            "data_module":{
                "train_dataloader": {
                    "batch_size": 2,# Can only be 1, cause we don't apply cropping to proteins in the multiple binary classification task.It's a protein-level task.
                    "num_workers": 32,
                },
                "val_dataloader":{
                    "batch_size": 1, # Can only be 1, cause we don't apply cropping to proteins in the validation set
                    "num_workers": 16, # We want metrics about the complete proteins
                },
                "predict_dataloader": {
                    "batch_size": 1, # Can only be 1, cause we don't apply cropping to proteins in the prediction set
                    "num_workers": 0,# We want metrics about the complete proteins
                }
            }
        },
        "downstream":{
            "encoder": "alpha_encoder",
            "encoder_checkpoint": None,
            "head":{
                "task_num": 489, #EC: 538, GO-CC: 320, GO-MF: 489, GO-BP: 1943
                "num_mlp_layers": 3,
                "model_out_dim": 384,
            },
            "metric": ['f1_max'],
            "encoder_fixed": False,
            "reweight": False,
        },
        "model":{
            "embedder": {
                "protein_angle_embedder": {
                    "c_in": 57,
                    "c_m": c_s*2,
                    "c_out": c_s,
                },
                "gaussian_layer": {
                    "kernel_num": 16,
                    "num_pair_distance": 25
                },
                "bias_proj_layer": {
                    "input_dim": 400,# 25*16 = 400
                    "out_dim": c_z,
                    "activation_fn": activation_fn,
                    "hidden": 2*c_z,
                },
                "centrality_proj_layer":{
                    "input_dim": 400,
                    "out_dim": c_s,
                    "activation_fn": activation_fn,
                    "hidden": 2*c_s,
                }   
            },
            "constrainformer":{
                "c_s": c_s,
                "c_z": c_z,
                "c_hidden_msa_att": 32,
                "c_hidden_opm": 32,
                "c_hidden_mul": 128,
                "c_hidden_pair_att": 32,
                "no_heads_msa": 8,
                "no_heads_pair": 4,
                "no_blocks": 1,
                "transition_n": 4,
                "msa_dropout": 0.15,
                "pair_dropout": 0.25,
                "inf": 1e9,
                "eps": eps,  # 1e-10,
            },
            "structure_module": {
                "c_s": c_s,
                "c_z": c_z,
                "c_ipa": 16,
                "c_resnet": 128,
                "no_heads_ipa": 12,
                "no_qk_points": 4,
                "no_v_points": 8,
                "dropout_rate": 0.1,
                "no_blocks": 8,
                "no_transition_layers": 1,
                "no_resnet_blocks": 2,
                "no_angles": 7,
                "trans_scale_factor": 15,
                "epsilon": eps,  # 1e-12,
                "inf": 1e5,
            },
        },
        "loss": {
            "distogram": {
                "min_bin": 2.3125,
                "max_bin": 21.6875,
                "no_bins": 64,
                "eps": eps,  # 1e-6,
                "weight": 0.3,
            },
            "experimentally_resolved": {
                "eps": eps,  # 1e-8,
                "min_resolution": 0.0,
                "max_resolution": 3.0,
                "weight": 0.0,
            },
            "fape": {
                "backbone": {
                    "clamp_distance": 10.0,
                    "loss_unit_distance": 10.0,
                    "weight": 0.5,
                },
                "sidechain": {
                    "clamp_distance": 10.0,
                    "length_scale": 10.0,
                    "weight": 0.5,
                },
                "eps": 1e-4,
                "weight": 1.0,
            },
            "lddt": {
                "min_resolution": 0.0,
                "max_resolution": 3.0,
                "cutoff": 15.0,
                "no_bins": 50,
                "eps": eps,  # 1e-10,
                "weight": 0.01,
            },
            "masked_msa": {
                "eps": eps,  # 1e-8,
                "weight": 2.0,
            },
            "supervised_chi": {
                "chi_weight": 0.5,
                "angle_norm_weight": 0.01,
                "eps": eps,  # 1e-6,
                "weight": 1.0,
            },
            "violation": {
                "violation_tolerance_factor": 12.0,
                "clash_overlap_tolerance": 1.5,
                "eps": eps,  # 1e-6,
                "weight": 0.1,
            },
            "tm": {
                "max_bin": 31,
                "no_bins": 64,
                "min_resolution": 0.1,
                "max_resolution": 3.0,
                "eps": eps,  # 1e-8,
                "weight": 0.0,
                "enabled": tm_enabled,
            },
            "eps": eps,
        },
        "train": {
            "base_lr": 0.,
            "max_lr":1e-4,
            "warmup_no_steps": 20400,
            "start_decay_after_n_steps": 100000,
            "decay_every_n_steps": 6800, 
        }
    }
)
