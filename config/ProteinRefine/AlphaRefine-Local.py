import copy

import ml_collections as mlc

from config._base import register_config


def set_inf(c, inf):
    for k, v in c.items():
        if isinstance(v, mlc.ConfigDict):
            set_inf(v, inf)
        elif k == "inf":
            c[k] = inf

@register_config("AlphaRefine-Local")
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

NUM_RES = "num residues placeholder"

config = mlc.ConfigDict(
    {
        "data":{
            "dataset":{
                "name": "DeepAccNet",
                "train_targets_path": "/root/HibikeFold/RefineDiff-FullAtom/DecoyDataset/train_proteins.npy",
                "eval_targets_path": "/root/HibikeFold/RefineDiff-FullAtom/DecoyDataset/valid_proteins.npy",
                "predict_targets_path": None,
                "root_dir": "/root/HibikeFold/RefineDiff-FullAtom/DecoyDataset/pdbs/",
                "gfeat_save_dir": "/root/HibikeFold/RefineDiff-FullAtom/DecoyDataset/gfeat/",
                "esm_save_dir": "/root/HibikeFold/RefineDiff-FullAtom/DecoyDataset/esm/",
                "include_native": False,
                "training_mode": True,
                "eval": True,
            },
            "decoy":{
                "distogram": {
                    "min_bin": 3.25,
                    "max_bin": 50.75,
                    "no_bins": 39,
                },
                "inf": 1e5,  # 1e9,
                "eps": eps,  # 1e-6,
            },
            "common": {
                "feat": {
                    "decoy_aatype": [NUM_RES],
                    "decoy_seq_mask": [NUM_RES],
                    "decoy_all_atom_mask": [NUM_RES, None],
                    "decoy_all_atom_positions": [NUM_RES, None, None],
                    "decoy_angle_feats": [NUM_RES, None],
                    "decoy_pair_feats": [NUM_RES, NUM_RES, None],
                    "atom_emb": [NUM_RES, None, None],
                    "esm_emb": [None, NUM_RES, None],#:=5120
                    "decoy_atom14_atom_exists": [NUM_RES, None],
                    "decoy_residx_atom14_to_atom37": [NUM_RES, None],
                    "decoy_residx_atom37_to_atom14": [NUM_RES, None],
                    "decoy_seq_length": [],
                    "decoy_residue_index": [NUM_RES],
                    "decoy_atom37_atom_exists": [NUM_RES, None],
                    "decoy_torsion_angles_sin_cos":[NUM_RES, None, None],
                    "decoy_alt_torsion_angles_sin_cos": [NUM_RES, None, None],
                    "decoy_torsion_angles_mask": [NUM_RES, None],
                    "label_aatype": [NUM_RES],
                    "label_seq_mask": [NUM_RES],
                    "label_all_atom_mask": [NUM_RES, None],
                    "label_all_atom_positions": [NUM_RES, None, None],
                    "label_atom14_alt_gt_exists": [NUM_RES, None],
                    "label_atom14_alt_gt_positions": [NUM_RES, None, None],
                    "label_atom14_atom_is_ambiguous": [NUM_RES, None],
                    "label_atom14_gt_exists": [NUM_RES, None],
                    "label_atom14_gt_positions": [NUM_RES, None, None],
                    "label_backbone_rigid_mask": [NUM_RES],
                    "label_backbone_rigid_tensor": [NUM_RES, None, None],
                    "label_chi_angles_sin_cos": [NUM_RES, None, None],
                    "label_chi_mask": [NUM_RES, None],
                    "label_pseudo_beta": [NUM_RES, None],
                    "label_pseudo_beta_mask": [NUM_RES],
                    "label_rigidgroups_alt_gt_frames": [NUM_RES, None, None, None],
                    "label_rigidgroups_group_exists": [NUM_RES, None],
                    "label_rigidgroups_group_is_ambiguous": [NUM_RES, None],
                    "label_rigidgroups_gt_exists": [NUM_RES, None],
                    "label_rigidgroups_gt_frames": [NUM_RES, None, None, None],
                  	"node": [NUM_RES, None],
                  	"edge": [NUM_RES, NUM_RES, None],
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
                "crop_size": 256,
                "supervised": True,
                "clamp_prob": 0.9,
                "uniform_recycling": True,
            },
            "data_module":{
                "train_dataloader": {
                    "batch_size": 2,
                    "num_workers": 16,
                },
                "val_dataloader":{
                    "batch_size": 1, # Can only be 1, cause we don't apply cropping to proteins in the validation set
                    "num_workers": 32, # We want metrics about the complete proteins
                },
                "predict_dataloader": {
                    "batch_size": 1, # Can only be 1, cause we don't apply cropping to proteins in the prediction set
                    "num_workers": 0,# We want metrics about the complete proteins
                }
            }
        },
        "globals": {
            "blocks_per_ckpt": blocks_per_ckpt,
            "chunk_size": chunk_size,
            "c_z": c_z,
            "c_m": c_m,
            "c_t": c_t,
            "c_e": c_e,
            "c_s": c_s,
            "eps": eps,
            "max_recycling_iters":2,
            "num_steps":100,
            "model_class": "denoise_module"
        },
        "model":{
            "_mask_trans": False,
            "decoy":{
                "decoy_angle_embedder": {
                    # DISCREPANCY: c_in is supposed to be 51.
                    "c_in": 320,
                    "c_m": c_m,
                    "c_out": c_s,
                },
                "decoy_pair_embedder": {
                    "c_in": 503,
                    "c_out": c_t,
                },
                "decoy_atom_embedder": {
                    "atom_emb_in": 7,
                    "atom_emb_h": 256,
                    "norm": 'instance',
                    "activation": 'relu',
                },
                "decoy_pair_stack": {
                    "c_t": c_t,
                    # DISCREPANCY: c_hidden_tri_att here is given in the supplement
                    # as 64. In the code, it's 16.
                    "c_hidden_tri_att": 16,
                    "c_hidden_tri_mul": 64,
                    "no_blocks": 2,
                    "no_heads": 4,
                    "pair_transition_n": 2,
                    "dropout_rate": 0.25,
                    "blocks_per_ckpt": blocks_per_ckpt,
                    "inf": 1e9,
                },
                "decoy_pointwise_attention": {
                    "c_t": c_t,
                    "c_z": c_z,
                    # DISCREPANCY: c_hidden here is given in the supplement as 64.
                    # It's actually 16.
                    "c_hidden": 16,
                    "no_heads": 4,
                    "inf": 1e5,  # 1e9,
                },
                "decoy_seq_stack":{
                    "c_s": c_s,
                    "c_hidden":64,
                    "no_heads": 16,
                    "inf": 1e5,
                    "dropout_rate": 0.25,
                    "no_blocks":5
                }
            },
            "input_embedder": {
                "c_in": 5120,
                "c_hidden": 2048,
                "c_out": c_s,
            },
            "outer_product_mean":{
                "c_s": c_s,
                "c_z": c_z,
                "c_hidden": 32
            },
            "recycling_embedder": {
                "c_z": c_z,
                "c_m": c_s,
                "min_bin": 3.25,
                "max_bin": 20.75,
                "no_bins": 15,
                "inf": 1e8,
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
        }
    }
)
