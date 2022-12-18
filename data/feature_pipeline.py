import sys

# from Manifold.openfold.data.input_pipeline import compose
sys.path.append("..")
import copy
import logging
import pdb
from typing import Any, Dict, Mapping, Optional, Sequence

import ml_collections
import numpy as np
import torch
import torch.nn as nn

from data import data_transform
from utils import protein
from utils import residue_constants as rc
from utils.rigid_utils import Rigid, Rotation

from .data_transform import atom37_to_torsion_angles, pseudo_beta_fn

TensorDict = Dict[str, torch.Tensor]
FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

def build_decoy_angle_feats(decoy_feats):
    torsion_func = atom37_to_torsion_angles("decoy_")
    decoy_feats = torsion_func(decoy_feats)
    if decoy_feats is None:
        logging.warning(f"Some error occurs in atom37_to_torsion_angles")
        return None
    decoy_aatype = decoy_feats["decoy_aatype"]
    torsion_angles_sin_cos = decoy_feats["decoy_torsion_angles_sin_cos"]
    alt_torsion_angles_sin_cos = decoy_feats[
        "decoy_alt_torsion_angles_sin_cos"
    ]
    torsion_angles_mask = decoy_feats["decoy_torsion_angles_mask"]
    # pdb.set_trace()
    try:
        decoy_feats["decoy_angle_feats"] = torch.cat(
            [
                nn.functional.one_hot(decoy_aatype, 22),
                torsion_angles_sin_cos.reshape(
                    *torsion_angles_sin_cos.shape[:-2], 14
                ),
                alt_torsion_angles_sin_cos.reshape(
                    *alt_torsion_angles_sin_cos.shape[:-2], 14
                ),
                torsion_angles_mask,
                # node_dim: (B, N, 28)
                decoy_feats["node"],
            ],
            dim=-1,
        )
    except:
        node_feats = decoy_feats["node"]
        logging.warning(f"Some error occurs in decoy_angle_feats")
        logging.warning(f"decoy_aatype: {decoy_aatype.shape}, decoy_feats: {node_feats.shape}")

    return decoy_feats

def gather_edges(edges, neighbor_idx):
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)

def _rbf(D, num_rbf):
    D_min, D_max, D_count = 0., 20., num_rbf
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu.view([1,1,-1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return RBF

def _get_rbf(A, B, E_idx=None, num_rbf=16):
    if E_idx is not None:
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = _rbf(D_A_B_neighbors, num_rbf)
    else:
        D_A_B = torch.sqrt(torch.sum((A[..., None,:] - B[...,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        RBF_A_B = _rbf(D_A_B, num_rbf) #[B, L, L, 16]
    return RBF_A_B

def build_decoy_pair_feats(decoy_feats, min_bin, max_bin, no_bins, eps=1e-20, inf=1e8):
    # shape: [B, N, N, 15]
    edge_feat = decoy_feats["edge"]
    all_atom_positions = decoy_feats["decoy_all_atom_positions"]
    tpb, decoy_mask = pseudo_beta_fn(decoy_feats["decoy_aatype"], all_atom_positions, decoy_feats["decoy_all_atom_mask"])
    decoy_mask_2d = decoy_mask[..., None] * decoy_mask[..., None, :]
    dgram = torch.sum(
        (tpb[..., None, :] - tpb[..., None, :, :]) ** 2, dim=-1, keepdim=True
    )
    lower = torch.linspace(min_bin, max_bin, no_bins, device=tpb.device) ** 2
    upper = torch.cat([lower[:-1], lower.new_tensor([inf])], dim=-1)
    dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)
    # pdb.set_trace()
    to_concat = [dgram, decoy_mask_2d[..., None], edge_feat]

    aatype_one_hot = nn.functional.one_hot(
        decoy_feats["decoy_aatype"],
        rc.restype_num + 2,
    )
    # pdb.set_trace()
    n_res = decoy_feats["decoy_aatype"].shape[-1]
    to_concat.append(
        aatype_one_hot[..., None, :, :].expand(
            *aatype_one_hot.shape[:-2], n_res, -1, -1
        )
    )
    to_concat.append(
        aatype_one_hot[..., None, :].expand(
            *aatype_one_hot.shape[:-2], -1, n_res, -1
        )
    )

    n, ca, c = [rc.atom_order[a] for a in ["N", "CA", "C"]]
    rigids = Rigid.make_transform_from_reference(
        n_xyz=all_atom_positions[..., n, :],
        ca_xyz=all_atom_positions[..., ca, :],
        c_xyz=all_atom_positions[..., c, :],
        eps=eps,
    )
    points = rigids.get_trans()[..., None, :, :]
    rigid_vec = rigids[..., None].invert_apply(points)

    inv_distance_scalar = torch.rsqrt(eps + torch.sum(rigid_vec ** 2, dim=-1))

    d_aa_masks = decoy_feats["decoy_all_atom_mask"]
    decoy_mask = (
        d_aa_masks[..., n] * d_aa_masks[..., ca] * d_aa_masks[..., c]
    )
    decoy_mask_2d = decoy_mask[..., None] * decoy_mask[..., None, :]

    inv_distance_scalar = inv_distance_scalar * decoy_mask_2d
    unit_vector = rigid_vec * inv_distance_scalar[..., None]
    to_concat.extend(torch.unbind(unit_vector[..., None, :], dim=-1))
    to_concat.append(decoy_mask_2d[..., None])
    #######################################################
    # Add ProteinMPNN's edge feature to get better structure embedding
    # shape: [N, N, 25]
    pair_lst = ['Ca-Ca', 'Ca-C', 'C-Ca', 'Ca-N', 'N-Ca', 'Ca-O', 'O-Ca', 'C-C', 'C-N', 'N-C', 'C-O', 'O-C', 'N-N', 'N-O', 'O-N', 'O-O','Cb-Cb','Ca-Cb','Cb-Ca','C-Cb','Cb-C','N-Cb','Cb-N','O-Cb','Cb-O']
    n, ca, c, o, cb= [rc.atom_order[a] for a in ["N", "CA", "C", "O", "CB"]]
    atom_N = all_atom_positions[..., n, :]
    atom_Ca = all_atom_positions[..., ca, :]
    atom_C = all_atom_positions[..., c, :]
    atom_O = all_atom_positions[..., o, :]
    atom_Cb = all_atom_positions[..., cb, :]
    edge_dist = []
    for pair in pair_lst:
        atom1, atom2 = pair.split('-')
        rbf = _get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2])
        edge_dist.append(rbf)
    E_dist = torch.cat(tuple(edge_dist), dim=-1) # shape: [B, N, N, 25*16=400]
    to_concat.append(E_dist)
    # pdb.set_trace()
    act = torch.cat(to_concat, dim=-1)
    act = act * decoy_mask_2d[..., None]
    # pdb.set_trace()
    return act

def process_decoy(
    pdb_path: str, 
    gnn_features: dict,
    alt_seq: str,
    decoy_config,
    chain_id: Optional[str] = None,) -> FeatureDict:
    """
        Generate and Assemble features for a decoy in a PDB file.
    """
    with open(pdb_path, "r") as f:
        pdb_str = f.read()

    protein_object = protein.from_pdb_string(pdb_str, alt_seq, chain_id)
    #read pdb extract raw feature
    # pdb.set_trace()
    seqIOlen = len(alt_seq)
    proteinlen = len(protein_object.aatype)
    correct_seq_len = min(seqIOlen, proteinlen)
    decoy_feats = {}
    decoy_feats["decoy_all_atom_positions"] = protein_object.atom_positions[0:correct_seq_len].astype(np.float32)
    decoy_feats["decoy_all_atom_mask"] = protein_object.atom_mask[0:correct_seq_len].astype(np.float32)
    decoy_feats["decoy_aatype"] = protein_object.aatype[0:correct_seq_len]
    gnn_features["node"] = gnn_features["node"][0:correct_seq_len]
    gnn_features["edge"] = gnn_features["edge"][0:correct_seq_len, 0:correct_seq_len]
    # gnn_features["atom_emb"] = gnn_features["atom_emb"][0:correct_seq_len]
    # gnn_features["esm_emb"] = gnn_features["esm_emb"][:,0:correct_seq_len]
    num_res = decoy_feats["decoy_seq_length"] = correct_seq_len
    if correct_seq_len != decoy_feats["decoy_aatype"].shape[0]:
        # pdb.set_trace()
        logging.warning(f"the problematic pdb file is {pdb_path}, the length in protein_object is less than the length in gnn_features")
    decoy_feats["decoy_residue_index"] = np.array(range(num_res))
    feature_names = ["decoy_all_atom_positions", "decoy_all_atom_mask", "decoy_aatype","decoy_seq_length", "decoy_residue_index","node","edge","atom_emb","esm_emb"]
    # feature_names = ["decoy_all_atom_positions", "decoy_all_atom_mask", "decoy_aatype","decoy_seq_length", "decoy_residue_index"]
    decoy_feats.update(gnn_features)
    # pdb.set_trace()
    decoy_feats = np_to_tensor_dict(np_example=decoy_feats, features=feature_names)
    # Shifted the CoM to the center
    all_atom_positions = decoy_feats["decoy_all_atom_positions"]
    all_atom_mask = decoy_feats["decoy_all_atom_mask"]
    points = torch.reshape(all_atom_positions, (-1, 3))
    atom_num = torch.sum(all_atom_mask)
    CoM = torch.sum(points, dim=-2) / atom_num
    CoM = CoM[None,None,...]
    shifted_all_atom_positions = (all_atom_positions - CoM) * all_atom_mask[...,None]
    decoy_feats["decoy_all_atom_positions"] = shifted_all_atom_positions
    # Build Angle feature
    # decoy_feats = build_decoy_angle_feats(decoy_feats)
    # decoy_feats["decoy_pair_feats"] = build_decoy_pair_feats(
    #     decoy_feats,
    #     inf=decoy_config.inf,
    #     eps=decoy_config.eps,
    #     **decoy_config.distogram
    # )
    decoy_feats["decoy_seq_mask"] = torch.ones(
        decoy_feats["decoy_aatype"].shape, dtype=torch.float32
    )
    decoy_feats = data_transform.make_atom14_masks(decoy_feats)
    # decoy_feats["decoy_rigids"] = data_transform.atom37_to_rigids(decoy_feats["decoy_aatype"], decoy_feats["decoy_all_atom_positions"], decoy_feats["decoy_all_atom_mask"])
    # pdb.set_trace()

    # pdb.set_trace()
    return decoy_feats

def nonensembled_transform_fns():
    """Input pipeline data transformers that are not ensembled."""
    transforms = [
        data_transform.make_atom14_positions,
        data_transform.atom37_to_frames("label_"),
        data_transform.atom37_to_torsion_angles("label_"),
        data_transform.make_pseudo_beta("label_"),
        data_transform.get_backbone_frames("label_"),
        data_transform.get_chi_angles,
    ]

    return transforms

def np_to_tensor_dict(
    np_example: Mapping[str, np.ndarray],
    features: Sequence[str]
) -> TensorDict:
    """Creates dict of tensors from a dict of NumPy arrays.

    Args:
        np_example: A dict of NumPy feature arrays.
        features: A list of strings of feature names to be returned in the dataset.

    Returns:
        A dictionary of features mapping feature names to features. Only the given
        features are returned, all other ones are filtered out.
    """
    tensor_dict = {
        k: torch.tensor(v) for k, v in np_example.items() if k in features
    }
    return tensor_dict

@data_transform.curry1
def compose(x, fs):
    for f in fs:
        x = f(x)
    return x

def process_label(
    pdb_path: str,
    feats: FeatureDict,
    _output_raw: bool = False,
    chain_id: Optional[str] = None) -> FeatureDict:
    """
        Generate and Assemble features for a label in a PDB file.
    """
    with open(pdb_path, "r") as f:
        pdb_str = f.read()

    protein_object = protein.from_pdb_string(pdb_str, chain_id)
    label_feats = {}
    label_feats["label_all_atom_positions"] = protein_object.atom_positions.astype(np.float32)
    label_feats["label_all_atom_mask"] = protein_object.atom_mask.astype(np.float32)
    label_feats["label_aatype"] = protein_object.aatype
    # label_feats["label_sequence"] = protein._aatype_to_str_sequence(protein_object.aatype)
    features_name = ["label_all_atom_positions", "label_all_atom_mask", "label_aatype"]
    tensor_dict = np_to_tensor_dict(np_example=label_feats, features=features_name)
    tensor_dict["label_seq_mask"] = torch.ones(
        tensor_dict["label_aatype"].shape, dtype=torch.float32
    )
    feats.update(tensor_dict)
    nonensembled = nonensembled_transform_fns()
    feats = compose(nonensembled)(feats)
    # feats.update(tensors)

    return feats

def process_features(
    raw_features: FeatureDict, mode: str, config: ml_collections.ConfigDict):
    """Crop the decoys and cooresponding groundtruths to form the batches"""
    # Turn np array into tensor if any
    # return 1
    cfg = copy.deepcopy(config)
    mode_cfg = cfg[mode]
    num_res = int(raw_features["decoy_seq_length"])
    with cfg.unlocked():
        if mode_cfg.crop_size is None:
            # if num_res <= 300:
            mode_cfg.crop_size = num_res
            # pdb.set_trace()
            # mode_cfg.crop_size = 300
            # logging.warning("Sequence length is too long, crop to 320")
    feature_names = config.common.feat.keys()
    tensor_dict = np_to_tensor_dict(np_example= raw_features, features=feature_names)
    with torch.no_grad():
        crop_transforms = make_crop_transforms(config.common, mode_cfg)
        features = compose(crop_transforms)(tensor_dict)
    
    return {k: v for k, v in features.items()}


def make_crop_transforms(common_cfg, mode_cfg):
    crop_feats = dict(common_cfg.feat)
    transforms = [
        data_transform.random_crop_to_size(
            mode_cfg.crop_size,
            crop_feats
        ),
        data_transform.make_fixed_size(
            crop_feats,
            mode_cfg.crop_size
        )
    ]
    
    return transforms
    

