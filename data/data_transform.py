# Batched Protein Structure data format convertion
# from openfold.np import residue_constants 
from openfold.data.data_transforms import(
    atom37_to_torsion_angles
)
from openfold.np.residue_constants import (
    restype_rigid_group_default_frame,
)
from openfold.utils.tensor_utils import (
    batched_gather,
)
NUM_RES = "num residues placeholder"
from openfold.utils.affine_utils import T
from openfold.np import residue_constants as rc
from openfold.utils.rigid_utils import Rigid, Rotation
import torch
# import torch_geometric, torch_cluster
# from gvp.atom3d import _edge_features
# import pdb
import logging
import numpy as np
import itertools


def _init_residue_constants(float_dtype, device):

    default_frames = torch.tensor(
        restype_rigid_group_default_frame,
        dtype=float_dtype,
        device=device,
        requires_grad=False,
    )

    return default_frames

def Torsion_angles_to_frames(
    r: Rigid,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    rrgdf: torch.Tensor,
):
    # [*, N, 8, 4, 4]
    default_4x4 = rrgdf[aatype, ...]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat(
        [bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2
    )

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global    


def torsion_angles_to_frames(r, alpha, f):
    # Lazily initialize the residue constants on the correct device
    default_frames = _init_residue_constants(alpha.dtype, alpha.device)
    # Separated purely to make testing less annoying
    return Torsion_angles_to_frames(r, alpha, f, default_frames)


def FullAtom_to_SmOutput(aatype, all_atom_positions, all_atom_mask):
    protein = {}
    protein["aatype"] = aatype
    protein["all_atom_positions"] = all_atom_positions
    protein["all_atom_mask"] = all_atom_mask
    protein = atom37_to_frames(protein)
    protein = get_backbone_frames(protein)
    bb_rigid_tensor = protein["backbone_rigid_tensor"]
    rigids = Rigid.from_tensor_4x4(bb_rigid_tensor)
    backb_to_global = Rigid(
        Rotation(
            rot_mats=rigids.get_rots().get_rot_mats(), 
            quats=None
        ),
        rigids.get_trans(),
    )

    trans_scale_factor = 10
    backb_to_global = backb_to_global.scale_translation(
        trans_scale_factor
    )

    torsion_func = atom37_to_torsion_angles("")
    protein = torsion_func(protein)
    angles = protein["torsion_angles_sin_cos"]

    all_frames_to_global = torsion_angles_to_frames(
        backb_to_global,
        angles,
        aatype,
    )

    scaled_rigids = rigids.scale_translation(trans_scale_factor)

    preds = {
        "frames": scaled_rigids.to_tensor_7(),
        "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
        "angles": angles
    }

    return preds



    
###########################################################################
###########################################################################


## For Denoise Refinement
def curry1(f):
    """Supply all arguments but the first."""

    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc

def get_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
      A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
      in the order specified in rc.restypes + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in rc.restypes:
        residue_name = rc.restype_1to3[residue_name]
        residue_chi_angles = rc.chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append([rc.atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append(
                [0, 0, 0, 0]
            )  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return chi_atom_indices

@curry1
def atom37_to_torsion_angles(
    protein,
    prefix="",
):
    """
    Convert coordinates to torsion angles.

    This function is extremely sensitive to floating point imprecisions
    and should be run with double precision whenever possible.

    Args:
        Dict containing:
            * (prefix)aatype:
                [*, N_res] residue indices
            * (prefix)all_atom_positions:
                [*, N_res, 37, 3] atom positions (in atom37
                format)
            * (prefix)all_atom_mask:
                [*, N_res, 37] atom position mask
    Returns:
        The same dictionary updated with the following features:

        "(prefix)torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Torsion angles
        "(prefix)alt_torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Alternate torsion angles (accounting for 180-degree symmetry)
        "(prefix)torsion_angles_mask" ([*, N_res, 7])
            Torsion angles mask
    """
    aatype = protein[prefix + "aatype"]
    all_atom_positions = protein[prefix + "all_atom_positions"]
    all_atom_mask = protein[prefix + "all_atom_mask"]

    aatype = torch.clamp(aatype, max=20)

    pad = all_atom_positions.new_zeros(
        [*all_atom_positions.shape[:-3], 1, 37, 3]
    )
    prev_all_atom_positions = torch.cat(
        [pad, all_atom_positions[..., :-1, :, :]], dim=-3
    )

    pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, 37])
    prev_all_atom_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)

    pre_omega_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 1:3, :], all_atom_positions[..., :2, :]],
        dim=-2,
    )
    phi_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 2:3, :], all_atom_positions[..., :3, :]],
        dim=-2,
    )
    psi_atom_pos = torch.cat(
        [all_atom_positions[..., :3, :], all_atom_positions[..., 4:5, :]],
        dim=-2,
    )

    pre_omega_mask = torch.prod(
        prev_all_atom_mask[..., 1:3], dim=-1
    ) * torch.prod(all_atom_mask[..., :2], dim=-1)
    phi_mask = prev_all_atom_mask[..., 2] * torch.prod(
        all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype
    )
    psi_mask = (
        torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype)
        * all_atom_mask[..., 4]
    )

    chi_atom_indices = torch.as_tensor(
        get_chi_atom_indices(), device=aatype.device
    )

    atom_indices = chi_atom_indices[..., aatype, :, :]
    chis_atom_pos = batched_gather(
        all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2])
    )

    chi_angles_mask = list(rc.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)

    chis_mask = chi_angles_mask[aatype, :]

    chi_angle_atoms_mask = batched_gather(
        all_atom_mask,
        atom_indices,
        dim=-1,
        no_batch_dims=len(atom_indices.shape[:-2]),
    )
    chi_angle_atoms_mask = torch.prod(
        chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype
    )
    chis_mask = chis_mask * chi_angle_atoms_mask
    if chis_atom_pos.shape[0] == pre_omega_atom_pos.shape[0]:
        try:
            torsions_atom_pos = torch.cat(
                [
                    pre_omega_atom_pos[..., None, :, :],
                    phi_atom_pos[..., None, :, :],
                    psi_atom_pos[..., None, :, :],
                    chis_atom_pos,
                ],
                dim=-3,
            )
        except RuntimeError:
            logging.warning("there are errors within torch.cat() function")
            return None
    else:
        logging.warning("the torsions_atom_pos batch_size mismatch")
        return None
        
    torsion_angles_mask = torch.cat(
        [
            pre_omega_mask[..., None],
            phi_mask[..., None],
            psi_mask[..., None],
            chis_mask,
        ],
        dim=-1,
    )

    torsion_frames = T.from_3_points(
        torsions_atom_pos[..., 1, :],
        torsions_atom_pos[..., 2, :],
        torsions_atom_pos[..., 0, :],
        eps=1e-8,
    )

    fourth_atom_rel_pos = torsion_frames.invert().apply(
        torsions_atom_pos[..., 3, :]
    )

    torsion_angles_sin_cos = torch.stack(
        [fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1
    )

    denom = torch.sqrt(
        torch.sum(
            torch.square(torsion_angles_sin_cos),
            dim=-1,
            dtype=torsion_angles_sin_cos.dtype,
            keepdims=True,
        )
        + 1e-8
    )
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom

    torsion_angles_sin_cos = torsion_angles_sin_cos * all_atom_mask.new_tensor(
        [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
    )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]

    chi_is_ambiguous = torsion_angles_sin_cos.new_tensor(
        rc.chi_pi_periodic,
    )[aatype, ...]

    mirror_torsion_angles = torch.cat(
        [
            all_atom_mask.new_ones(*aatype.shape, 3),
            1.0 - 2.0 * chi_is_ambiguous,
        ],
        dim=-1,
    )

    alt_torsion_angles_sin_cos = (
        torsion_angles_sin_cos * mirror_torsion_angles[..., None]
    )

    protein[prefix + "torsion_angles_sin_cos"] = torsion_angles_sin_cos
    protein[prefix + "alt_torsion_angles_sin_cos"] = alt_torsion_angles_sin_cos
    protein[prefix + "torsion_angles_mask"] = torsion_angles_mask

    return protein

def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    is_gly = aatype == rc.restype_order["G"]
    ca_idx = rc.atom_order["CA"]
    cb_idx = rc.atom_order["CB"]
    pseudo_beta = torch.where(
        is_gly[..., None].expand(*((-1,) * len(is_gly.shape)), 3),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    if all_atom_masks is not None:
        pseudo_beta_mask = torch.where(
            is_gly,
            all_atom_masks[..., ca_idx],
            all_atom_masks[..., cb_idx],
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta

@curry1
def make_pseudo_beta(protein, prefix=""):
    """Create pseudo-beta (alpha for glycine) position and mask."""
    assert prefix in ["decoy_", "label_"]
    
    (
        protein[prefix + "pseudo_beta"],
        protein[prefix + "pseudo_beta_mask"],
    ) = pseudo_beta_fn(
        protein["label_aatype" if prefix == "label_" else "decoy_aatype"],
        protein[prefix + "all_atom_positions"],
        protein["label_all_atom_mask" if prefix == "label_" else "decoy_all_atom_mask"],
    )
    return protein

def atom37_to_rigids(aatype, all_atom_positions, all_atom_mask):
    protein = {}
    protein["aatype"] = aatype
    protein["all_atom_positions"] = all_atom_positions
    protein["all_atom_mask"] = all_atom_mask
    FrameTrans = atom37_to_frames("")
    protein = FrameTrans(protein)
    GetBackbone = get_backbone_frames("")
    protein = GetBackbone(protein)
    bb_rigid_tensor = protein["backbone_rigid_tensor"]
    rigids = Rigid.from_tensor_4x4(bb_rigid_tensor)
    return rigids

def make_atom14_masks(protein):
    """Construct denser atom positions (14 dimensions instead of 37)."""
    # pdb.set_trace()
    restype_atom14_to_atom37 = []
    restype_atom37_to_atom14 = []
    restype_atom14_mask = []

    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        restype_atom14_to_atom37.append(
            [(rc.atom_order[name] if name else 0) for name in atom_names]
        )
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append(
            [
                (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                for name in rc.atom_types
            ]
        )

        restype_atom14_mask.append(
            [(1.0 if name else 0.0) for name in atom_names]
        )

    # Add dummy mapping for restype 'UNK'
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.0] * 14)

    restype_atom14_to_atom37 = torch.tensor(
        restype_atom14_to_atom37,
        dtype=torch.int32,
        device=protein["decoy_aatype"].device,
    )
    restype_atom37_to_atom14 = torch.tensor(
        restype_atom37_to_atom14,
        dtype=torch.int32,
        device=protein["decoy_aatype"].device,
    )
    restype_atom14_mask = torch.tensor(
        restype_atom14_mask,
        dtype=torch.float32,
        device=protein["decoy_aatype"].device,
    )

    # create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein
    residx_atom14_to_atom37 = restype_atom14_to_atom37[protein["decoy_aatype"]]
    residx_atom14_mask = restype_atom14_mask[protein["decoy_aatype"]]

    protein["decoy_atom14_atom_exists"] = residx_atom14_mask
    protein["decoy_residx_atom14_to_atom37"] = residx_atom14_to_atom37.long()

    # create the gather indices for mapping back
    residx_atom37_to_atom14 = restype_atom37_to_atom14[protein["decoy_aatype"]]
    protein["decoy_residx_atom37_to_atom14"] = residx_atom37_to_atom14.long()

    # create the corresponding mask
    restype_atom37_mask = torch.zeros(
        [21, 37], dtype=torch.float32, device=protein["decoy_aatype"].device
    )
    for restype, restype_letter in enumerate(rc.restypes):
        restype_name = rc.restype_1to3[restype_letter]
        atom_names = rc.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = rc.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[protein["decoy_aatype"]]
    protein["decoy_atom37_atom_exists"] = residx_atom37_mask

    return protein

def make_atom14_positions(protein):
    """Constructs denser atom positions (14 dimensions instead of 37)."""
    residx_atom14_mask = protein["decoy_atom14_atom_exists"]
    residx_atom14_to_atom37 = protein["decoy_residx_atom14_to_atom37"]

    # Create a mask for known ground truth positions.
    residx_atom14_gt_mask = residx_atom14_mask * batched_gather(
        protein["label_all_atom_mask"],
        residx_atom14_to_atom37,
        dim=-1,
        no_batch_dims=len(protein["label_all_atom_mask"].shape[:-1]),
    )

    # Gather the ground truth positions.
    residx_atom14_gt_positions = residx_atom14_gt_mask[..., None] * (
        batched_gather(
            protein["label_all_atom_positions"],
            residx_atom14_to_atom37,
            dim=-2,
            no_batch_dims=len(protein["label_all_atom_positions"].shape[:-2]),
        )
    )

    protein["decoy_atom14_atom_exists"] = residx_atom14_mask
    protein["label_atom14_gt_exists"] = residx_atom14_gt_mask
    protein["label_atom14_gt_positions"] = residx_atom14_gt_positions

    # As the atom naming is ambiguous for 7 of the 20 amino acids, provide
    # alternative ground truth coordinates where the naming is swapped
    restype_3 = [rc.restype_1to3[res] for res in rc.restypes]
    restype_3 += ["UNK"]

    # Matrices for renaming ambiguous atoms.
    all_matrices = {
        res: torch.eye(
            14,
            dtype=protein["label_all_atom_mask"].dtype,
            device=protein["label_all_atom_mask"].device,
        )
        for res in restype_3
    }
    for resname, swap in rc.residue_atom_renaming_swaps.items():
        correspondences = torch.arange(
            14, device=protein["label_all_atom_mask"].device
        )
        for source_atom_swap, target_atom_swap in swap.items():
            source_index = rc.restype_name_to_atom14_names[resname].index(
                source_atom_swap
            )
            target_index = rc.restype_name_to_atom14_names[resname].index(
                target_atom_swap
            )
            correspondences[source_index] = target_index
            correspondences[target_index] = source_index
            renaming_matrix = protein["label_all_atom_mask"].new_zeros((14, 14))
            for index, correspondence in enumerate(correspondences):
                renaming_matrix[index, correspondence] = 1.0
        all_matrices[resname] = renaming_matrix
    renaming_matrices = torch.stack(
        [all_matrices[restype] for restype in restype_3]
    )

    # Pick the transformation matrices for the given residue sequence
    # shape (num_res, 14, 14).
    renaming_transform = renaming_matrices[protein["label_aatype"]]

    # Apply it to the ground truth positions. shape (num_res, 14, 3).
    alternative_gt_positions = torch.einsum(
        "...rac,...rab->...rbc", residx_atom14_gt_positions, renaming_transform
    )
    protein["label_atom14_alt_gt_positions"] = alternative_gt_positions

    # Create the mask for the alternative ground truth (differs from the
    # ground truth mask, if only one of the atoms in an ambiguous pair has a
    # ground truth position).
    alternative_gt_mask = torch.einsum(
        "...ra,...rab->...rb", residx_atom14_gt_mask, renaming_transform
    )
    protein["label_atom14_alt_gt_exists"] = alternative_gt_mask

    # Create an ambiguous atoms mask.  shape: (21, 14).
    restype_atom14_is_ambiguous = protein["label_all_atom_mask"].new_zeros((21, 14))
    for resname, swap in rc.residue_atom_renaming_swaps.items():
        for atom_name1, atom_name2 in swap.items():
            restype = rc.restype_order[rc.restype_3to1[resname]]
            atom_idx1 = rc.restype_name_to_atom14_names[resname].index(
                atom_name1
            )
            atom_idx2 = rc.restype_name_to_atom14_names[resname].index(
                atom_name2
            )
            restype_atom14_is_ambiguous[restype, atom_idx1] = 1
            restype_atom14_is_ambiguous[restype, atom_idx2] = 1

    # From this create an ambiguous_mask for the given sequence.
    protein["label_atom14_atom_is_ambiguous"] = restype_atom14_is_ambiguous[
        protein["label_aatype"]
    ]

    return protein
    
def get_chi_angles(protein):
    dtype = protein["label_all_atom_mask"].dtype
    protein["label_chi_angles_sin_cos"] = (
        protein["label_torsion_angles_sin_cos"][..., 3:, :]
    ).to(dtype)
    protein["label_chi_mask"] = protein["label_torsion_angles_mask"][..., 3:].to(dtype)

    return protein

@curry1
def get_backbone_frames(protein, prefix=""):
    # DISCREPANCY: AlphaFold uses tensor_7s here. I don't know why.
    protein[prefix + "backbone_rigid_tensor"] = protein[prefix + "rigidgroups_gt_frames"][
        ..., 0, :, :
    ]
    protein[prefix + "backbone_rigid_mask"] = protein[prefix + "rigidgroups_gt_exists"][..., 0]

    return protein

@curry1
def atom37_to_frames(protein, prefix=""):
    aatype = protein[prefix + "aatype"]
    all_atom_positions = protein[prefix + "all_atom_positions"]
    all_atom_mask = protein[prefix + "all_atom_mask"]

    batch_dims = len(aatype.shape[:-1])

    restype_rigidgroup_base_atom_names = np.full([21, 8, 3], "", dtype=object)
    restype_rigidgroup_base_atom_names[:, 0, :] = ["C", "CA", "N"]
    restype_rigidgroup_base_atom_names[:, 3, :] = ["CA", "C", "O"]

    for restype, restype_letter in enumerate(rc.restypes):
        resname = rc.restype_1to3[restype_letter]
        for chi_idx in range(4):
            if rc.chi_angles_mask[restype][chi_idx]:
                names = rc.chi_angles_atoms[resname][chi_idx]
                restype_rigidgroup_base_atom_names[
                    restype, chi_idx + 4, :
                ] = names[1:]

    restype_rigidgroup_mask = all_atom_mask.new_zeros(
        (*aatype.shape[:-1], 21, 8),
    )
    restype_rigidgroup_mask[..., 0] = 1
    restype_rigidgroup_mask[..., 3] = 1
    restype_rigidgroup_mask[..., :20, 4:] = all_atom_mask.new_tensor(
        rc.chi_angles_mask
    )

    lookuptable = rc.atom_order.copy()
    lookuptable[""] = 0
    lookup = np.vectorize(lambda x: lookuptable[x])
    restype_rigidgroup_base_atom37_idx = lookup(
        restype_rigidgroup_base_atom_names,
    )
    restype_rigidgroup_base_atom37_idx = aatype.new_tensor(
        restype_rigidgroup_base_atom37_idx,
    )
    restype_rigidgroup_base_atom37_idx = (
        restype_rigidgroup_base_atom37_idx.view(
            *((1,) * batch_dims), *restype_rigidgroup_base_atom37_idx.shape
        )
    )

    residx_rigidgroup_base_atom37_idx = batched_gather(
        restype_rigidgroup_base_atom37_idx,
        aatype,
        dim=-3,
        no_batch_dims=batch_dims,
    )

    base_atom_pos = batched_gather(
        all_atom_positions,
        residx_rigidgroup_base_atom37_idx,
        dim=-2,
        no_batch_dims=len(all_atom_positions.shape[:-2]),
    )

    gt_frames = T.from_3_points(
        p_neg_x_axis=base_atom_pos[..., 0, :],
        origin=base_atom_pos[..., 1, :],
        p_xy_plane=base_atom_pos[..., 2, :],
        eps=1e-8,
    )

    group_exists = batched_gather(
        restype_rigidgroup_mask,
        aatype,
        dim=-2,
        no_batch_dims=batch_dims,
    )

    gt_atoms_exist = batched_gather(
        all_atom_mask,
        residx_rigidgroup_base_atom37_idx,
        dim=-1,
        no_batch_dims=len(all_atom_mask.shape[:-1]),
    )
    gt_exists = torch.min(gt_atoms_exist, dim=-1)[0] * group_exists

    rots = torch.eye(3, dtype=all_atom_mask.dtype, device=aatype.device)
    rots = torch.tile(rots, (*((1,) * batch_dims), 8, 1, 1))
    rots[..., 0, 0, 0] = -1
    rots[..., 0, 2, 2] = -1

    gt_frames = gt_frames.compose(T(rots, None))

    restype_rigidgroup_is_ambiguous = all_atom_mask.new_zeros(
        *((1,) * batch_dims), 21, 8
    )
    restype_rigidgroup_rots = torch.eye(
        3, dtype=all_atom_mask.dtype, device=aatype.device
    )
    restype_rigidgroup_rots = torch.tile(
        restype_rigidgroup_rots,
        (*((1,) * batch_dims), 21, 8, 1, 1),
    )

    for resname, _ in rc.residue_atom_renaming_swaps.items():
        restype = rc.restype_order[rc.restype_3to1[resname]]
        chi_idx = int(sum(rc.chi_angles_mask[restype]) - 1)
        restype_rigidgroup_is_ambiguous[..., restype, chi_idx + 4] = 1
        restype_rigidgroup_rots[..., restype, chi_idx + 4, 1, 1] = -1
        restype_rigidgroup_rots[..., restype, chi_idx + 4, 2, 2] = -1

    residx_rigidgroup_is_ambiguous = batched_gather(
        restype_rigidgroup_is_ambiguous,
        aatype,
        dim=-2,
        no_batch_dims=batch_dims,
    )

    residx_rigidgroup_ambiguity_rot = batched_gather(
        restype_rigidgroup_rots,
        aatype,
        dim=-4,
        no_batch_dims=batch_dims,
    )

    alt_gt_frames = gt_frames.compose(T(residx_rigidgroup_ambiguity_rot, None))

    gt_frames_tensor = gt_frames.to_4x4()
    alt_gt_frames_tensor = alt_gt_frames.to_4x4()

    protein[prefix + "rigidgroups_gt_frames"] = gt_frames_tensor
    protein[prefix + "rigidgroups_gt_exists"] = gt_exists
    protein[prefix + "rigidgroups_group_exists"] = group_exists
    protein[prefix + "rigidgroups_group_is_ambiguous"] = residx_rigidgroup_is_ambiguous
    protein[prefix + "rigidgroups_alt_gt_frames"] = alt_gt_frames_tensor

    return protein

@curry1
def random_crop_to_size(
    protein,
    crop_size,
    shape_schema
):
    g = torch.Generator(device=protein["decoy_aatype"].device)
    seq_length = protein["decoy_seq_length"]
    num_res_crop_size = min(int(seq_length), crop_size)

    def _randint(lower, upper):
        return int(torch.randint(
                lower,
                upper + 1,
                (1,),
                device=protein["decoy_seq_length"].device,
                generator=g,
        )[0])

    n = seq_length - num_res_crop_size
    right_anchor = n
    num_res_crop_start = _randint(0, right_anchor)
    for k, v in protein.items():
        if k not in shape_schema or NUM_RES not in shape_schema[k]:
            continue
        slices = []
        for i, (dim_size, dim) in enumerate(zip(shape_schema[k], v.shape)):
            is_num_res = dim_size == NUM_RES
            crop_start = num_res_crop_start if is_num_res else 0
            crop_size = num_res_crop_size if is_num_res else dim
            slices.append(slice(crop_start, crop_start + crop_size))
        protein[k] = v[slices]
    
    protein["decoy_seq_length"] = protein["decoy_seq_length"].new_tensor(num_res_crop_size)
    return protein

@curry1
def make_seq_mask(prefix, protein):
    protein[prefix + "seq_mask"] = torch.ones(
        protein["aatype"].shape, dtype=torch.float32
    )
    return protein

@curry1
def make_fixed_size(
    protein, 
    shape_schema,
    num_res=0,
):
    pad_size_map = {
        NUM_RES: num_res
    }
    for k, v in protein.items():
        # Don't transfer this to the accelerator.
        shape = list(v.shape)
        schema = shape_schema[k]
        msg = "Rank mismatch between shape and shape schema for"
        assert len(shape) == len(schema), f"{msg} {k}: {shape} vs {schema}"
        pad_size = [
            pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)
        ]

        padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
        padding.reverse()
        padding = list(itertools.chain(*padding))
        if padding:
            protein[k] = torch.nn.functional.pad(v, padding)
            protein[k] = torch.reshape(protein[k], pad_size)
    
    return protein


def atom14_to_atom37(atom14, batch):
    atom37_data = batched_gather(
        atom14,
        batch["decoy_residx_atom37_to_atom14"],
        dim=-2,
        no_batch_dims=len(atom14.shape[:-2]),
    )
    # pdb.set_trace()
    atom37_data = atom37_data * batch["decoy_atom37_atom_exists"][..., None]

    return atom37_data


## For full-atom scale refinement

# def Protein_Structure_Convert(all_atom_position, all_atom_mask, seq_mask, aatype, device):
#     """
#     Args:
#         all_atom_position:
#             shape: [N, 37, 3]
#         all_atom_mask: 
#             shape: [N, 37]
#         seq_mask:
#             shape: [N,]
#         aatype:
#             shape: [N,]
#         device:
#             "cpu" or "CUDA"
#     Note:
#         This function get seqs with seqs padding
#     function:
#         Convert full atom position format in AlphaFold2 to atom-level protein graph in torch_geometric.data.Data format in GVP-GNN
#         The key is to eliminate padding and find cooresponding atom type
    
#     """
#     # get rid of sequence padding
#     seq_index = torch.sum(seq_mask, dtype=torch.int)
#     # pdb.set_trace()
#     # logging.warning(f"the seq_index is {seq_index}")
#     all_atom_position = all_atom_position[:seq_index]
#     all_atom_mask = all_atom_mask[:seq_index]
#     aatype = aatype[:seq_index]
#     # Some constants for graph building
#     edge_cutoff = 4.5 # build a edge between atoms with distance < 4.5 A
#     num_rbf=16

#     # numeric representation of atom type
#     atom_orders = residue_constants.atom_order.values()
#     # [N_atom, 3](atom coordinates without atom padding)
#     coords = []
#     # [N_atom] (atom type without padding)
#     atoms = []
#     # [N*37, 3] (atom coordinates with atom padding)
#     # all_atom_position = all_atom_position.reshape(-1,3)

#     # skip padding according to all_atom_mask
#     for i in range(aatype.shape[0]):
#         for atom_type, atom_pos, mask in zip(
#             atom_orders, all_atom_position[i], all_atom_mask[i]
#         ):
#             # pdb.set_trace()
#             if mask < 0.5:
#                 continue
#             coords.append(atom_pos.cpu().detach().numpy())
#             atoms.append(atom_type)
    
#     # Build Protein Graph
#     with torch.no_grad():
#         try:
#             if len(coords) != 0:
#                 coords = torch.as_tensor(np.array(coords), dtype=torch.float32, device=device)
#                 atoms = torch.as_tensor(np.array(atoms), dtype=torch.long, device=device)
#                 edge_index = torch_cluster.radius_graph(coords, r=edge_cutoff)
#                 edge_s, edge_v = _edge_features(coords, edge_index, D_max=edge_cutoff, num_rbf=num_rbf,device=device)
#             else:
#                 # logging.warning(f"\nSome error occur in Protein Convert, the content of all_atom_position is {all_atom_position}, the content of seq mask is {seq_mask}, , the content of all_atom_mask is {all_atom_mask}")
#                 return None
#         except BaseException:
#             logging.warning(f"\nsome errors occur in Protein Convert, the shape of coords is {coords.shape}, the content is {coords}")
#             logging.warning(f"\nsome errors occur in Protein Convert, the shape of coords is {all_atom_position.shape}, the content is {all_atom_position}")
#             logging.warning(f"the seq_mask is {seq_mask}, the seq index is {seq_index}, the aatype is {aatype}")
#             import traceback
#             traceback.print_exc()
#             return None
        
#         return torch_geometric.data.Data(x=coords.unsqueeze(-2), atoms=atoms, edge_index=edge_index, edge_s=edge_s, edge_v=edge_v)

# def Batched_Graph(all_atom_position, all_atom_mask, seq_mask, aatype, device):
#     """
#     Args:
#         all_atom_position:
#             shape: [*, N, 37, 3]
#         all_atom_mask: 
#             shape: [*, N, 37]
#         seq_mask:
#             shape: [*, N]
#         aatype:
#             shape: [*, N]
#         device:
#             "cpu" or "CUDA"
#     function:
#         Convert full atom position format in AlphaFold2 to atom-level batched protein graph in torch_geometric.data.Data format in GVP-GNN
#         The key is to eliminate padding and find cooresponding atom type
    
#     """
#     PyG_Data_List = []
#     # For every protein in the batch
#     # pdb.set_trace()
#     for i in range(aatype.shape[0]):
#         protein_graph = Protein_Structure_Convert(all_atom_position[i],all_atom_mask[i], seq_mask[i], aatype[i], device)
#         if protein_graph is not None:
#             PyG_Data_List.append(protein_graph)
#         else:
#             logging.warning(f"\none protein_graph construction failed")
#     # pdb.set_trace()
#     if len(PyG_Data_List) != 0:
#         batch_graph = torch_geometric.data.Batch.from_data_list(PyG_Data_List)
#     else:
#         batch_graph = None
#         logging.warning(f"\nthe batch_graph is None, Please Check the data")
#         logging.warning(f"\none protein_graph construction failed, the coords is {all_atom_position}, the all_atom_mask is {all_atom_mask}, the seq_mask is {seq_mask}, the aatype is {aatype}, please check")

#     return batch_graph

# def Graph_to_FullAtom(coords, all_atom_mask, seq_mask):
#     '''
#     Args:
#         Atributes of one single protein
#         coords: 
#             shape: [N_atom',3]
#             full atom representation without seq padding and atom padding
#         all_atom_mask:
#             shape: [N, 37]
#             atom_mask and seq mask
#         seq_mask:
#             shape: [N,]
#             seq mask, coresponding to sequence padding 
#     '''
#     # Atom position with padding, shape [N, 37, 3]
#     atom_position = torch.zeros((all_atom_mask.shape[0], all_atom_mask.shape[1],3))
#     atom_index = 0
#     # Fill the updated atom coordinates
#     # pdb.set_trace()
#     for res in range(atom_position.shape[0]):
#         if seq_mask[res] < 0.5:
#             continue
#         for atom_type, atom_mask in enumerate(all_atom_mask[res]):
#             if atom_mask > 0.5:
#                 atom_position[res][atom_type] = coords[atom_index]
#                 atom_index += 1
    
#     return atom_position   

# def BatchedGraph_to_FullAtom(model_out, batch_graph, all_atom_mask, seq_mask, device):
#     '''
#     Args:
#         model_out:
#             shape: [N_atom, 1, 3]
#         batch_graph:
#             batched graph, we need to use batch.batch to split batched graph to individul graph and finally convert individul graph to full-atom format in AlphaFold
#         all_atom_mask:
#             shape: [*, N, 37]
#         seq_mask:
#             shape: [*, N,]
#         we need to spilt BatchedGraph to batched full-atom format and calculate loss
#     '''
#     # Update the coordinates
#     batch_graph.x = model_out
#     # Split the batched Graph
#     PyG_Data_list = batch_graph.to_data_list()
#     # batched Full atom postion
#     # shape [*, N, 37, 3]
#     all_atom_position = []
#     offset = 0
#     Update_Protein_id = []
#     for index, Graph in enumerate(PyG_Data_list):
#         # coords: shape [N_atom, 1, 3]
#         coords = Graph.x.squeeze(-2)
#         while torch.count_nonzero(all_atom_mask[index+offset]).item() == 0:
#            offset+=1
#         Update_Protein_id.append(index+offset) 
#         coords = Graph_to_FullAtom(coords, all_atom_mask[index+offset], seq_mask[index+offset])
#         all_atom_position.append(coords.cpu().detach().numpy())
    
#     return torch.as_tensor(np.array(all_atom_position), dtype=torch.float32, device=device),torch.as_tensor(np.array(Update_Protein_id), dtype=torch.int, device=device)