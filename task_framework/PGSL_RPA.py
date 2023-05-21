import pdb
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sublayers.structure_head import StructureHead

from modules.common.so3 import rotation_to_so3vec, so3vec_to_rotation
from modules.diffusion.transition import PositionTransition, RotationTransition
from utils.rigid_utils import Rigid, Rotation
from utils import residue_constants


class PGSL_head(nn.Module):
    """
    The main part of the PGSL-RPA Framework, it take the embeddings of encoder and decode the structure.
    Or it can be seen as pretrain head of the PGSL-RPA framework.
    """
    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)

        """
        super(PGSL_head, self).__init__()

        self.head_config = config.pretrain.head
        self.globals = config.globals
        self.num_steps = self.globals.num_steps

        self.structure_head = StructureHead(
            **self.head_config
        )

        self.trans_rot = RotationTransition(self.num_steps)
        self.trans_pos = PositionTransition(self.num_steps)
        self.register_buffer('_dummy', torch.empty([0, ]))
    
    def iteration(self, batch, s_init, z_init, rigids_init, _recycle=False):
        s, z, rigids = s_init, z_init, rigids_init
        outputs = {}
        seq_mask = batch["decoy_seq_mask"]


        outputs["sm"] = self.structure_head(
            s,
            z,
            rigids,
            seq_mask,
        )

        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        return outputs
        
    def forward(self, batch, s, z, time_step=None, denoise_strcuture=True):
        # Prep some features
        seq_mask = batch["decoy_seq_mask"]
        # embedding
        # s: shape: [Batch, L, dim]
        # z: shape: [Batch, L, L, dim]
        rigids = Rigid.from_tensor_4x4(batch["bb_rigid_tensors"])
        # For Diffusion to get out of the local minimum
        R_0 = rigids.get_rots().get_rot_mats()
        p_0 = rigids.get_trans()
        v_0 = rotation_to_so3vec(R_0)
        N, L = s.shape[:2]
        if time_step == None:
            time_step = torch.randint(0, self.num_steps, (N,), dtype=torch.long, device=self._dummy.device)
        if denoise_strcuture:
            v_noisy, _ = self.trans_rot.add_noise(v_0, seq_mask.bool(), time_step)
            p_noisy, _ = self.trans_pos.add_noise(p_0, seq_mask.bool(), time_step)
            R_noisy = so3vec_to_rotation(v_noisy) # Rotation tensor in shape: [N,L,3,3]
            rigids = Rigid(Rotation(R_noisy), p_noisy)

            outputs = self.iteration(
                batch,
                s_init = s,
                z_init = z,
                rigids_init = rigids,
            )
        # pdb.set_trace()
        return outputs