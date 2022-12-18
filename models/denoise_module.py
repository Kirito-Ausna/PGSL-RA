import pdb
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.data_transform import (atom14_to_atom37, atom37_to_rigids,
                                 pseudo_beta_fn)
from data.feature_pipeline import (build_decoy_angle_feats,
                                   build_decoy_pair_feats)
from models._base import register_model
from models.backbone.constrformer import ConstrainFormer
from models.backbone.structure_module import StructureModule
from models.encoders.embedder import (AtomEmbLayer, DecoyAngleEmbedder,
                                      DecoyPairEmbedder, DecoyPairStack,
                                      DecoyPointwiseAttention, InputEmbedder,
                                      OuterProductMean, RecyclingEmbedder)
from modules.common.so3 import rotation_to_so3vec, so3vec_to_rotation
from modules.diffusion.transition import PositionTransition, RotationTransition
from openfold.utils.rigid_utils import Rigid, Rotation
from utils import residue_constants


@register_model("denoise_module")
class DenoiseModule(nn.Module):
    """
    A general framework of Protein Structure Denoising Module(PSDM) 
    for Protein Structure Prediction/Protein Structure Refinement, and the most importantly Protein Structure Generation.

    We want to provide a general PSDM framework that's suitable for various generative framework such as DDPM or VAE.
    """
    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)

        """
        super(DenoiseModule, self).__init__()

        self.config = config.model
        decoy_config = self.config.decoy
        # self.data_config = config.data
        self.data_config = config.data.decoy
        self.globals = config.globals
        self.num_steps = self.globals.num_steps

        self.decoy_angle_embedder = DecoyAngleEmbedder(
            **decoy_config["decoy_angle_embedder"]
        )
        self.decoy_pair_embedder = DecoyPairEmbedder(
            **decoy_config["decoy_pair_embedder"]
        )
        self.decoy_atom_embedder = AtomEmbLayer(
            **decoy_config["decoy_atom_embedder"]
        )
        self.decoy_pair_stack = DecoyPairStack(
            **decoy_config["decoy_pair_stack"]
        )
        self.decoy_pointwise_att = DecoyPointwiseAttention(
            **decoy_config["decoy_pointwise_attention"]
        )
        # self.decoy_seq_stack = DecoySeqStack(
        #     **decoy_config["decoy_seq_stack"]
        # )
        self.input_embedder = InputEmbedder(
            **self.config["input_embedder"]
        )
        self.OuterProductMean = OuterProductMean(
           **self.config["outer_product_mean"]
        )
        self.costrformer = ConstrainFormer(
            **self.config["constrainformer"]
        )
        self.structure_module = StructureModule(
            **self.config["structure_module"]
        )
        self.recycling_embedder = RecyclingEmbedder(
            **self.config["recycling_embedder"]
        )
        self.esm_weights = nn.Parameter(torch.ones(1,4,1,1), requires_grad=True)
        # init_weights = torch.ones(4)
        # self.esm_weights.data.fill_(init_weights)
        self.trans_rot = RotationTransition(self.num_steps)
        self.trans_pos = PositionTransition(self.num_steps)
        self.register_buffer('_dummy', torch.empty([0, ]))


    def embed_decoy(self, batch, pair_mask=None, seq_mask=None):
        decoy_embeds = {}
        # ESM Embedding Here
        esm_embedding = batch["esm_emb"] # [B,4,N,5120]
        # pdb.set_trace()
        s = torch.sum(F.softmax(self.esm_weights) * esm_embedding, dim=1) # [B,N,5120]
        s = self.input_embedder(s,seq_mask)
        # pdb.set_trace()
        batch = build_decoy_angle_feats(batch)
        decoy_pair_feats = build_decoy_pair_feats(
                batch,
                inf=self.data_config.inf,
                eps=self.data_config.eps,
                **self.data_config.distogram
            )
        #updated embedding
        decoy_angle_feats = self.decoy_atom_embedder(batch)
        a = self.decoy_angle_embedder(decoy_angle_feats) + s
        # a = self.decoy_seq_stack(a)
        decoy_embeds["angle"] = a
        # decoy_pair_feats = batch["decoy_pair_feats"]
        z = self.OuterProductMean(a[:,None,...], seq_mask[:,None,...])
        d = self.decoy_pair_embedder(decoy_pair_feats)
        decoy_embeds["pair"] = d
        d = self.decoy_pair_stack(
            decoy_embeds["pair"].unsqueeze(-4),
            pair_mask.unsqueeze(-3),
            chunk_size=self.globals.chunk_size,
            _mask_trans=self.config._mask_trans,
        )
        t = d
        # pdb.set_trace()
        d = self.decoy_pointwise_att(
            d,
            # torch.cat((d[:,0,...], d[:,0,...]), dim=-1),
            z,
            chunk_size=self.globals.chunk_size,
        )
        ret = {}
        ret["decoy_angle_embedding"] = decoy_embeds["angle"]
        ret["decoy_pair_embedding"] = d

        return ret, t
    def decoy2rigids(self, batch):
        return atom37_to_rigids(batch["decoy_aatype"], batch["decoy_all_atom_positions"], batch["decoy_all_atom_mask"])
    
    def iteration(self, batch, s_init, z_init, t, rigids_init, s_prev, z_prev, x_prev, _recycle=True):
        outputs = {}
        batch_dims = s_init.shape[0]
        n = s_init.shape[1]
        seq_mask = batch["decoy_seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]

        s = s_init
        z = z_init
        # pdb.set_trace()
        if _recycle:
            # Initialize the recycling embeddings, if needs be
            if None in [s_prev, z_prev, x_prev]:
                # [*, N, C_m]
                s_prev = torch.zeros_like(
                    s_init,
                    requires_grad = False
                )# the first row of msa embedding

                # [*, N, N, C_z]
                z_prev = torch.zeros_like(
                    z_init,
                    requires_grad = False,
                )
                # [*, N, 37, 3]
                x_prev = z.new_zeros(
                    (batch_dims, n, residue_constants.atom_type_num, 3),
                    requires_grad = False,
                )
            x_prev = pseudo_beta_fn(
                batch["decoy_aatype"], x_prev, None
                ).to(dtype=z_init.dtype)
            s_prev_emb, z_prev_emb = self.recycling_embedder(
                s_prev,
                z_prev,
                x_prev,
            )
            s = s + s_prev_emb
            z = z + z_prev_emb

            # template
            # pdb.set_trace()
            z = self.decoy_pointwise_att(
                t,
                z,
                chunk_size=self.globals.chunk_size
            )


            del s_prev_emb, z_prev_emb
        # pdb.set_trace()
        s, z = self.costrformer(
            s[:,None,...],
            z,
            seq_mask[:,None,...],
            pair_mask,
        )
        # s: [b,n_seq,n_res,dim]
        outputs["sm"] = self.structure_module(
            s.squeeze(1),
            z,
            batch["decoy_aatype"],
            rigids_init,
            mask=batch["decoy_seq_mask"]
        )

        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], batch
        )
        outputs["final_atom_mask"] = batch["decoy_atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        s_prev = s.squeeze()
        z_prev = z
        x_prev = outputs["final_atom_positions"]
        return outputs, s_prev, z_prev, x_prev
    
    def forward(self, batch, time_step=None, denoise_strcuture=True):
        # pdb.set_trace()
        # Prep some features
        # pdb.set_trace()
        seq_mask = batch["decoy_seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        # embedding
        decoy_embedding, t = self.embed_decoy(batch, pair_mask, seq_mask)
        # s: shape: [Batch, L, dim]
        s = decoy_embedding["decoy_angle_embedding"]
        z = decoy_embedding["decoy_pair_embedding"]
        # pdb.set_trace()
        s_prev, z_prev, x_prev= None, None, None
        rigids = self.decoy2rigids(batch)
        # For Diffusion to get out of the local minimum
        R_0 = rigids.get_rots().get_rot_mats()
        p_0 = rigids.get_trans()
        v_0 = rotation_to_so3vec(R_0)
        N, L = s.shape[:2]
        if time_step == None:
            time_step = torch.randint(0, self.num_steps, (N,), dtype=torch.long, device=self._dummy.device)
        if denoise_strcuture:
            v_noisy, _ = self.trans_rot.add_noise(v_0, seq_mask.bool(), time_step)
            p_noisy, eps_p = self.trans_pos.add_noise(p_0, seq_mask.bool(), time_step)
            R_noisy = so3vec_to_rotation(v_noisy) # Rotation tensor in shape: [N,L,3,3]
            rigids = Rigid(Rotation(R_noisy), p_noisy)

        is_grad_enabled = torch.is_grad_enabled()
        ## Recycling
        num_iters = self.globals.max_recycling_iters + 1
        for cycle_no in range(num_iters):
            is_final_iter = cycle_no == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()
                outputs, s_prev, z_prev, x_prev = self.iteration(
                    batch,
                    s_init = s,
                    z_init = z,
                    t = t,
                    rigids_init = rigids,
                    s_prev = s_prev, 
                    z_prev = z_prev, 
                    x_prev = x_prev,
                    _recycle = True,
                    )
        # pdb.set_trace()
        return outputs