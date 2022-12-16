import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import pdb

from models.encoders.embedder import (
    DecoyAngleEmbedder,
    DecoyPairEmbedder,
    AtomEmbLayer,
    DecoyPairStack,
    DecoyPointwiseAttention,
    InputEmbedder,
    OuterProductMean,
)
from data.feature_pipeline import (
    build_decoy_angle_feats,
    build_decoy_pair_feats,
)

from models.backbone.constrformer import ConstrainFormer
from models.backbone.ipaformer import Ipaformer
from data.data_transform import atom14_to_atom37, atom37_to_rigids
from models._base import register_model
# from openfold.utils.rigid_utils import Rigid, Rotation
# from utils import residue_constants
@register_model("alpha_encoder")
class AlphaEncoder(nn.Module):
    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)

        """
        super().__init__()

        self.config = config.model
        self.data_config = config.data.decoy
        decoy_config = self.config.decoy
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
        self.structure_module = Ipaformer(
            **self.config["structure_module"]
        )
        self.esm_weights = nn.Parameter(torch.ones(1,4,1,1), requires_grad=True)


    def embed_decoy(self, batch, pair_mask=None, seq_mask=None):
        decoy_embeds = {}
        # ESM Embedding Here
        esm_embedding = batch["esm_emb"] # [B,4,N,5120]
        # pdb.set_trace()
        try:
            s = torch.sum(F.softmax(self.esm_weights) * esm_embedding, dim=1) # [B,4,N,5120]
        except RuntimeError as e:
            print(e)
            pdb.set_trace()
        s = self.input_embedder(s,seq_mask)
        #updated embedding
        # decoy_angle_feats = batch["decoy_angle_feats"]
        batch = build_decoy_angle_feats(batch)
        decoy_pair_feats = build_decoy_pair_feats(
                batch,
                inf=self.data_config.inf,
                eps=self.data_config.eps,
                **self.data_config.distogram
            )
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
    def forward(self, batch):
        seq_mask = batch["decoy_seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        # embedding
        decoy_embedding, t = self.embed_decoy(batch, pair_mask, seq_mask)
        # s: shape: [Batch, L, dim]
        s = decoy_embedding["decoy_angle_embedding"]
        z = decoy_embedding["decoy_pair_embedding"]
        rigids = self.decoy2rigids(batch)
        s, z = self.costrformer(
            s[:,None,...],
            z,
            seq_mask[:,None,...],
            pair_mask,
        )

        s = self.structure_module(
            s.squeeze(1),
            z,
            rigids,
            mask=seq_mask
        )


        return s