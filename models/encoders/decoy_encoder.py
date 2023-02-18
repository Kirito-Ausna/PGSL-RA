import sys
sys.path.append("../../data")
import torch
import torch.nn as nn
import torch.nn.functional as F
from embedder import (
    DecoyAngleEmbedder,
    DecoyPairEmbedder,
    AtomEmbLayer,
    DecoyPairStack,
    DecoyPointwiseAttention,
    OuterProductMean,
)

from data.feature_pipeline import (
    build_decoy_angle_feats,
    build_decoy_pair_feats,
)


class DecoyEncoder(nn.Module):
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
        # self.num_steps = self.globals.num_steps

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
        self.OutProductMean = OuterProductMean(
              **decoy_config["outer_product_mean"]
        )
    def forward(self, batch, pair_mask=None, seq_mask=None):
        """
        Args:
            decoy_batch:
                A dict-like object that contains the following keys:
                - "decoy_atom": [B, N, 37]
                - "decoy_atom_mask": [B, N]
                - "decoy_angle": [B, N, 6]
                - "decoy_angle_mask": [B, N]
                - "decoy_pair": [B, N, N, 6]
                - "decoy_pair_mask": [B, N, N]
                - "decoy_seq": [B, N]
                - "decoy_seq_mask": [B, N]
                - "decoy_seq_len": [B]
                - "decoy_seq_len_mask": [B]
                - "decoy_atom_mask": [B, N]
                - "decoy_angle_mask": [B, N]
                - "decoy_pair_mask": [B, N, N]
        """
        decoy_embeds = {}
        batch = build_decoy_angle_feats(batch)
        decoy_pair_feats = build_decoy_pair_feats(
                batch,
                inf=self.data_config.inf,
                eps=self.data_config.eps,
                **self.data_config.distogram
            )
        decoy_angle_feats = self.decoy_atom_embedder(batch)
        a = self.decoy_angle_embedder(decoy_angle_feats)
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
        t = d # For the sake of recyclying
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
