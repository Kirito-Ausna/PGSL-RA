import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.encoders.gaussian_encoder import GaussianEncoder
from models.backbone.graphformer import TransformerEncoderWithPair
from models._base import register_model

@register_model("uni_encoder")
class UniEncoder(nn.Module):
    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super().__init__()
        self.config = config.model
        self.protein_embedder = GaussianEncoder(config)
        self.encoder = TransformerEncoderWithPair(
            **self.config["graphformer"]
        )
    
    def forward(self, batch):

        seq_mask = batch["decoy_seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        padding_mask = 1 - seq_mask
        # embed protein
        x, graph_attn_bias = self.protein_embedder(batch, pair_mask)
        (
            encoder_rep, 
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("inf")] = 0

        return encoder_rep
        
