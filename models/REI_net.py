import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.gaussian_encoder import GaussianEncoder
from models.backbone.graphformer import TransformerEncoderWithPair
from models.backbone.ipaformer import Ipaformer
from models._base import register_model
from data.data_transform import atom37_to_rigids
from utils.rigid_utils import Rigid

@register_model("REI_net")
class REINet(nn.Module):
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
        self.ipaformer = Ipaformer(
            **self.config["ipaformer"]
        )
    
    # def decoy2rigid(self, batch):
    #     return atom37_to_rigids(batch["decoy_aatype"], 
    #                             batch["decoy_all_atom_positions"], batch["decoy_all_atom_mask"])

    def forward(self, batch, pretrain=False):

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
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias, pair_mask=pair_mask)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0
        # rigids = self.decoy2rigid(batch)
        rigids = Rigid.from_tensor_4x4(batch["bb_rigid_tensors"])
        encoder_rep = self.ipaformer(
            encoder_rep,
            encoder_pair_rep,
            rigids,
            mask=seq_mask
        )# encoder_pair_rep is fixed in this manner. It's okay when using powerful inizialization like esm embedding, 
         # but not good when directly learning from structure 
        # encoder_rep = self.ipaformer(
        #     x,
        #     graph_attn_bias,
        #     rigids,
        #     mask=seq_mask
        # )
        if pretrain:
            return encoder_rep, encoder_pair_rep
        return encoder_rep
        
