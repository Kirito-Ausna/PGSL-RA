import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.gaussian_encoder import GaussianEncoder
# from models.backbone.graphformer import TransformerEncoderWithPair
from models.backbone.ipaformer import Ipaformer
from models._base import register_model
# from data.data_transform import atom37_to_rigids
from utils.rigid_utils import Rigid

@register_model("ipa_encoder")
class IpaEncoder(nn.Module):
    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super().__init__()
        self.config = config.model
        self.protein_embedder = GaussianEncoder(config)
        self.encoder = Ipaformer(
            **self.config["ipaformer"]
        )
        # self.ipaformer = Ipaformer(
        #     **self.config["ipaformer"]
        # )
    
    def forward(self, batch):

        seq_mask = batch["decoy_seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        # padding_mask = 1 - seq_mask
        # embed protein
        rigids = Rigid.from_tensor_4x4(batch["bb_rigid_tensors"])
        x, graph_attn_bias = self.protein_embedder(batch, pair_mask, get_bias=False)
        # encoder_rep = self.ipaformer(
        #     x,
        #     graph_attn_bias,
        #     rigids,
        #     mask=seq_mask
        # )
        encoder_rep = self.encoder(
            x,
            graph_attn_bias,
            rigids,
            mask=seq_mask
        )
        return encoder_rep