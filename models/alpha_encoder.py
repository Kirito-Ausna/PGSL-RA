import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import pdb

# from models.encoders.embedder import (
#     DecoyAngleEmbedder,
#     DecoyPairEmbedder,
#     AtomEmbLayer,
#     DecoyPairStack,
#     DecoyPointwiseAttention,
#     InputEmbedder,
#     OuterProductMean,
# )
# from data.feature_pipeline import (
#     build_decoy_angle_feats,
#     build_decoy_pair_feats,
# )
# from models.encoders.decoy_encoder import DecoyEncoder
from models.encoders.gaussian_encoder import GaussianEncoder
from models.backbone.constrformer import ConstrainFormer
from models.backbone.ipaformer import Ipaformer
from data.data_transform import atom14_to_atom37, atom37_to_rigids
from models._base import register_model
from utils.rigid_utils import Rigid

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
        self.embed_gaussian = GaussianEncoder(config)
        self.costrformer = ConstrainFormer(
            **self.config["constrainformer"]
        )
        self.structure_module = Ipaformer(
            **self.config["structure_module"]
        )
        # self.esm_weights = nn.Parameter(torch.ones(1,4,1,1), requires_grad=True)

    def forward(self, batch):
        seq_mask = batch["decoy_seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        # embedding
        # decoy_embedding, _ = self.embed_decoy(batch, pair_mask, seq_mask)
        # s: shape: [Batch, L, dim]
        # s = decoy_embedding["decoy_angle_embedding"]
        # z = decoy_embedding["decoy_pair_embedding"]
        s, z = self.embed_gaussian(batch, pair_mask, get_bias=False)
        # rigids = self.decoy2rigids(batch)
        rigids = Rigid.from_tensor_4x4(batch["bb_rigid_tensors"])
        s, z = self.costrformer(
            s[:,None,...],
            z,
            seq_mask[:,None,...],
            pair_mask,
        ) # Probably good but too costly, powerful yet difficult to train

        s = self.structure_module(
            s.squeeze(1),
            z,
            rigids,
            mask=seq_mask
        ) # Z is fixed in this manner. It's okay when using powerful inizialization like esm embedding, but not good when directly learning from structure 


        return s