import sys
sys.path.append("../../data")
sys.path.append("../../utils")
import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedder import (
    DecoyAngleEmbedder
)
from models.sublayers import utils
from utils.residue_constants import (
    restypes_with_x
)
import utils.residue_constants as rc

from data.feature_pipeline import (
    build_unimol_angle_feats,
    build_unimol_pair_feats,
)
import pdb

@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x

class GaussianLayer(nn.Module):
    def __init__(self, num_distance=25, K=16, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, num_distance*self.K)  # 16 * 25 = 400, it's the total number of kernels
        self.stds = nn.Embedding(1, num_distance*self.K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        # pdb.set_trace()
        # x = mul * x.unsqueeze(-1) + bias # [B, N, N, 25, 1]
        x = mul * x + bias # [B, N, N, 25]
        x = x.unsqueeze(-1) # [B, N, N, 25, 1]
        x = x.expand(-1, -1, -1, -1, self.K) # [B, N, N, 25, K]
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1) # [B, N, N, 25*K]
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)
    
class GaussianEncoder(nn.Module):
    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)

        """
        super().__init__()

        # self.encoder_config = config.model.encoder
        # self.data_config = config.data.decoy
        self.embedder_config = config.model.embedder

        self.decoy_angle_embedder = DecoyAngleEmbedder(
            **self.embedder_config["protein_angle_embedder"]
        )
        num_distance = self.embedder_config["gaussian_layer"]["num_pair_distance"]
        K = self.embedder_config["gaussian_layer"]["kernel_num"]
        n_edge_types = len(restypes_with_x) * len(restypes_with_x)
        self.gbf = GaussianLayer(num_distance, K, n_edge_types)
        self.gbf_proj = NonLinearHead(
            **self.embedder_config["bias_proj_layer"]
        )
        self.centrality_proj = NonLinearHead(
            **self.embedder_config["centrality_proj_layer"]
        )
    
    def get_encoding_features(self, dist, et, pair_mask=None, get_bias=True):
        n_node = dist.size(-2)
        gbf_feature = self.gbf(dist, et)

        if pair_mask is not None:
            centrality_encoding = gbf_feature * pair_mask.unsqueeze(-1)
        else:
            centrality_encoding = gbf_feature # [B, N, N, 25*K]
        centrality_encoding = self.centrality_proj(centrality_encoding.sum(dim=-2)) # [B, N, encoder_embed_dim]

        gbf_result = self.gbf_proj(gbf_feature)
        graph_attn_bias = gbf_result # [B, N, N, num_head]
        # pdb.set_trace()
        # sum over the length dimension (-3)
        if get_bias:
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous() # [B, num_head, N, N]
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node) # [B*num_head, N, N]
        return graph_attn_bias, centrality_encoding
    
    def forward(self, batch, pair_mask=None, get_bias=True):
        """
        Args:
            decoy_batch:
                A dict-like object that contains the following keys:
        """
        # batch = build_unimol_angle_feats(batch)
        # pdb.set_trace()
        x = self.decoy_angle_embedder(batch["decoy_angle_feats"])
        # pdb.set_trace()
        dist, et = build_unimol_pair_feats(batch, ca_only=False, pair_mask=pair_mask)
        # dist, et = batch["dist"], batch["edge_type"]
        graph_attn_bias, centrality_encoding = self.get_encoding_features(dist, et, pair_mask=pair_mask, get_bias=get_bias)
        x = x + centrality_encoding

        return x, graph_attn_bias


        
