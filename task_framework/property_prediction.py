import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import pdb
from torchdrug import layers
from openfold.utils.rigid_utils import Rigid, Rotation
from utils import residue_constants

class MultipleBinaryClassification(nn.Module):
    """
    Multiple binary classification task for graphs / molecules / proteins.

    Parameters:
        
    """
    def __init__(self, model, task_num, num_mlp_layers=3,
                 model_out_dim=384):
        super().__init__()
        self.model = model
        self.task_num = task_num
        self.num_mlp_layers = num_mlp_layers
        self.model_out_dim = model_out_dim
        # pdb.set_trace()
        hidden_dims = [self.model_out_dim] * (self.num_mlp_layers -1)

        self.mlp = layers.MLP(self.model_out_dim, hidden_dims + [task_num])

    def forward(self, batch):
        # Shape: node_repr [B, N, c_s]
        # seq_mask [B,N]
        # return 111
        seq_mask = batch["decoy_seq_mask"]
        node_repr = self.model(batch)
        # pdb.set_trace()
        node_repr = node_repr * seq_mask.unsqueeze(-1)
        # [B,c_s] #TODO: we need to modify the padding/trunc and mask system to allow batch_size > 1
        graph_repr = torch.sum(node_repr, dim=-2)
        pred = self.mlp(graph_repr) # [B, tasks] with logits

        return pred




