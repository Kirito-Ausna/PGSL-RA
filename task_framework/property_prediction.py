import pdb

import torch
import torch.nn as nn
from torchdrug import layers


class MultiBinaryClassifyHead(nn.Module):
    """
    Multiple binary classification task for graphs / molecules / proteins.

    Parameters:
        
    """
    def __init__(self, task_num, num_mlp_layers=3,
                 model_out_dim=384):
        super().__init__()
        # self.model = model
        self.task_num = task_num
        self.num_mlp_layers = num_mlp_layers
        self.model_out_dim = model_out_dim
        # pdb.set_trace()
        hidden_dims = [self.model_out_dim] * (self.num_mlp_layers -1)

        self.mlp = layers.MLP(self.model_out_dim, hidden_dims + [task_num])

    def forward(self, batch, node_repr):
        # Shape: node_repr [B, N, c_s]
        # seq_mask [B,N]
        # return 111
        seq_mask = batch["decoy_seq_mask"]
        # node_repr = self.model(batch)
        # pdb.set_trace()
        node_repr = node_repr * seq_mask.unsqueeze(-1)
        # [B,c_s]
        # graph_repr = torch.sum(node_repr, dim=-2)
        # pdb.set_trace()
        graph_repr = torch.sum(node_repr, dim=-2) / torch.sum(seq_mask, dim=-1, keepdim=True)
        pred = self.mlp(graph_repr) # [B, tasks] with logits

        return pred




