import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from openfold.model.primitives import Linear, LayerNorm, ipa_point_weights_init_
from utils.rigid_utils import Rigid, Rotation
from utils.tensor_utils import (
    dict_multimap,
)
from models.backbone.structure_module import (
    InvariantPointAttention,
    BackboneUpdate,
    StructureModuleTransition
)

class StructureHead(nn.Module):
    def __init__(
            self,
            no_blocks,
            c_s,
            c_z,
            c_ipa,
            no_heads_ipa,
            no_qk_points,
            no_v_points,
            dropout_rate,
            no_transition_layers,
            trans_scale_factor,
            epsilon,
            inf,
            ):
        super().__init__()
        self.no_blocks = no_blocks
        self.c_s = c_s
        self.c_z = c_z
        self.c_ipa = c_ipa
        self.no_heads_ipa = no_heads_ipa
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.dropout_rate = dropout_rate
        self.no_transition_layers = no_transition_layers
        self.trans_scale_factor = trans_scale_factor
        self.epsilon = epsilon
        self.inf = inf

        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)

        self.linear_in = Linear(self.c_s, self.c_s)
        self.ipa = InvariantPointAttention(
            self.c_s,
            self.c_z,
            self.c_ipa,
            self.no_heads_ipa,
            self.no_qk_points,
            self.no_v_points,
            inf=self.inf,
            eps=self.epsilon,
        )
        self.layer_norm_ipa = LayerNorm(self.c_s)
        self.layer_norm_trans = LayerNorm(self.c_s)
        self.transition_layer = StructureModuleTransition(
                                        self.c_s,
                                        self.no_transition_layers,
                                        self.dropout_rate
                                    )
        self.backbone_update = BackboneUpdate(self.c_s)
        self.ipa_dropout = nn.Dropout(self.dropout_rate)
    
    def forward(
            self,
            s,
            z,
            rigids,
            mask=None
    ):
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        if mask is None:
            # [*, N]
            mask = s.new_ones(s.shape[:-1])

        # [*, N, C_s]
        s = self.layer_norm_s(s)

        # [*, N, N, C_z]
        z = self.layer_norm_z(z)

        # [*, N, C_s]
        s = self.linear_in(s)
        
        outputs = []
        rigids = rigids.scale_translation(1/self.trans_scale_factor)
        for i in range(self.no_blocks):
            s = s + self.ipa(s, z, rigids, mask)
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = s + self.transition_layer(s)
            s = self.ipa_dropout(s)
            s = self.layer_norm_trans(s)

            rigids = rigids.compose_q_update_vec(self.backbone_update(s))
            scaled_rigids = rigids.scale_translation(self.trans_scale_factor)

            preds = {
                "frames": scaled_rigids.to_tensor_7()
            }
            outputs.append(preds)

            if i < (self.no_blocks - 1):
                rigids = rigids.stop_rot_gradient()
    
        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s

        return outputs

        
        




