# from gvp.atom3d import BaseModel
# from openfold.np import residue_constants
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from gvp import GVP, GVPConvLayer, LayerNorm
# import pdb

# _DEFAULT_V_DIM = (100, 16)
# _DEFAULT_E_DIM = (32, 1)
# _NUM_ATOM_TYPES = residue_constants.atom_type_num
# class RefineModel(BaseModel):
#     '''
#     GVP-GNN for full atom protein refinement
#     Extends BaseModel to get node embedding as tuple of (S, V)
#     V for atom coordinates as shape (1,3) 
#     '''
#     def __init__(self, num_rbf=16):
#         super().__init__(num_rbf)
#         activations = (F.relu, None)
#         self.embed = nn.Embedding(residue_constants.atom_type_num, residue_constants.atom_type_num)

#         self.W_v = nn.Sequential(
#             LayerNorm((_NUM_ATOM_TYPES, 1)),
#             GVP((_NUM_ATOM_TYPES, 1), _DEFAULT_V_DIM,
#                 activations=(None, None), vector_gate=True)
#         )

#         ns, nv = _DEFAULT_V_DIM
#         self.W_out = nn.Sequential(
#             LayerNorm(_DEFAULT_V_DIM),
#             GVP(_DEFAULT_V_DIM, _DEFAULT_V_DIM, 
#                 activations=activations, vector_gate=True),
#             LayerNorm(_DEFAULT_V_DIM),
#             GVP(_DEFAULT_V_DIM, (2*ns, 2*nv), 
#                 activations=activations, vector_gate=True),
#             LayerNorm((2*ns,2*nv)),
#             GVP((2*ns,2*nv), (1, 1), 
#                 activations=activations, vector_gate=True)
#         )

#     def forward(self, batch):
#         h_V = (self.embed(batch.atoms), batch.x)
#         h_E = (batch.edge_s, batch.edge_v)
#         # pdb.set_trace()
#         h_V = self.W_v(h_V)
#         h_E = self.W_e(h_E)

#         batch_id = batch.batch
#         # torch.use_deterministic_algorithms(False)
#         for layer in self.layers:
#             h_V = layer(h_V, batch.edge_index, h_E)
#         # torch.use_deterministic_algorithms(True)
        
#         # shape out[0]:(N_atoms, 1) out[1]: (N_atom, 1, 3)
#         out = self.W_out(h_V)
#         return out[1]


        