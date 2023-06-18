import copy
import pdb
import random
from functools import wraps

import torch
from torch import nn
from torch.nn import functional as F

from task_framework.PGSL_vanilla import denoise_head
from utils.loss import backbone_loss

# helper functions

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def pooling(seq_mask, node_repr):
    seq_mask = seq_mask.unsqueeze(-1)
    node_repr = node_repr * seq_mask
    graph_repr = torch.sum(node_repr, dim=-2) / torch.sum(seq_mask, dim=-2)
    return graph_repr

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# split a the feacture dict batch into two parts
def split_batch(batch):
    decoy_batch = {}
    exp_batch = {}
    for key, value in batch.items():
        if key.startswith('decoy'):
            decoy_batch[key] = value
        elif key.startswith('exp'):
            exp_batch[key.replace('exp', 'decoy')] = value

    decoy_batch["bb_rigid_tensors"] = batch["bb_rigid_tensors"]
    exp_batch["bb_rigid_tensors"] = batch["label_bb_rigid_tensors"]
    exp_batch["decoy_seq_mask"] = batch["decoy_seq_mask"]
    exp_batch["decoy_aatype"] = batch["decoy_aatype"]

    return decoy_batch, exp_batch
    # for key, value in batch.items():
    #     batch1[key] = value[:split_size]
    #     batch2[key] = value[split_size:]
    # return batch1, batch2
# loss fn

def align_loss(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    align_loss = 2 - 2 * (x * y).sum(dim=-1)
    return align_loss.mean()


# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
    
def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

def MLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )
# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter nets
class NetWrapper(nn.Module):
    def __init__(self, net, dim, projection_size, projection_hidden_size, use_simsiam_mlp = False):
        super().__init__()
        self.net = net
        # self.layer = layer

        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        self.dim = dim
        self.use_simsiam_mlp = use_simsiam_mlp
        self.projector = self._get_projector()

    def _get_projector(self):
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(self.dim, self.projection_size, self.projection_hidden_size)
        return projector

    def forward(self, batch, return_projection = True):
        struct_repr = self.net(batch, pretrain=return_projection)
        if not return_projection:
            representation = pooling(batch["decoy_seq_mask"], struct_repr)
            return representation

        # pdb.set_trace()
        representation = struct_repr[0]
        representation = pooling(batch["decoy_seq_mask"], representation)
        projection = self.projector(representation)
        return projection, struct_repr
    
class SAO(nn.Module):
    def __init__(self,
                 encoder,
                 config,
                 ):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.denoise_head = denoise_head(config)
        self.framework = config.pretrain.framework
        #TODO: add mask structure modeling head
        self.use_momentum = self.framework.SAO.use_momentum
        self.moving_average_decay = self.framework.SAO.moving_average_decay
        self.online_encoder = NetWrapper(encoder, **self.framework.projector)
        if self.use_momentum:
            self.target_encoder = copy.deepcopy(self.online_encoder)
            set_requires_grad(self.target_encoder, False)
            self.target_ema_updater = EMA(self.moving_average_decay)
        else:
            self.target_encoder = self.online_encoder # Siamese network
        self.online_predictor = MLP(**self.framework.predictor)

        # get device of network and make wrapper same device
        device = get_module_device(encoder)
        self.to(device)
    
    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def get_pred_embeddings(self, pred_struct, return_projection=True):
        
        pred_struct_proj, pred_struct_emb = self.online_encoder(pred_struct, return_projection = return_projection)
        pred_struct_pred = self.online_predictor(pred_struct_proj)

        return pred_struct_pred, pred_struct_emb
    
    def denoise_struct(self, batch, pred_struct_emb):
        s, z = pred_struct_emb
        outputs = self.denoise_head(batch, s, z)

        return outputs
    
    def get_exp_embeddings(self, exp_struct, return_projection=True):

        with torch.no_grad():
            exp_struct_proj, _ = self.target_encoder(exp_struct, return_projection = return_projection)
            exp_struct_proj.detach_()
        
        return exp_struct_proj

    def SAO_loss(self, outputs, batch, return_breakdown):
        loss_fns = {
            "fape": lambda: backbone_loss(
                label_backbone_rigid_tensor=batch["label_bb_rigid_tensors"],
                label_backbone_rigid_mask=batch["decoy_seq_mask"],
                traj=outputs["sm"]["frames"],
                **self.config.loss.fape,
            ),
            "pred_align_loss": lambda: align_loss(
                outputs["pred_struct_pred"], outputs["exp_struct_proj"]
            ), #TODO: add mask_align_loss and mlm loss -->ignore_index = -1
        }

        cum_loss = 0
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            # pdb.set_trace()
            weight = self.config.loss[loss_name].weight
            loss = loss_fn()
            # pdb.set_trace()
            if(torch.isnan(loss) or torch.isinf(loss)):
                # logging.warning(f"{loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0., requires_grad=True)
            cum_loss = cum_loss + weight * loss
            losses[loss_name] = loss.detach().clone()
        
        losses["SAO_loss"] = cum_loss

        if (not return_breakdown):
            return cum_loss
        
        return (cum_loss, losses)
    
    def forward(self, 
                batch, 
                return_embedding = False,
                return_projection = True,
                return_breakdown = False):
        assert not (self.training and batch['decoy_aatype'].shape == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(batch, return_projection = return_projection)
        
        # Split the three views
        pred_struct, exp_struct = split_batch(batch)
        # Go through different pipelines
        pred_struct_pred, pred_struct_emb = self.get_pred_embeddings(pred_struct, return_projection = return_projection)
        exp_struct_proj = self.get_exp_embeddings(exp_struct, return_projection = return_projection)
        outputs = self.denoise_struct(batch, pred_struct_emb)

        outputs["pred_struct_pred"] = pred_struct_pred
        outputs["exp_struct_proj"] = exp_struct_proj

        loss = self.SAO_loss(outputs, batch, return_breakdown)

        return loss, outputs