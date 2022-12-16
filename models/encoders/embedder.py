import re
from turtle import forward
import torch
import torch.nn as nn
from typing import Tuple
import sys
import pdb
# sys.path.append("..")
from openfold.model.primitives import Linear, Attention, LayerNorm
from openfold.utils.tensor_utils import (
    one_hot, 
    chunk_layer, 
    permute_final_dims
)
from openfold.utils.checkpointing import checkpoint_blocks
from functools import partialmethod, partial
from typing import Optional, List, Union
from ..sublayers.triangular_attention import (
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
)
from ..sublayers.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing
)

class DecoyAngleEmbedder(nn.Module):
    """
    Embeds the "template_angle_feat" feature.

    Implements Algorithm 2, line 7.
    """

    def __init__(
        self,
        c_in: int,
        c_m: int,
        c_out: int,
        **kwargs,
    ):
        """
        Args:
            c_in:
                Final dimension of "template_angle_feat"
            c_out:
                Output channel dimension
        """
        super(DecoyAngleEmbedder, self).__init__()

        self.c_out = c_out
        self.c_m = c_m
        self.c_in = c_in

        self.linear_1 = Linear(self.c_in, self.c_out, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.c_out, self.c_out, init="relu")
        # self.linear_3 = Linear(self.c_m, self.c_out, init="relu")
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, N_templ, N_res, c_in] "template_angle_feat" features
        Returns:
            x: [*, N_templ, N_res, C_out] embedding
        """
        # x_identity = x
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        # x = self.relu(x)
        # x = self.linear_3(x)

        return x

class InputEmbedder(nn.Module):
    """
    Feed-forward network applied to ESM activations before attention.
    Implements Algorithm 9
    """
    def __init__(
        self, 
        c_in: int,
        c_hidden: int,
        c_out: int
    ):
        """
        c_in:
                Final dimension of "template_angle_feat"
        c_out:
                Output channel dimension
        """
        super(InputEmbedder, self).__init__()

        self.c_in = c_in
        self.c_m = c_hidden
        self.c_out = c_out

        self.layer_norm = LayerNorm(self.c_in)
        self.linear_1 = Linear(self.c_in, self.c_m, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.c_m, self.c_out, init="final")

    def _transition(self, m, mask):
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m) * mask
        return m

    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_m] Single Sequence activation
            mask:
                [*, N_res] MSA mask
        Returns:
            s:
                [*, N_res, C_m] Sequence activation update
        """
        # DISCREPANCY: DeepMind forgets to apply the MSA mask here.
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        mask = mask.unsqueeze(-1)

        m = self.layer_norm(m)

    
        m = self._transition(m, mask)

        return m

class DecoyPairEmbedder(nn.Module):
    """
    Embeds "template_pair_feat" features.

    Implements Algorithm 2, line 9.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        **kwargs,
    ):
        """
        Args:
            c_in:

            c_out:
                Output channel dimension
        """
        super(DecoyPairEmbedder, self).__init__()

        self.c_in = c_in
        self.c_out = c_out

        # Despite there being no relu nearby, the source uses that initializer
        self.linear = Linear(self.c_in, self.c_out, init="relu")

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, C_in] input tensor
        Returns:
            [*, C_out] output tensor
        """
        # pdb.set_trace()
        x = self.linear(x)

        return x


## Pair feature embedding
class Dropout(nn.Module):
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.

    If not in training mode, this module computes the identity function.
    """

    def __init__(self, r: float, batch_dim: Union[int, List[int]]):
        """
        Args:
            r:
                Dropout rate
            batch_dim:
                Dimension(s) along which the dropout mask is shared
        """
        super(Dropout, self).__init__()

        self.r = r
        if type(batch_dim) == int:
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                Tensor to which dropout is applied. Can have any shape
                compatible with self.batch_dim
        """
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        mask = x.new_ones(shape)
        mask = self.dropout(mask)
        x = x * mask
        return x


class DropoutRowwise(Dropout):
    """
    Convenience class for rowwise dropout as described in subsection
    1.11.6.
    """

    __init__ = partialmethod(Dropout.__init__, batch_dim=-3)

class DropoutColumnwise(Dropout):
    """
    Convenience class for columnwise dropout as described in subsection
    1.11.6.
    """

    __init__ = partialmethod(Dropout.__init__, batch_dim=-2)

class PairTransition(nn.Module):
    """
    Implements Algorithm 15.
    """

    def __init__(self, c_z, n):
        """
        Args:
            c_z:
                Pair transition channel dimension
            n:
                Factor by which c_z is multiplied to obtain hidden channel
                dimension
        """
        super(PairTransition, self).__init__()

        self.c_z = c_z
        self.n = n

        self.layer_norm = nn.LayerNorm(self.c_z)
        self.linear_1 = Linear(self.c_z, self.n * self.c_z, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_z, c_z, init="final")

    def _transition(self, z, mask):
        # [*, N_res, N_res, C_hidden]
        z = self.linear_1(z)
        z = self.relu(z)

        # [*, N_res, N_res, C_z]
        z = self.linear_2(z) * mask

        return z

    @torch.jit.ignore
    def _chunk(self,
        z: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        return chunk_layer(
            self._transition,
            {"z": z, "mask": mask},
            chunk_size=chunk_size,
            no_batch_dims=len(z.shape[:-2]),
        )


    def forward(self, 
        z: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        # DISCREPANCY: DeepMind forgets to apply the mask in this module.
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        # [*, N_res, N_res, 1]
        mask = mask.unsqueeze(-1)

        # [*, N_res, N_res, C_z]
        z = self.layer_norm(z)

        if chunk_size is not None:
            z = self._chunk(z, mask, chunk_size)
        else:
            z = self._transition(z=z, mask=mask)

        return z

class DecoyPairStackBlock(nn.Module):
    def __init__(
        self,
        c_t: int,
        c_hidden_tri_att: int,
        c_hidden_tri_mul: int,
        no_heads: int,
        pair_transition_n: int,
        dropout_rate: float,
        inf: float,
        **kwargs,
    ):
        super(DecoyPairStackBlock, self).__init__()

        self.c_t = c_t
        self.c_hidden_tri_att = c_hidden_tri_att
        self.c_hidden_tri_mul = c_hidden_tri_mul
        self.no_heads = no_heads
        self.pair_transition_n = pair_transition_n
        self.dropout_rate = dropout_rate
        self.inf = inf

        self.dropout_row = DropoutRowwise(self.dropout_rate)
        self.dropout_col = DropoutColumnwise(self.dropout_rate)

        self.tri_att_start = TriangleAttentionStartingNode(
            self.c_t,
            self.c_hidden_tri_att,
            self.no_heads,
            inf=inf,
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            self.c_t,
            self.c_hidden_tri_att,
            self.no_heads,
            inf=inf,
        )

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            self.c_t,
            self.c_hidden_tri_mul,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            self.c_t,
            self.c_hidden_tri_mul,
        )

        self.pair_transition = PairTransition(
            self.c_t,
            self.pair_transition_n,
        )

    def forward(self, 
        z: torch.Tensor, 
        mask: torch.Tensor, 
        chunk_size: Optional[int] = None, 
        _mask_trans: bool = True
    ):
        single_templates = [
            t.unsqueeze(-4) for t in torch.unbind(z, dim=-4)
        ]
        single_templates_masks = [
            m.unsqueeze(-3) for m in torch.unbind(mask, dim=-3)
        ]
        for i in range(len(single_templates)):
            single = single_templates[i]
            single_mask = single_templates_masks[i]
            
            single = single + self.dropout_row(
                self.tri_att_start(
                    single,
                    chunk_size=chunk_size,
                    mask=single_mask
                )
            )
            single = single + self.dropout_col(
                self.tri_att_end(
                    single,
                    chunk_size=chunk_size,
                    mask=single_mask
                )
            )
            single = single + self.dropout_row(
                self.tri_mul_out(
                    single,
                    mask=single_mask
                )
            )
            single = single + self.dropout_row(
                self.tri_mul_in(
                    single,
                    mask=single_mask
                )
            )
            single = single + self.pair_transition(
                single,
                mask=single_mask if _mask_trans else None,
                chunk_size=chunk_size,
            )

            single_templates[i] = single

        z = torch.cat(single_templates, dim=-4)

        return z

class DecoyPairStack(nn.Module):
    """
    Implements Algorithm 16.
    """
    def __init__(
        self,
        c_t,
        c_hidden_tri_att,
        c_hidden_tri_mul,
        no_blocks,
        no_heads,
        pair_transition_n,
        dropout_rate,
        blocks_per_ckpt,
        inf=1e9,
        **kwargs,
    ):
        """
        Args:
            c_t:
                Template embedding channel dimension
            c_hidden_tri_att:
                Per-head hidden dimension for triangular attention
            c_hidden_tri_att:
                Hidden dimension for triangular multiplication
            no_blocks:
                Number of blocks in the stack
            pair_transition_n:
                Scale of pair transition (Alg. 15) hidden dimension
            dropout_rate:
                Dropout rate used throughout the stack
            blocks_per_ckpt:
                Number of blocks per activation checkpoint. None disables
                activation checkpointing
        """
        super(DecoyPairStack, self).__init__()

        self.blocks_per_ckpt = blocks_per_ckpt

        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = DecoyPairStackBlock(
                c_t=c_t,
                c_hidden_tri_att=c_hidden_tri_att,
                c_hidden_tri_mul=c_hidden_tri_mul,
                no_heads=no_heads,
                pair_transition_n=pair_transition_n,
                dropout_rate=dropout_rate,
                inf=inf,
            )
            self.blocks.append(block)

        self.layer_norm = nn.LayerNorm(c_t)

    def forward(
        self,
        t: torch.tensor,
        mask: torch.tensor,
        chunk_size: int,
        _mask_trans: bool = True,
    ):
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] template embedding
            mask:
                [*, N_templ, N_res, N_res] mask
        Returns:
            [*, N_templ, N_res, N_res, C_t] template embedding update
        """
        if(mask.shape[-3] == 1):
            expand_idx = list(mask.shape)
            expand_idx[-3] = t.shape[-4]
            mask = mask.expand(*expand_idx)

        t, = checkpoint_blocks(
            blocks=[
                partial(
                    b,
                    mask=mask,
                    chunk_size=chunk_size,
                    _mask_trans=_mask_trans,
                )
                for b in self.blocks
            ],
            args=(t,),
            blocks_per_ckpt=self.blocks_per_ckpt if self.training else None,
        )

        t = self.layer_norm(t)

        return t

class DecoyPointwiseAttention(nn.Module):
    """
    Implements Algorithm 17.
    """
    def __init__(self, c_t, c_z, c_hidden, no_heads, inf, **kwargs):
        """
        Args:
            c_t:
                Template embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
        """
        super(DecoyPointwiseAttention, self).__init__()

        self.c_t = c_t
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self.mha = Attention(
            self.c_z,
            self.c_t,
            self.c_t,
            self.c_hidden,
            self.no_heads,
            gating=False,
        )
        # self.linear = Linear(self.c_t, self.c_z)

    def _chunk(self,
        z: torch.Tensor,
        t: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
    ) -> torch.Tensor:
        mha_inputs = {
            "q_x": z,
            "k_x": t,
            "v_x": t,
            "biases": biases,
        }
        return chunk_layer(
            self.mha,
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(z.shape[:-2]),
        )


    def forward(self, 
        t: torch.Tensor, 
        z: torch.Tensor, 
        template_mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] template embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            template_mask:
                [*, N_templ] template mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if template_mask is None:
            template_mask = t.new_ones(t.shape[:-3])

        bias = self.inf * (template_mask[..., None, None, None, None, :] - 1)

        # template pair embedding: c_t to c_z
        # z = self.linear(z)
        
        # [*, N_res, N_res, 1, C_z]
        z = z.unsqueeze(-2)

        # [*, N_res, N_res, N_temp, C_t]
        t = permute_final_dims(t, (1, 2, 0, 3))

        # [*, N_res, N_res, 1, C_z]
        biases = [bias]
        if chunk_size is not None:
            z = self._chunk(z, t, biases, chunk_size)
        else:
            z = self.mha(q_x=z, kv_x=t, biases=biases)

        # [*, N_res, N_res, C_z]
        z = z.squeeze(-2)

        return z

class DecoySeqStackBlock(nn.Module):
    def __init__(self, c_s, c_hidden, no_heads, inf, dropout_rate, **kwargs):
        """
            Implement a GPT3-like Transformer to update sequence embeddding
        """
        super().__init__()
        self.c_s = c_s
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.dropout_rate = dropout_rate
        self.mlp = nn.Sequential(
            Linear(c_s, 4*c_s, init="relu"),
            nn.ReLU(),
            Linear(4*c_s, c_s, init="relu"),
            nn.ReLU(),
        )
        self.layer_norm_1 = nn.LayerNorm(self.c_s)
        self.layer_norm_2 = nn.LayerNorm(self.c_s)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.self_mha = Attention(
            self.c_s,
            self.c_s,
            self.c_s,
            self.c_hidden,
            self.no_heads,
            gating=True,
        )
        self.gamma1 = nn.Parameter(torch.empty(1,1,self.c_s), requires_grad=True)
        nn.init.constant_(self.gamma1, 1e-6)
        self.gamma2 = nn.Parameter(torch.empty(1,1,self.c_s), requires_grad=True)
        nn.init.constant_(self.gamma2, 1e-6)
        
    def forward(self, s: torch.Tensor):
        s_identity = s
        s = self.layer_norm_1(s)
        s = s_identity + self.dropout(self.gamma1 * self.self_mha(q_x=s, kv_x=s))
        s = s + self.dropout(self.gamma2 * self.mlp(self.layer_norm_2(s)))

        return s

class DecoySeqStack(nn.Module):
    def __init__(self, c_s, c_hidden, no_heads, inf, dropout_rate, no_blocks, **kwargs):
        super().__init__()
        self.c_s = c_s
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.dropout_rate = dropout_rate
        self.no_blocks = no_blocks
        self.blocks = nn.ModuleList()
        for _ in range(self.no_blocks):
            self.blocks.append(
                DecoySeqStackBlock(c_s, c_hidden, no_heads, inf, dropout_rate)
            )
        # self.layer_norm = nn.LayerNorm(c_s)

    def forward(self, s):
        for i in range(self.no_blocks):
            s = self.blocks[i](s)

        # s = self.layer_norm(s)

        return s
        
class OuterProductMean(nn.Module):
    """
    Implements Algorithm 10.
    """

    def __init__(self, c_s, c_z, c_hidden, eps=1e-3):
        """
        Args:
            c_s:
                single embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
        """
        super(OuterProductMean, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = nn.LayerNorm(c_s)
        self.linear_1 = Linear(c_s, c_hidden)
        self.linear_2 = Linear(c_s, c_hidden)
        self.linear_out = Linear(c_hidden ** 2, c_z, init="final")

    def _opm(self, a, b):
        # [*, N_res, N_res, C, C]
        outer = torch.einsum("...bac,...dae->...bdce", a, b)

        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        return outer

    @torch.jit.ignore
    def _chunk(self, 
        a: torch.Tensor, 
        b: torch.Tensor, 
        chunk_size: int
    ) -> torch.Tensor:
        # Since the "batch dim" in this case is not a true batch dimension
        # (in that the shape of the output depends on it), we need to
        # iterate over it ourselves
        a_reshape = a.reshape((-1,) + a.shape[-3:])
        b_reshape = b.reshape((-1,) + b.shape[-3:])
        out = []
        for a_prime, b_prime in zip(a_reshape, b_reshape):
            outer = chunk_layer(
                partial(self._opm, b=b_prime),
                {"a": a_prime},
                chunk_size=chunk_size,
                no_batch_dims=1,
            )
            out.append(outer)
        outer = torch.stack(out, dim=0)
        outer = outer.reshape(a.shape[:-3] + outer.shape[1:])

        return outer

    def forward(self, 
        m: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            mask:
                [*, N_seq, N_res] MSA mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        # [*, N_seq, N_res, C_m]
        m = self.layer_norm(m)

        # [*, N_seq, N_res, C]
        mask = mask.unsqueeze(-1)
        a = self.linear_1(m) * mask
        b = self.linear_2(m) * mask

        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        if chunk_size is not None:
            outer = self._chunk(a, b, chunk_size)
        else:
            outer = self._opm(a, b)

        # [*, N_res, N_res, 1]
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)

        # [*, N_res, N_res, C_z]
        outer = outer / (self.eps + norm)

        return outer

    
def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
    ])[activation]

def norm_func(norm, n_channel):
    return  nn.ModuleDict([
        ['instance', nn.InstanceNorm1d(n_channel)],
    ])[norm]

class AtomEmbLayer(nn.Module):
    def __init__(self,
        atom_emb_in=7, atom_emb_h=256,
        norm='instance', activation='relu', 
        *args, **kwargs
        ):
        super(AtomEmbLayer, self).__init__()
        self.norm = norm

        self.fn_atom_norm = norm_func(norm, atom_emb_in)
        self.fn_atom_linear = nn.Linear(atom_emb_in, atom_emb_h, bias=False)
        self.fn_atom_activation = activation_func(activation)
        self.fn_atom_norm2 = norm_func(norm, atom_emb_h)
        self.fn_atom_linear2 = nn.Linear(atom_emb_h, atom_emb_h, bias=False)

    def forward(self, batch):
        atom_emb = batch['atom_emb']
        # first layer
        # shape: [B,N,14,7]
        batch_size = atom_emb.shape[0]
        atom_num = atom_emb.shape[-2]
        dim = atom_emb.shape[-1]
        atom_emb = torch.reshape(atom_emb, (-1, atom_num, dim))
        # shape: [B*N, 14, 7]
        atom_emb = self.fn_atom_norm(atom_emb)
        atom_emb = self.fn_atom_linear(atom_emb)
        atom_emb = torch.mean(atom_emb, 1)
        # shape: [B*N,256]
        atom_emb = self.fn_atom_activation(atom_emb)
        # second layer
        atom_emb = self.fn_atom_norm2(atom_emb.unsqueeze(1)).squeeze() if self.norm=="instance" else self.fn_atom_norm2(atom_emb)
        atom_emb = self.fn_atom_linear2(atom_emb)
        atom_emb = self.fn_atom_activation(atom_emb)
        atom_emb_h = atom_emb.shape[-1]
        # cat
        atom_emb = torch.reshape(atom_emb,(batch_size,-1,atom_emb_h))
        # shape: [B,N,256]
        # pdb.set_trace()
        x = torch.cat((batch["decoy_angle_feats"], atom_emb), dim=-1)

        return x

class RecyclingEmbedder(nn.Module):
    """
    Embeds the output of an iteration of the model for recycling.

    Implements Algorithm 32.
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        min_bin: float,
        max_bin: float,
        no_bins: int,
        inf: float = 1e8,
        **kwargs,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair embedding channel dimension
            min_bin:
                Smallest distogram bin (Angstroms)
            max_bin:
                Largest distogram bin (Angstroms)
            no_bins:
                Number of distogram bins
        """
        super(RecyclingEmbedder, self).__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.inf = inf

        self.bins = None

        self.linear = Linear(self.no_bins, self.c_z)
        self.layer_norm_m = nn.LayerNorm(self.c_m)
        self.layer_norm_z = nn.LayerNorm(self.c_z)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                First row of the MSA embedding. [*, N_res, C_m]
            z:
                [*, N_res, N_res, C_z] pair embedding
            x:
                [*, N_res, 3] predicted C_beta coordinates
        Returns:
            m:
                [*, N_res, C_m] MSA embedding update
            z:
                [*, N_res, N_res, C_z] pair embedding update
        """
        if self.bins is None:
            self.bins = torch.linspace(
                self.min_bin,
                self.max_bin,
                self.no_bins,
                dtype=x.dtype,
                device=x.device,
            )

        # [*, N, C_m]
        m_update = self.layer_norm_m(m)

        # This squared method might become problematic in FP16 mode.
        # I'm using it because my homegrown method had a stubborn discrepancy I
        # couldn't find in time.
        squared_bins = self.bins ** 2
        upper = torch.cat(
            [squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1
        )
        d = torch.sum(
            (x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdims=True
        )

        # d_sc = torch.sum(
        #     (x_sc[..., None, :] - x_sc[..., None, :, :]) ** 2, dim=-1, keepdims=True
        # )

        # [*, N, N, no_bins]
        d = ((d > squared_bins) * (d < upper)).type(x.dtype)
        # d_sc = ((d_sc > squared_bins) * (d_sc < upper)).type(x.dtype)

        # [*, N, N, C_z]
        d = self.linear(d)
        # d_sc = self.linear(d_sc)
        z_update = d + self.layer_norm_z(z)

        return m_update, z_update



