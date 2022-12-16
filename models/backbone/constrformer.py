from tkinter import N
from turtle import forward
from numpy import block
import torch
import torch.nn as nn
from openfold.model.primitives import (
    Linear,
    LayerNorm, 
    Attention,
)
from openfold.utils.tensor_utils import (
    masked_mean,
    one_hot, 
    chunk_layer, 
    permute_final_dims
)
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
from typing import Tuple
from ..encoders.embedder import (
    PairTransition,
    DropoutRowwise,
    DropoutColumnwise
)
import pdb
class SeqAttention(nn.Module):
    def __init__(self, c_s, c_z, c_hidden, no_heads, inf=1e9):
        """
        Args:
            c_s:
                Input channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
            inf:
                Large number used to construct attention masks
        """
        super(SeqAttention, self).__init__()
        
        self.c_s = c_s
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.c_z = c_z
        self.inf = inf

        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)
        self.linear_z = Linear(
            self.c_z, self.no_heads, bias=False, init="normal"
        )
        self.mha = Attention(
            self.c_s, self.c_s, self.c_s, self.c_hidden, self.no_heads
        )
    
    def _prep_inputs(self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [*, N_seq, N_res, C_m]
        s = self.layer_norm_s(s)

        n_seq, n_res = s.shape[-3:-1]
        if mask is None:
            # [*, N_seq, N_res]
            mask = s.new_ones(
                s.shape[:-3] + (n_seq, n_res),
            )

        # [*, N_seq, 1, 1, N_res]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # This step simply returns a larger view of the bias, and does not
        # consume additional memory.
        # [*, N_seq, no_heads, N_res, N_res]
        #bias = bias.expand(
        #    ((-1,) * len(bias.shape[:-4])) + (-1, self.no_heads, n_res, -1)
        #)

        # [*, N_res, N_res, C_z]
        z = self.layer_norm_z(z)
        
        # [*, N_res, N_res, no_heads]
        z = self.linear_z(z)
        # pdb.set_trace()
        # [*, 1, no_heads, N_res, N_res]
        z = permute_final_dims(z, (2, 0, 1)).unsqueeze(-4)

        return s, mask_bias, z

    def forward(self, 
        s: torch.Tensor, 
        z: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None, 
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] sequence embedding
            z:
                [*, N_res, N_res, C_z] pair embedding. Required only if
                pair_bias is True
            mask:
                [*, N_seq, N_res] MSA mask
        """
        s, mask_bias, z = self._prep_inputs(s, z, mask)
        biases = [mask_bias]
        if(z is not None):
            biases.append(z)
        s = self.mha(
                q_x=s, 
                kv_x=s, 
                biases=biases 
            )

        return s

class SeqTransition(nn.Module):
    """
    Feed-forward network applied to MSA activations after attention.
    Implements Algorithm 9
    """
    def __init__(self, c_s, n):
        """
        Args:
            c_s:
                Seq channel dimension
            n:
                Factor multiplied to c_s to obtain the hidden channel
                dimension
        """
        super(SeqTransition, self).__init__()

        self.c_s = c_s
        self.n = n

        self.layer_norm = LayerNorm(self.c_s)
        self.linear_1 = Linear(self.c_s, self.n * self.c_s, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_s, self.c_s, init="final")

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
                [*, N_seq, N_res, C_m] Single Sequence activation
            mask:
                [*, N_seq, N_res] MSA mask
        Returns:
            s:
                [*, N_seq, N_res, C_m] Sequence activation update
        """
        # DISCREPANCY: DeepMind forgets to apply the MSA mask here.
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        mask = mask.unsqueeze(-1)

        m = self.layer_norm(m)

    
        m = self._transition(m, mask)

        return m

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

    def forward(self, 
        m: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_s] MSA embedding
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

        outer = self._opm(a, b)

        # [*, N_res, N_res, 1]
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)

        # [*, N_res, N_res, C_z]
        outer = outer / (self.eps + norm)

        return outer

class OuterDifferenceMean(nn.Module):
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
        super(OuterDifferenceMean, self).__init__()

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

    def forward(self, 
        m: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
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
        c = (a-b)/(a + 1e-9) # avoid overflow

        outer = self._opm(a, c) # when a = 0, the out = 0,
        # so it's ok to use this way

        # [*, N_res, N_res, 1]
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)

        # [*, N_res, N_res, C_z]
        outer = outer / (self.eps + norm)

        return outer

class FoldingBlock(nn.Module):
    def __init__(self,
        c_s: int,
        c_z: int,
        c_hidden_seq_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_seq: int,
        no_heads_pair: int,
        transition_n: int,
        seq_dropout: float,
        pair_dropout: float,
        inf: float,
        eps: float,
    ):
        super().__init__()

        self.mha = SeqAttention(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden_seq_att,
            no_heads=no_heads_seq,
            inf=inf
        )
        self.seqtransition = SeqTransition(
            c_s=c_s,
            n=transition_n,
        )
        self.seq_dropout_layer = DropoutRowwise(seq_dropout)
        self.outer_linear = Linear(
            in_dim=2*c_z,
            out_dim=c_z,
        )
        self.outer_product_mean = OuterProductMean(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden_opm
        )
        self.outer_difference_mean = OuterDifferenceMean(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden_opm
        )
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z,
            c_hidden_mul,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_z,
            c_hidden_mul,
        )

        self.tri_att_start = TriangleAttentionStartingNode(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )

        self.pair_transition = PairTransition(
            c_z,
            transition_n,
        )
        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)
        self.ps_dropout_col_layer = DropoutColumnwise(pair_dropout)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        _mask_trans: bool = False,
    )-> Tuple[torch.Tensor, torch.Tensor]:
        msa_trans_mask = msa_mask if _mask_trans else None
        pair_trans_mask = pair_mask if _mask_trans else None
        # pdb.set_trace()
        s = s + self.seq_dropout_layer(
            self.mha(s,z=z,mask=msa_mask)
        )
        s = s + self.seqtransition(s, msa_trans_mask)
        outer1 = self.outer_product_mean(s, mask=msa_mask)
        outer2 = self.outer_difference_mean(s, mask=msa_mask)
        # pdb.set_trace()
        outer = torch.cat(
            (outer1, outer2),dim=-1
            )
        # pdb.set_trace()
        z = z + self.outer_linear(outer)
        z = z + self.ps_dropout_row_layer(self.tri_mul_out(z, mask=pair_mask))
        z = z + self.ps_dropout_row_layer(self.tri_mul_in(z, mask=pair_mask))
        z = z + self.ps_dropout_row_layer(
            self.tri_att_start(z, mask=pair_mask)
        )
        z = z + self.ps_dropout_col_layer(
            self.tri_att_end(z, mask=pair_mask)
        )
        z = z + self.pair_transition(
            z, mask=pair_trans_mask
        )

        return s, z

class ConstrainFormer(nn.Module):
    """
    Embedding constrain or feature into high dim general embedding just like the output of evoformer, in order to give the structure module enough information to decode for getting high quaility structure.
    """
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        inf: float,
        eps: float, 
        **kwargs,
    ):
        """
        Args:
            c_s:
                Channel dimension of the output "single" embedding
            c_z:
                Pair channel dimension
            c_hidden_msa_att:
                Hidden dimension in MSA attention
            c_hidden_opm:
                Hidden dimension in outer product mean module
            c_hidden_mul:
                Hidden dimension in multiplicative updates
            c_hidden_pair_att:
                Hidden dimension in triangular attention
            no_heads_msa:
                Number of heads used for MSA attention
            no_heads_pair:
                Number of heads used for pair attention
            no_blocks:
                Number of Evoformer blocks in the stack
            transition_n:
                Factor by which to multiply c_m to obtain the MSATransition
                hidden dimension
            msa_dropout:
                Dropout rate for MSA activations
            pair_dropout:
                Dropout used for pair activations
        """
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = FoldingBlock(
                c_s=c_s,
                c_z=c_z,
                c_hidden_seq_att=c_hidden_msa_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_seq=no_heads_msa,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                seq_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                inf=inf,
                eps=eps
            )
            self.blocks.append(block)

    def forward(self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        _mask_trans: bool = False,
    )-> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            msa_mask:
                [*, N_seq, N_res] MSA mask
            pair_mask:
                [*, N_res, N_res] pair mask
        Returns:
            s:
                [*, N_res, C_s] single embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
        """
        blocks = [
            partial(
                b,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        args=(m, z)

        def wrap(a):
            return (a,) if type(a) is not tuple else a
        def exec(b, a):
            for block in b:
                a = wrap(block(*a))
                # pdb.set_trace()
            return a
        # pdb.set_trace()
        m, z =  exec(blocks, args)

        return m, z

       
            


        








