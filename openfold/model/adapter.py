import torch.nn as nn
from typing import Tuple, Optional
import torch
from functools import partial
class AdapterShare(nn.Module):

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_InEmb: int,
        c_AdaEmb: int,
        **kwargs,
    ):
        super(AdapterShare, self).__init__()
        self.linear_share_DMT = nn.Linear(c_InEmb, c_AdaEmb)
        self.linear_seq = nn.Linear(c_AdaEmb, c_s)
        self.linear_seq_fuse = nn.Linear(2*c_s, c_s)

        self.linear_str_i = nn.Linear(c_AdaEmb, c_z)
        self.linear_str_j = nn.Linear(c_AdaEmb, c_z)
        self.linear_str_fuse = nn.Linear(2*c_z, c_z)

    def forward(
        self,
        MSAQuaryEmb: torch.Tensor, #s
        PairEmb: torch.Tensor, #z
        SeqEmb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        AdaEmb = self.linear_share_DMT(SeqEmb)

        PseudoMSAQuaryEmb= self.linear_seq(AdaEmb)
        NewMSAQuaryEmb = self.linear_seq_fuse(torch.cat((PseudoMSAQuaryEmb, MSAQuaryEmb), -1))

        PseudoPairEmb_i = self.linear_str_i(AdaEmb)
        PseudoPairEmb_j = self.linear_str_j(AdaEmb)
        PseudoPairEmb = PseudoPairEmb_i[..., None, :] + PseudoPairEmb_j[..., None, :, :]
        NewPairEmb = self.linear_str_fuse(torch.cat((PseudoPairEmb, PairEmb), -1))

        return NewMSAQuaryEmb, NewPairEmb, AdaEmb


class AdapterSeperate(nn.Module):

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_InEmb: int,
        c_AdaEmb: int,
        **kwargs,
    ):
        super(AdapterSeperate, self).__init__()
        self.linear_seq_DMT = nn.Linear(c_InEmb, c_AdaEmb)
        self.linear_seq = nn.Linear(c_AdaEmb, c_s)
        self.linear_seq_fuse = nn.Linear(2*c_s, c_s)

        self.linear_str_DMT = nn.Linear(c_InEmb, c_AdaEmb)
        self.linear_str_i = nn.Linear(c_AdaEmb, c_z)
        self.linear_str_j = nn.Linear(c_AdaEmb, c_z)
        self.linear_str_fuse = nn.Linear(2*c_z, c_z)

        self.skip_connection_init()
        print('finshi init')

    def skip_connection_init(self, ):
        for n, m in self.named_children():
            if n == 'linear_seq_fuse' or n == 'linear_str_fuse':
                with torch.no_grad():
                    m.bias.data.fill_(0.0)
                    m.weight.data.fill_(0.0)

                    n_row, n_col = m.weight.data.shape
                    col_idx = int(n_col/2)
                    for row_idx in range(n_row):
                        m.weight.data[row_idx, col_idx+row_idx] = 1.0


    def forward(
        self,
        MSAQuaryEmb: torch.Tensor, #s
        PairEmb: torch.Tensor, #z
        SeqEmb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        AdaSeqEmb = self.linear_seq_DMT(SeqEmb)
        PseudoMSAQuaryEmb= self.linear_seq(AdaSeqEmb)
        NewMSAQuaryEmb = self.linear_seq_fuse(torch.cat((PseudoMSAQuaryEmb, MSAQuaryEmb), -1))

        AdaStrEmb = self.linear_str_DMT(SeqEmb)
        PseudoPairEmb_i = self.linear_str_i(AdaStrEmb)
        PseudoPairEmb_j = self.linear_str_j(AdaStrEmb)
        PseudoPairEmb = PseudoPairEmb_i[..., None, :] + PseudoPairEmb_j[..., None, :, :]
        NewPairEmb = self.linear_str_fuse(torch.cat((PseudoPairEmb, PairEmb), -1))

        return NewMSAQuaryEmb, NewPairEmb, AdaSeqEmb, AdaStrEmb


