# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numbers
from typing import Callable
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
import pdb

class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        assert elementwise_affine
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.bias = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()
        def torch_layer_norm(input):
            return F.layer_norm(
                input, self.normalized_shape, self.weight.type(input.dtype), self.bias.type(input.dtype), self.eps)
        # def fused_layer_norm(input):
        #     if input.is_cuda:
        #         return FusedLayerNormFastFunction.apply(
        #             input, self.weight.type(input.dtype), self.bias.type(input.dtype), self.normalized_shape, self.eps)
        #     else:
        #         return F.layer_norm(
        #             input, self.normalized_shape, self.weight.type(input.dtype), self.bias.type(input.dtype), self.eps)
        # self.func = torch_layer_norm if (not HAS_LAYER_NORM or normalized_shape[0] not in FUSED_LAYER_NORM_SUPPORT_DIM) else fused_layer_norm
        self.func = torch_layer_norm

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input):
        return self.func(input)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine=True'.format(**self.__dict__)
    
def softmax_dropout(input, dropout_prob, is_training=True, mask=None, bias=None, inplace=True):
    """softmax dropout, and mask, bias are optional.
    Args:
        input (torch.Tensor): input tensor
        dropout_prob (float): dropout probability
        is_training (bool, optional): is in training or not. Defaults to True.
        mask (torch.Tensor, optional): the mask tensor, use as input + mask . Defaults to None.
        bias (torch.Tensor, optional): the bias tensor, use as input + bias . Defaults to None.
    Returns:
        torch.Tensor: the result after softmax
    """
    input = input.contiguous()
    if not inplace:
        # copy a input for non-inplace case
        input = input.clone()
    # if input.is_cuda and HAS_SOFTMAX:
    #     input_size = input.size()
    #     if mask is not None:
    #         if _check_mask(mask, input):
    #             mask = mask.contiguous().view(-1, mask.shape[-2], mask.shape[-1])
    #         else:
    #             input += mask
    #             mask = None
    #     if bias is not None:
    #         if _check_bias(bias, input):
    #             bias = bias.contiguous().view(-1, input_size[-2], input_size[-1])
    #         else:
    #             input += bias
    #             bias = None
    #     input = input.view(-1, input_size[-2], input_size[-1])
    #     if dropout_prob <= 0.0 or input_size[-1] <= 1024:
    #         return SoftmaxDropoutFast.apply(
    #             is_training, input, mask, bias, dropout_prob
    #         ).view(*input_size)
    #     else:
    #         return F.dropout(SoftmaxDropoutFast.apply(
    #             is_training, input, mask, bias, 0.0
    #         ).view(*input_size), p=dropout_prob, training=is_training)
    # else:
    if mask is not None:
        input += mask
    if bias is not None:
        input += bias

    return F.dropout(F.softmax(input, dim=-1), p=dropout_prob, training=is_training)

def get_activation_fn(activation: str) -> Callable:
    """ Returns the activation function corresponding to `activation` """

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))