import torch
from einops import rearrange
from torch import nn

import tsl
from tsl.nn.base import TemporalConv2d


class SpatialConvOrderK(nn.Module):
    """
    Dense implementation the spatial diffusion of order K.
    Adapted from: https://github.com/nnzhan/Graph-WaveNet

    Args:
        input_size (int): Size of the input.
        output_size (int): Size of the output.
        support_len (int): Number of reference operators.
        order (int): Order of the diffusion process.
        include_self (bool): Whether to include the central node or not.
        channel_last(bool, optional): Whether to use the layout "B S N C" as opposed to "B C N S"
    """

    def __init__(self, input_size, output_size, support_len=3, order=2, include_self=True, channel_last=False):
        super(SpatialConvOrderK, self).__init__()
        self.channel_last = channel_last
        self.include_self = include_self
        input_size = (order * support_len + (1 if include_self else 0)) * input_size
        self.mlp = nn.Conv2d(input_size, output_size, kernel_size=1)
        self.order = order

    @staticmethod
    def compute_support(adj, device=None):
        if device is not None:
            adj = adj.to(device)
        adj_bwd = adj.T
        adj_fwd = adj / (adj.sum(1, keepdims=True) + tsl.epsilon)
        adj_bwd = adj_bwd / (adj_bwd.sum(1, keepdims=True) + tsl.epsilon)
        support = [adj_fwd, adj_bwd]
        return support

    @staticmethod
    def compute_support_orderK(adj, k, include_self=False, device=None):
        if isinstance(adj, (list, tuple)):
            support = adj
        else:
            support = SpatialConvOrderK.compute_support(adj, device)
        supp_k = []
        for a in support:
            ak = a
            for i in range(k - 1):
                ak = torch.matmul(ak, a)
                supp_k.append(ak)

        if not include_self:
            for ak in supp_k:
                ak.fill_diagonal_(0.)
        return support + supp_k

    def forward(self, x, support):
        """"""
        squeeze = False
        if self.channel_last:
            if x.dim() == 3:
                # [batch, nodes, channels]
                squeeze = True
                x = rearrange(x, 'b n c -> b c n 1')
            else:
                # [batch, steps, nodes, channels]
                x = rearrange(x, 'b s n c -> b c n s')
        else:
            if x.dim() == 3:
                # [batch, channels, nodes]
                squeeze = True
                x = torch.unsqueeze(x, -1)
        out = [x] if self.include_self else []
        if (type(support) is not list):
            support = [support]
        for a in support:
            x1 = x
            for k in range(self.order):
                x1 = torch.einsum('ncvl, wv -> ncwl', (x1, a)).contiguous()
                out.append(x1)

        out = torch.cat(out, dim=1)
        out = self.mlp(out)
        if squeeze:
            out = out.squeeze(-1)
        if self.channel_last:
            out = rearrange(out, 'b c n ... -> b ... n c')
        return out


class SpatialConv(nn.Module):
    """
    Simple Graph Convolution. Expects data to have layout B C N S.

    Args:
        input_size: Size fo the input.
        output_size: Output size.
        bias: Whether to add a learnable bias.
    """
    def __init__(self, input_size, output_size, bias=True):
        super(SpatialConv, self).__init__()
        self.c_in = input_size
        self.c_out = output_size
        self.linear = TemporalConv2d(self.c_in, self.c_out, kernel_size=1, bias=False)
        if bias:
            self.b = nn.Parameter(torch.zeros(self.c_out))
        else:
            self.register_parameter('b', None)

    def forward(self, x, adj):
        """"""
        b, c, n, s = x.size()
        # linear transformation
        x = self.linear(x)

        # reshape to have features+T as last dim
        x = rearrange(x, 'b c n s -> b n (s c)')
        # message passing
        x = torch.matmul(adj, x)
        x = rearrange(x, 'b n (s c) -> b n s c', s=s, c=c)
        if self.b is not None:
            x = x + self.b
        x = rearrange(x, 'b n s c -> b x n s')
        return x