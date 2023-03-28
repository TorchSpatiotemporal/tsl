import torch
from einops import rearrange
from torch import nn

import tsl


class DenseGraphConvOrderK(nn.Module):
    """Dense implementation of the spatial diffusion convolution of order
    :math:`K`.

    Args:
        input_size (int): Size of the input.
        output_size (int): Size of the output.
        support_len (int): Number of reference operators.
        order (int): Order of the diffusion process.
        include_self (bool): Whether to include the central node or not.
        channel_last(bool, optional): Whether to use the pattern "b t n f" as
            opposed to "b f n t".
    """

    def __init__(self,
                 input_size,
                 output_size,
                 support_len=3,
                 order=2,
                 include_self=True,
                 channel_last=False):
        super(DenseGraphConvOrderK, self).__init__()
        self.channel_last = channel_last
        self.include_self = include_self
        input_size = (order * support_len +
                      (1 if include_self else 0)) * input_size
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
            support = DenseGraphConvOrderK.compute_support(adj, device)
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

    # Adapted from: https://github.com/nnzhan/Graph-WaveNet
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


class DenseGraphConv(nn.Module):
    r"""A dense graph convolution performing :math:`\mathbf{X}^{\prime} =
    \mathbf{\tilde{A}} \mathbf{X} \boldsymbol{\Theta} + \mathbf{b}`.

    Args:
        input_size: Size of the input.
        output_size: Output size.
        bias: Whether to add a learnable bias.
    """

    def __init__(self, input_size, output_size, bias=True):
        super(DenseGraphConv, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        if bias:
            self.b = nn.Parameter(torch.Tensor(output_size))
        else:
            self.register_parameter('b', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()
        if self.b is not None:
            self.b.data.zero_()

    def forward(self, x, adj):
        """"""
        b, s, n, f = x.size()
        # linear transformation
        x = self.linear(x)

        # reshape to have features+T as last dim
        x = rearrange(x, 'b s n f -> b n (s f)')
        # message passing
        x = torch.matmul(adj, x)
        x = rearrange(x, 'b n (s c) -> b n s f', s=s, f=f)
        if self.b is not None:
            x = x + self.b
        return x
