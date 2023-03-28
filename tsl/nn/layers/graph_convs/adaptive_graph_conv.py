import math

import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveGraphConv(nn.Module):
    """The Dense Adaptive Graph Convolution operator from the paper
    `"Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting"
    <https://arxiv.org/abs/2007.02842>`_ (Bai et al., NeurIPS 2020).

    Args:
        input_size: Size of the input.
        emb_size: Size of the input node embeddings.
        output_size: Output size.
        num_nodes: Number of nodes in the input graph.
        bias: Whether to add a learnable bias.
    """

    def __init__(self,
                 input_size: int,
                 emb_size: int,
                 output_size: int,
                 num_nodes: int,
                 bias: bool = True):
        super(AdaptiveGraphConv, self).__init__()
        self.weight = nn.Parameter(
            torch.Tensor(emb_size, 2, input_size, output_size))
        self.num_nodes = num_nodes
        if bias:
            self.b = nn.Parameter(torch.Tensor(emb_size, output_size))
        else:
            self.register_parameter('b', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.zero_()

    @staticmethod
    def compute_adj(node_emb):
        return F.softmax(F.relu(node_emb @ node_emb.transpose(0, 1)), -1)

    def forward(self, x, e, adj=None):
        """"""
        # compute adaptive adj
        if adj is None:
            adj = self.compute_adj(e)
        # compute adaptive weights
        weight_adp = torch.einsum('nd, dkio->nkio', e, self.weight)
        # propagate + skip_con
        out = torch.stack([torch.matmul(adj, x), x], 2)
        # update features
        out = torch.einsum('bnki, nkio->bno', out, weight_adp)
        if self.b is not None:
            bias_adp = e @ self.b
            out = out + bias_adp
        return out
