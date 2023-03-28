import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor, matmul

from tsl.nn.layers.graph_convs.mixin import NormalizedAdjacencyMixin
from tsl.nn.utils import get_functional_activation


class _TopK(torch.nn.Module):

    def __init__(self, input_size, k):
        super().__init__()
        self.k = k

        self.p = Parameter(torch.Tensor(input_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize based on the number of rows
        stdv = 1. / math.sqrt(self.p.size(0))
        self.p.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """"""
        scores = x @ self.p / self.p.norm()
        vals, topk_idxs = scores.topk(self.k, dim=-2)  # b k 1 -> b k f
        idxs = topk_idxs.expand(-1, self.k, x.size(-1))
        x = torch.gather(x, -2, idxs)
        out = x * F.tanh(vals)

        return out


class _EvolveGCNCell(MessagePassing, NormalizedAdjacencyMixin):
    r"""
    """

    def __init__(self,
                 in_size,
                 out_size,
                 norm,
                 activation='relu',
                 root_weight=False,
                 bias=True,
                 cached=False):
        super(_EvolveGCNCell, self).__init__(aggr='add')
        self.in_size = in_size
        self.out_size = out_size
        self.norm = norm
        self.cached = cached

        self.activation_fn = get_functional_activation(activation)

        self.W0 = nn.Parameter(torch.Tensor(in_size, out_size))

        if bias:
            self.bias = Parameter(torch.Tensor(out_size))
        else:
            self.register_parameter('bias', None)

        if root_weight:
            self.skip_con = nn.Linear(in_size, out_size, bias=False)
        else:
            self.register_parameter('skip_con', None)

    def reset_parameters(self):
        std = 1. / math.sqrt(self.out_size)
        self.W0.data.uniform_(-std, std)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
        if self.skip_con is not None:
            self.skip_con.reset_parameters()


class EvolveGCNHCell(_EvolveGCNCell):
    r"""EvolveGCNH cell from the paper `"EvolveGCN: Evolving Graph Convolutional
    Networks for Dynamic Graphs" <https://arxiv.org/abs/1902.10191>`_ (Pereja et
    al., AAAI 2020).

    This variant of the model adapts the weights of the graph convolution by
    looking at node features.

    Args:
        in_size (int): Size of the input.
        out_size (int): Number of units in the hidden state.
        norm (bool): Methods used to normalize the adjacency matrix.
        activation (str): Activation function after the GCN layer.
        root_weight (bool): Whether to add a parametrized skip connection.
        bias (bool): Whether to learn a bias.
        cached (bool): Whether to cache normalized edge_weights.
    """
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self,
                 in_size,
                 out_size,
                 norm,
                 activation='relu',
                 root_weight=False,
                 bias=True,
                 cached=False):
        super(EvolveGCNHCell, self).__init__(in_size,
                                             out_size,
                                             norm=norm,
                                             activation=activation,
                                             root_weight=root_weight,
                                             bias=bias,
                                             cached=cached)
        self.gru_cell = nn.GRUCell(input_size=in_size, hidden_size=in_size)
        self.pooling_layer = _TopK(input_size=in_size, k=out_size)
        self.reset_parameters()

    def reset_parameters(self):
        super(EvolveGCNHCell, self).reset_parameters()
        self.gru_cell.reset_parameters()
        self.pooling_layer.reset_parameters()

    def forward(self, x, h, edge_index, edge_weight=None):
        """"""
        edge_index, edge_weight = self.normalize_edge_index(
            x, edge_index, edge_weight, use_cached=self.cached)

        if h is None:
            W = repeat(self.W0, 'din dout -> b din dout', b=x.size(0))
        else:
            W = h

        h_gru = rearrange(W, 'b din dout -> (b dout) din')
        x_gru = rearrange(self.pooling_layer(x), 'b dout din -> (b dout) din')
        h_gru = self.gru_cell(x_gru, h_gru)
        W = rearrange(h_gru, '(b dout) din -> b din dout', dout=self.out_size)

        out = torch.matmul(x, W)
        out = self.propagate(edge_index, x=out, edge_weight=edge_weight)

        if self.bias is not None:
            out += self.bias

        if self.skip_con is not None:
            out += self.skip_con(x)

        return self.activation_fn(out), W

    def message(self, x_j: Tensor, edge_weight) -> Tensor:
        """"""
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        """"""
        # adj_t: SparseTensor [nodes, nodes]
        # x: [(batch,) nodes, channels]
        return matmul(adj_t, x, reduce=self.aggr)


class EvolveGCNOCell(_EvolveGCNCell):
    r"""EvolveGCNO cell from the paper `"EvolveGCN: Evolving Graph Convolutional
    Networks for Dynamic Graphs" <https://arxiv.org/abs/1902.10191>`_ (Pereja et
    al., AAAI 2020).

    This variant of the model simply updates the weights of the graph
    convolution.

    Args:
        in_size (int): Size of the input.
        out_size (int): Number of units in the hidden state.
        norm (str): Method used to normalize the adjacency matrix.
        activation (str): Activation function after the GCN layer.
        root_weight (bool): Whether to add a parametrized skip connection.
        bias (bool): Whether to learn a bias.
        cached (bool): Whether to cache normalized edge_weights.
    """
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self,
                 in_size,
                 out_size,
                 norm,
                 activation='relu',
                 root_weight=False,
                 bias=True,
                 cached=False):
        super(EvolveGCNOCell, self).__init__(in_size,
                                             out_size,
                                             norm=norm,
                                             activation=activation,
                                             root_weight=root_weight,
                                             bias=bias,
                                             cached=cached)
        self.lstm_cell = nn.LSTMCell(input_size=in_size, hidden_size=in_size)
        self.reset_parameters()

    def reset_parameters(self):
        super(EvolveGCNOCell, self).reset_parameters()
        self.lstm_cell.reset_parameters()

    def forward(self, x, hs, edge_index, edge_weight=None):
        """"""
        edge_index, edge_weight = self.normalize_edge_index(
            x, edge_index, edge_weight, use_cached=self.cached)

        if hs is None:
            W = repeat(self.W0, 'din dout -> b din dout', b=x.size(0))
            hc = None
        else:
            W, hc = hs

        x_lstm = rearrange(W, 'b din dout -> (b dout) din')
        hc = self.lstm_cell(x_lstm, hc)
        W = rearrange(hc[0], '(b dout) din -> b din dout', dout=self.out_size)

        out = torch.matmul(x, W)
        out = self.propagate(edge_index, x=out, edge_weight=edge_weight)

        if self.bias is not None:
            out += self.bias

        if self.skip_con is not None:
            out += self.skip_con(x)

        return self.activation_fn(out), (W, hc)
