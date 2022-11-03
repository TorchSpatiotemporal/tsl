import math
from typing import Optional, Tuple

import torch
from einops import rearrange, repeat
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor, matmul

from tsl.nn.blocks.encoders.gcrnn import _GraphGRUCell, _GraphRNN

from torch import nn, Tensor

from tsl.nn.layers.graph_convs.mixin import NormalizedAdjacencyMixin
from tsl.nn.utils import get_functional_activation
from tsl.ops.connectivity import normalize_connectivity


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
        scores = x @ self.p / self.p.norm()
        vals, topk_idxs = scores.topk(self.k, dim=-2)  # b k 1 -> b k f
        idxs = topk_idxs.expand(-1, self.k, x.size(-1))
        x = torch.gather(x, -2, idxs)
        out = x * F.tanh(vals)

        return out


class _EvolveGCNCell(MessagePassing, NormalizedAdjacencyMixin):
    r"""
    """

    def __init__(self, in_size, out_size, asymmetric_norm, activation='relu', root_weight=False, bias=True, cached=False):
        super(_EvolveGCNCell, self).__init__(aggr='add')
        self.in_size = in_size
        self.out_size = out_size
        self.asymmetric_norm = asymmetric_norm
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
    r"""

    EvolveGCNH model from Pereja et al., "EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs", AAAI 2020.
    This variant of the model adapts the weights of the graph convolution by looking at node features.

    Args:
        in_size (int): Size of the input.
        out_size (int): Number of units in the hidden state.
        asymmetric_norm (bool): Whether to consider the graph as directed when normalizaing weights.
        activation (str): Activation function after the GCN layer.
        root_weight (bool): Whether to add a parametrized skip connection.
        bias (bool): Whether to learn a bias.
        cached (bool): Whether to cache normalized edge_weights.
    """
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_size, out_size, asymmetric_norm, activation='relu', root_weight=False, bias=True, cached=False):
        super(EvolveGCNHCell, self).__init__(in_size,
                                             out_size,
                                             asymmetric_norm=asymmetric_norm,
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
        edge_index, edge_weight = self._normalize_edge_index(x,
                                                             edge_index,
                                                             edge_weight,
                                                             use_cached=self.cached)

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
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        """"""
        # adj_t: SparseTensor [nodes, nodes]
        # x: [(batch,) nodes, channels]
        return matmul(adj_t, x, reduce=self.aggr)

class EvolveGCNOCell(_EvolveGCNCell):
    r"""
    EvolveGCNH model from Pereja et al., "EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs", AAAI 2020.
    This variant of the model simply updates the weights of the graph convolution.

    Args:
        in_size (int): Size of the input.
        out_size (int): Number of units in the hidden state.
        asymmetric_norm (bool): Whether to consider the graph as directed when normalizaing weights.
        activation (str): Activation function after the GCN layer.
        root_weight (bool): Whether to add a parametrized skip connection.
        bias (bool): Whether to learn a bias.
        cached (bool): Whether to cache normalized edge_weights.
    """
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_size, out_size, asymmetric_norm, activation='relu', root_weight=False, bias=True, cached=False):
        super(EvolveGCNOCell, self).__init__(in_size,
                                             out_size,
                                             asymmetric_norm=asymmetric_norm,
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
        edge_index, edge_weight = self._normalize_edge_index(x, edge_index, edge_weight, use_cached=self.cached)

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


class EvolveGCN(nn.Module):
    r"""
    EvolveGCN encoder form Pereja et al., "EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs", AAAI 2020.

    Args:
        input_size (int): Size of the input.
        hidden_size (int): Number of hidden units in each hidden layer.
        n_layers (int): Number of layers in the encoder.
        asymmetric_norm (bool): Whether to consider the input graph as directed.
        variant (str): Variant of EvolveGCN to use (options: 'H' or 'O')
        root_weight (bool): Whether to add a parametrized skip connection.
        cached (bool): Whether to cache normalized edge_weights.
        activation (str): Activation after each GCN layer.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 n_layers,
                 asymmetric_norm,
                 variant='H',
                 root_weight=False,
                 cached=False,
                 activation='relu'):
        super(EvolveGCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn_cells = nn.ModuleList()
        if variant == 'H':
            cell = EvolveGCNHCell
        elif variant == 'O':
            cell = EvolveGCNOCell
        else:
            raise NotImplementedError

        for i in range(self.n_layers):
            self.rnn_cells.append(cell(in_size=self.input_size if i == 0 else self.hidden_size,
                                       out_size=self.hidden_size,
                                       asymmetric_norm=asymmetric_norm,
                                       activation=activation,
                                       root_weight=root_weight,
                                       cached=cached))

    def forward(self, x, edge_index, edge_weight=None):
        # x : b t n f
        steps = x.size(1)
        h = [None, ] * len(self.rnn_cells)
        for t in range(steps):
            out = x[:, t]
            for c, cell in enumerate(self.rnn_cells):
                out, h[c] = cell(out, h[c], edge_index, edge_weight)
        return out
