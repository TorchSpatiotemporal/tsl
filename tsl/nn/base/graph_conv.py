import torch
from torch import Tensor
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul

from tsl.nn.layers.graph_convs.mixin import NormalizedAdjacencyMixin
from tsl.nn.utils import get_functional_activation
from tsl.ops.connectivity import normalize_connectivity


class GraphConv(MessagePassing, NormalizedAdjacencyMixin):
    r"""A simple graph convolutional operator where the message function is a simple linear projection and aggregation
    a simple average. In other terms:

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1} \mathbf{A} \mathbf{X} \boldsymbol{\Theta}

    Args:
        input_size (int): Size of the input features.
        output_size (int): Size of each output features.
        add_self_loops (bool, optional): If set to :obj:`True`, will add
            self-loops to the input graph. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, input_size: int,
                 output_size: int,
                 bias: bool = True,
                 asymmetric_norm: bool = True,
                 root_weight: bool = True,
                 activation='linear',
                 cached: bool = False,
                 **kwargs):
        super(GraphConv, self).__init__(aggr="add", node_dim=-2)
        super().__init__(**kwargs)

        self.in_channels = input_size
        self.out_channels = output_size
        self.asymmetric_norm = asymmetric_norm
        self.cached = cached
        self.activation = get_functional_activation(activation)

        self.lin = nn.Linear(input_size, output_size, bias=False)

        if root_weight:
            self.root_lin = nn.Linear(input_size, output_size, bias=False)
        else:
            self.register_parameter('root_lin', None)

        if bias:
            self.bias = Parameter(torch.Tensor(output_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.root_lin is not None:
            self.root_lin.reset_parameters()
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        out = self.lin(x)

        edge_index, edge_weight = self.normalize_edge_index(x,
                                                            edge_index=edge_index,
                                                            edge_weight=edge_weight,
                                                            use_cached=self.cached)

        out = self.propagate(edge_index, x=out, edge_weight=edge_weight)

        if self.root_lin is not None:
            out += self.root_lin(x)

        if self.bias is not None:
            out += self.bias

        return self.activation(out)

    def message(self, x_j: Tensor, edge_weight) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        """"""
        # adj_t: SparseTensor [nodes, nodes]
        # x: [(batch,) nodes, channels]
        return matmul(adj_t, x, reduce=self.aggr)
