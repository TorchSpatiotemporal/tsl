import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul

from tsl.nn.layers.graph_convs.mixin import NormalizedAdjacencyMixin
from tsl.nn.utils import get_functional_activation


class GraphConv(MessagePassing, NormalizedAdjacencyMixin):
    r"""A simple graph convolutional operator where the message function is a
    simple linear projection and aggregation a simple average. In other terms:

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1} \mathbf{\tilde{A}}
        \mathbf{X} \boldsymbol{\Theta} + \mathbf{b} .

    Args:
        input_size (int): Size of the input features.
        output_size (int): Size of the output features.
        bias (bool): If :obj:`False`, then the layer will not learn an additive
            bias vector.
            (default: :obj:`True`)
        norm (str): The normalization used for edges and edge weights. If
            :obj:`'mean'`, then edge weights are normalized as
            :math:`a_{j \rightarrow i} =  \frac{a_{j \rightarrow i}} {deg_{i}}`,
            other available options are: :obj:`'gcn'`, :obj:`'asym'` and
            :obj:`'none'`.
            (default: :obj:`'mean'`)
        root_weight (bool): If :obj:`True`, then add a linear layer for the root
            node itself (a skip connection).
            (default :obj:`True`)
        activation (str, optional): Activation function to be used, :obj:`None`
            for identity function (i.e., no activation).
            (default: :obj:`None`)
        cached (bool): If :obj:`True`, then cached the normalized edge weights
            computed in the first call.
            (default :obj:`False`)
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 norm: str = 'mean',
                 root_weight: bool = True,
                 activation: str = None,
                 cached: bool = False):
        super(GraphConv, self).__init__(aggr="add", node_dim=-2)

        self.in_channels = input_size
        self.out_channels = output_size
        self.norm = norm
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

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        out = self.lin(x)

        edge_index, edge_weight = self.normalize_edge_index(
            x,
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
        """"""
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        """"""
        # adj_t: SparseTensor [nodes, nodes]
        # x: [(batch,) nodes, channels]
        return matmul(adj_t, x, reduce=self.aggr)
