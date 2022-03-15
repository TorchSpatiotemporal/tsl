import torch
from torch import Tensor
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor

from tsl.nn.utils.connectivity import normalize


class GraphConv(MessagePassing):
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

    def __init__(self, input_size: int, output_size: int, bias: bool = True, root_weight: bool = True, **kwargs):
        super(GraphConv, self).__init__(aggr="add", node_dim=-2)
        super().__init__(**kwargs)

        self.in_channels = input_size
        self.out_channels = output_size

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
        n = x.size(-2)
        out = self.lin(x)

        _, edge_weight = normalize(edge_index, edge_weight, dim=1, num_nodes=n)
        out = self.propagate(edge_index, x=out, edge_weight=edge_weight)

        if self.root_lin is not None:
            out += self.root_lin(x)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight) -> Tensor:
        return edge_weight.view(-1, 1) * x_j