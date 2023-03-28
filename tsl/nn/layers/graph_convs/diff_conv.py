from typing import List

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor
from torch_sparse import cat as cat_sparse
from torch_sparse import matmul

from tsl.ops.connectivity import asymmetric_norm, transpose


def diff_conv_gso(edge_index: Tensor,
                  edge_weight: OptTensor = None,
                  k: int = 2,
                  num_nodes: int = None,
                  add_backward: bool = True):
    if isinstance(edge_index, Tensor):
        # transpose
        col, row = edge_index
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        adj = SparseTensor(row=row,
                           col=col,
                           value=edge_weight,
                           sparse_sizes=(num_nodes, num_nodes))
    elif isinstance(edge_index, SparseTensor):
        adj = edge_index
    else:
        raise RuntimeError("Edge index must be (edge_index, edge_weight) "
                           "tuple or SparseTensor.")

    # normalize
    adj0, _ = asymmetric_norm(adj, dim=1, num_nodes=num_nodes)

    out = [adj0]
    for _ in range(k - 1):
        out.append(adj0 @ out[-1])

    if add_backward:
        out_bwd = DiffConv.gso(adj.t(),
                               k=k,
                               num_nodes=num_nodes,
                               add_backward=False)
        return cat_sparse(out + [out_bwd], dim=0)

    return cat_sparse(out, dim=0)


class DiffConv(MessagePassing):
    r"""The Diffusion Convolution Layer from the paper `"Diffusion Convolutional
    Recurrent Neural Network: Data-Driven Traffic Forecasting"
    <https://arxiv.org/abs/1707.01926>`_ (Li et al., ICLR 2018).

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        k (int): Filter size :math:`K`.
        root_weight (bool): If :obj:`True`, then add a filter also for the
            :math:`0`-order neighborhood (i.e., the root node itself).
            (default :obj:`True`)
        add_backward (bool): If :obj:`True`, then additional :math:`K` filters
            are learnt for the transposed connectivity.
            (default :obj:`True`)
        bias (bool, optional): If :obj:`True`, add a trainable additive bias.
            (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 k: int,
                 root_weight: bool = True,
                 add_backward: bool = True,
                 bias: bool = True):
        super(DiffConv, self).__init__(aggr="add", node_dim=-2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        self.root_weight = root_weight
        self.add_backward = add_backward

        n_filters = k
        if add_backward:
            n_filters *= 2
        if root_weight:
            n_filters += 1

        self.filters = nn.Linear(in_channels * n_filters,
                                 out_channels,
                                 bias=bias)

        self._support = None
        self.reset_parameters()

    @staticmethod
    def compute_support_index(edge_index: Adj,
                              edge_weight: OptTensor = None,
                              num_nodes: int = None,
                              add_backward: bool = True) -> List:
        """Normalize the connectivity weights and (optionally) add normalized
        backward weights."""
        norm_edge_index, \
            norm_edge_weight = asymmetric_norm(edge_index, edge_weight,
                                               dim=1, num_nodes=num_nodes)
        # Add backward matrices
        if add_backward:
            return [(norm_edge_index, norm_edge_weight)] + \
                DiffConv.compute_support_index(transpose(edge_index),
                                               edge_weight=edge_weight,
                                               num_nodes=num_nodes,
                                               add_backward=False)
        # Normalize
        return [(norm_edge_index, norm_edge_weight)]

    def reset_parameters(self):
        self.filters.reset_parameters()
        self._support = None

    def message(self, x_j: Tensor, weight: Tensor) -> Tensor:
        """"""
        # x_j: [batch, edges, channels]
        return weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        """"""
        # adj_t: SparseTensor [nodes, nodes]
        # x: [(batch,) nodes, channels]
        return matmul(adj_t, x, reduce=self.aggr)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, cache_support: bool = False) \
            -> Tensor:
        """"""
        # x: [batch, (steps), nodes, nodes]
        n = x.size(-2)
        if self._support is None:
            support = self.compute_support_index(
                edge_index,
                edge_weight,
                add_backward=self.add_backward,
                num_nodes=n)
            if cache_support:
                self._support = support
        else:
            support = self._support

        out = []
        if self.root_weight:
            out += [x]

        for sup_index, sup_weights in support:
            x_sup = x
            for _ in range(self.k):
                x_sup = self.propagate(sup_index, x=x_sup, weight=sup_weights)
                out += [x_sup]

        out = torch.cat(out, -1)
        return self.filters(out)
