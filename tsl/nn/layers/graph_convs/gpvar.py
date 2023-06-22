import math
from typing import Optional

import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj
from torch_sparse import SparseTensor, matmul

from .mixin import NormalizedAdjacencyMixin


class GraphPolyVAR(MessagePassing, NormalizedAdjacencyMixin):
    r"""Polynomial spatiotemporal graph filter from the paper `"Forecasting time
    series with VARMA recursions on graphs."
    <https://arxiv.org/abs/1810.08581>`_ (Isufi et al., IEEE Transactions on
    Signal Processing 2019).

    .. math::

        \mathbf{X}_t = \sum_{p=1}^{P} \sum_{l=1}^{L} \Theta_{p,l} \cdot
        \mathbf{\tilde{A}}^{l-1} \mathbf{X}_{t-p}

    where
     - :math:`\mathbf{\tilde{A}}` is a graph shift operator (GSO);
     - :math:`\Theta \in \mathbb{R}^{P \times L}` are the filter coefficients
       accounting for up to :math:`L`-hop neighbors and :math:`P` time steps
       in the past.

    Args:
        temporal_order (int): The filter temporal order :math:`P`.
        spatial_order (int): The filter spatial order :math:`L`.
        norm (str): The normalization used for edges and edge weights. The
            available options are: :obj:`'gcn'`, :obj:`'asym'` and
            :obj:`'none'`.
            (default: :obj:`'none'`)
        cached (bool): If :obj:`True`, then cache the normalized edge weights
            computed in the first call.
            (default :obj:`False`)
    """
    norm = 'none'
    cached = False

    def __init__(self,
                 temporal_order: int,
                 spatial_order: int,
                 norm: str = 'none',
                 cached: bool = False):
        super().__init__(aggr="add", node_dim=-2)
        self.temporal_order = temporal_order
        self.spatial_order = spatial_order
        self.norm = norm
        self.cached = cached

        self.weight = nn.Parameter(Tensor(spatial_order + 1, temporal_order))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @classmethod
    def from_params(cls,
                    filter_params: Tensor,
                    norm: str = 'none',
                    cached: bool = False):
        temporal_order = filter_params.shape[1]  # p
        spatial_order = filter_params.shape[0] - 1  # l
        model = cls(spatial_order=spatial_order,
                    temporal_order=temporal_order,
                    norm=norm,
                    cached=cached)
        model.weight.data.copy_(filter_params)
        return model

    def message(self, x_j: Tensor, weight: Optional[Tensor] = None) -> Tensor:
        """"""
        # x_j: [*, edges, channels]
        if weight is None:
            return x_j
        return weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        """"""
        # adj_t: SparseTensor [nodes, nodes]
        # x: [*, nodes, channels]
        return matmul(adj_t, x, reduce=self.aggr)

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_weight: Optional[Tensor] = None):
        """"""
        assert x.shape[-3] >= self.temporal_order  # time steps
        assert x.shape[-1] == 1  # node features

        # [b, t>=p, n, f=1] -> [b, n, p]
        out = rearrange(x[:, -self.temporal_order:],
                        "... p n f -> ... n (p f)")

        if self.norm != 'none':
            edge_index, edge_weight = self.normalize_edge_index(
                x,
                edge_index=edge_index,
                edge_weight=edge_weight,
                use_cached=self.cached)

        # [b n p] -> [b n l]
        h = F.linear(out, self.weight)
        for i in range(1, self.spatial_order + 1):
            h[..., i:] = self.propagate(edge_index=edge_index,
                                        x=h[..., i:],
                                        weight=edge_weight)

        # [... n l] -> [... t=1 n f=1]
        out = h.sum(axis=-1).unsqueeze(-2).unsqueeze(-1)

        return out
