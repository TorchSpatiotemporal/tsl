import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch_geometric.nn import MessagePassing

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
    """
    norm = 'none'
    cached = False

    def __init__(self, temporal_order, spatial_order, gcn_norm=False):
        super().__init__(aggr="add", node_dim=-2)

        self.temporal_order = temporal_order
        self.spatial_order = spatial_order
        self.weight = nn.Parameter(
            torch.Tensor(spatial_order + 1, temporal_order))
        if gcn_norm:
            self.norm = 'gcn'

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @classmethod
    def from_params(cls, filter_params, gcn_norm=False):
        temporal_order = filter_params.shape[1]  # p
        spatial_order = filter_params.shape[0] - 1  # l
        model = cls(spatial_order=spatial_order,
                    temporal_order=temporal_order,
                    gcn_norm=gcn_norm)
        model.weight.data.copy_(filter_params)
        return model

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        assert x.shape[-3] >= self.temporal_order  # time steps
        assert x.shape[-1] == 1  # node features

        # [b, t>=p, n, f=1] -> [b, n, p]
        out = rearrange(x[:, -self.temporal_order:],
                        "... p n f -> ... n (p f)")

        if self.gcn_norm:
            edge_index, edge_weight = self.normalize_edge_index(
                x,
                edge_index=edge_index,
                edge_weight=edge_weight,
                use_cached=False)

        # [b n p] -> [b n l]
        h = F.linear(out, self.weight)
        for i in range(1, self.spatial_order + 1):
            h[..., i:] = self.propagate(edge_index=edge_index,
                                        x=h[..., i:],
                                        norm=edge_weight)

        # [... n l] -> [... t=1 n f=1]
        out = h.sum(axis=-1).unsqueeze(-2).unsqueeze(-1)

        return out
