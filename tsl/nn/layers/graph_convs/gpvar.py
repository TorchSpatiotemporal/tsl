import math

import torch
from einops import rearrange
from torch_geometric.nn import MessagePassing

from .mixin import NormalizedAdjacencyMixin
from tsl.nn.utils import get_functional_activation

from torch import nn
import torch.nn.functional as F


class GraphPolyVAR(MessagePassing, NormalizedAdjacencyMixin):
    """
    Polynomial spatiotemporal graph filter.
    For x in (nodes, time_steps), the filter is given by
    for t in range(T):
        x[:, t] = eps[:, t] + sum_{p=1}^P  sum_{l=0}^L  psi[l, p] * S**l . x[:, t-p]
    where
     - eps is some noise component
     - S is a graph shift operator (GSO), and
     - psi are the filter coefficients with L-hop neighbors, and P steps in the past

    See Eq. 13, Isufi et al. "Forecasting time series with VARMA recursions
    on graphs." IEEE Transactions on Signal Processing 2019.
    """
    asymmetric_norm = False
    cached = False

    def __init__(self,
                 temporal_order,
                 spatial_order,
                 gcn_norm=False):
        super().__init__(aggr="add", node_dim=-2)

        self.temporal_order = temporal_order
        self.spatial_order = spatial_order
        self.weight = nn.Parameter(torch.Tensor(spatial_order + 1, temporal_order))
        self.gcn_norm = gcn_norm

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @classmethod
    def from_params(cls, filter_params, gcn_norm=False, activation='linear'):
        temporal_order = filter_params.shape[1]  # p
        spatial_order = filter_params.shape[0] - 1  # l
        model = cls(spatial_order=spatial_order,
                    temporal_order=temporal_order,
                    gcn_norm=gcn_norm)
        model.weight = torch.nn.Parameter(filter_params)
        return model

    def forward(self, x, edge_index, edge_weight=None):
        assert x.shape[-3] >= self.temporal_order  # time steps
        assert x.shape[-1] == 1  # node features

        # [b, t>=p, n, f=1] -> [b, n, p]
        out = rearrange(x[:, -self.temporal_order:], "... p n f -> ... n (p f)")

        if self.gcn_norm:
            edge_index, edge_weight = self.normalize_edge_index(x,
                                                                edge_index=edge_index,
                                                                edge_weight=edge_weight,
                                                                use_cached=False)

        # [b n p] -> [b n l]
        h = F.linear(out, self.weight)
        for l in range(1, self.spatial_order + 1):
            h[..., l:] = self.propagate(edge_index=edge_index, x=h[..., l:], norm=edge_weight)

        # [... n l] -> [... t=1 n f=1]
        out = h.sum(axis=-1).unsqueeze(-2).unsqueeze(-1)

        return out
