import math
from typing import *

import torch
from torch import Tensor

from tsl.ops.connectivity import normalize_connectivity


class NormalizedAdjacencyMixin:
    r"""
    Mixin for layers which use a normalized adjiacency matrix to propagate messages
    """
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]] = None
    cached: bool = False
    asymmetric_norm: bool = False

    def _normalize_edge_index(self, x, edge_index, edge_weight, use_cached):
        if use_cached:
            if self._cached_edge_index is None:
                return self._normalize_edge_index(x, edge_index, edge_weight, False)
            return self._cached_edge_index
        edge_index, edge_weight = normalize_connectivity(edge_index,
                                                         edge_weight,
                                                         symmetric=not self.asymmetric_norm,
                                                         add_self_loops=not self.asymmetric_norm,
                                                         num_nodes=x.size(-2))
        if self.cached:
            self._cached_edge_index = (edge_index, edge_weight)
        return edge_index, edge_weight
