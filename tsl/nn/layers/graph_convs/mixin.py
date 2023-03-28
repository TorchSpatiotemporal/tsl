from typing import Optional, Tuple

from torch import Tensor

from tsl.ops.connectivity import normalize_connectivity


class NormalizedAdjacencyMixin:
    """Mixin for layers which use a normalized adjacency matrix to propagate
     messages."""
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]] = None
    cached: bool = False
    norm: str = 'none'

    def normalize_edge_index(self, x, edge_index, edge_weight, use_cached):
        if use_cached:
            if self._cached_edge_index is None:
                return self.normalize_edge_index(x, edge_index, edge_weight,
                                                 False)
            return self._cached_edge_index

        edge_index, edge_weight = normalize_connectivity(edge_index,
                                                         edge_weight,
                                                         norm=self.norm,
                                                         num_nodes=x.size(-2))
        if self.cached:
            self._cached_edge_index = (edge_index, edge_weight)
        return edge_index, edge_weight
