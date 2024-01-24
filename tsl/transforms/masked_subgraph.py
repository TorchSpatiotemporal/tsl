from torch_geometric.transforms import BaseTransform

from tsl.data import Data


class MaskedSubgraph(BaseTransform):
    """Reduce graph in :attr:`sample` removing masked nodes."""

    def __call__(self, data: Data) -> Data:
        if not data.has_mask:
            return data
        live_nodes = data.mask.any(0).any(-1)
        node_index = live_nodes.nonzero().squeeze(dim=1)
        return data.subgraph_(node_index)
