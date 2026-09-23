"""Transforms for selecting node-induced subgraphs."""

import warnings

import torch
from torch_geometric.transforms import BaseTransform

from tsl.data import Data


class SubgraphTransform(BaseTransform):
    """Select an induced subgraph from a :class:`~tsl.data.Data` object.

    The transform accepts either a one-dimensional boolean node mask or a
    one-dimensional tensor of node indices. It also works when ``data.edge_index``
    is absent, in which case only node attributes are reduced.

    Note that this transform modifies the input data in-place, and does not return
    a new data object.

    Args:
        node_mask (Tensor, optional): Boolean nodes-to-keep mask or node indices.
            If :obj:`None`, return data unchanged.
            (default: :obj:`None`)
        add_node_idx (bool): Store the original selected node indices as the
            ``node_idx`` input attribute.
            (default: :obj:`False`)
    """

    def __init__(self, node_mask: torch.Tensor = None, add_node_idx: bool = False):
        if node_mask is not None and node_mask.dim() != 1:
            raise ValueError("node_mask must be a one-dimensional tensor")

        self.add_node_idx = add_node_idx
        self.num_nodes = None
        self.node_idx = None
        if node_mask is not None:
            if node_mask.dtype == torch.bool:
                self.num_nodes = node_mask.numel()
                full_node_idx = torch.arange(self.num_nodes, device=node_mask.device)
                self.node_idx = full_node_idx[node_mask]
            else:
                self.node_idx = node_mask

    def __repr__(self) -> str:
        if self.node_idx is None:
            return f'{self.__class__.__name__}(kept_nodes=all)'
        if self.num_nodes is None:
            return f'{self.__class__.__name__}(kept_nodes={len(self.node_idx)})'
        masked = self.num_nodes - len(self.node_idx)
        return '{cls}(kept_nodes={kept}, masked_nodes={masked})'.format(
            cls=self.__class__.__name__, kept=len(self.node_idx), masked=masked
        )

    def __call__(self, data: Data) -> Data:
        """Apply the legacy in-place transform."""
        # Don't shallow-copy the data otherwise the subgraph edit will be lost
        return self.forward(data)

    def forward(self, data: Data) -> Data:
        if self.node_idx is not None:
            data = data.subgraph_(self.node_idx)
        if self.add_node_idx:
            node_idx = (
                self.node_idx.clone()
                if self.node_idx is not None
                else torch.arange(data.num_nodes, device=data.x.device)
            )
            data.input['node_idx'] = node_idx
            data.pattern['node_idx'] = 'n'
        return data


class MaskedSubgraph(BaseTransform):
    """Deprecated transform deriving a subgraph from :attr:`data.mask`.

    A node is retained if at least one entry of its mask is :obj:`True`. The
    mask must have a pattern declaring its node dimension.

    Use :class:`SubgraphTransform` with an explicit node mask instead. A
    :class:`DeprecationWarning` is emitted when this transform is created. For
    backward compatibility, this deprecated transform still modifies ``data``
    in place.
    """

    def __init__(self):
        warnings.warn(
            "MaskedSubgraph is deprecated; use SubgraphTransform with an "
            "explicit node mask instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()

    def __call__(self, data: Data) -> Data:
        """Apply the legacy in-place transform."""
        return self.forward(data)

    def forward(self, data: Data) -> Data:
        if not data.has_mask:
            return data
        mask = data.mask.bool()
        pattern = data.pattern['mask'].replace(' ', '')
        node_dim = pattern.index('n')
        node_mask = mask.movedim(node_dim, 0).reshape(mask.size(node_dim), -1).any(1)
        node_idx = node_mask.nonzero().squeeze(-1)
        return data.subgraph_(node_idx)
