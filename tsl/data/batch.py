from typing import Callable
from typing import (Optional, Any, Union, List, Mapping)

import torch
from torch import Tensor
from torch.utils.data.dataloader import default_collate

from .data import Data


def _collate_scaler_modules(batch: List[Mapping[str, Any]]):
    transform = batch[0]
    for k, v in transform.items():
        # scaler params are supposed to be the same for all elements in
        # minibatch, just add a fake, 1-sized, batch dimension
        if v.bias is not None:
            transform[k].bias = transform[k].bias[None]
        if v.scale is not None:
            transform[k].scale = transform[k].scale[None]
        if v.trend is not None:
            trend = torch.stack([b[k].trend for b in batch], 0)
            transform[k].trend = trend
    return transform


def static_graph_collate(batch: List[Data], cls: Optional[type] = None) -> Data:
    # collate subroutine
    def _collate(items: List[Union[Tensor, Mapping[str, Any]]], key: str,
                 pattern: str):
        if key == 'transform':
            return _collate_scaler_modules(items), None
        # if key.startswith('edge_'):
        #     return items[0]
        if pattern is not None:
            if 's' in pattern:
                return default_collate(items), 'b ' + pattern
            return items[0], pattern
        return default_collate(items), None

    # collate all sample-wise elements
    elem = batch[0]
    if cls is None:
        cls = elem.__class__
    out = cls()
    out = out.stores_as(elem)
    for k in elem.keys:
        pattern = elem.pattern.get(k)
        out[k], pattern = _collate([b[k] for b in batch], k, pattern)
        if pattern is not None:
            out.pattern[k] = pattern
    return out


class Batch(Data):
    _collate_fn: Callable = static_graph_collate

    @classmethod
    def from_data_list(cls, data_list: List[Data]):
        r"""Constructs a :class:`~tsl.data.Batch` object from a Python list of
         :class:`~tsl.data.Data`, representing temporal signals on static
         graphs."""

        batch = cls._collate_fn(data_list, cls)

        batch.__dict__['batch_size'] = len(data_list)

        return batch
