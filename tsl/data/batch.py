from typing import Callable
from typing import (Optional, Any, Union, List, Mapping)

from torch import Tensor
from torch.utils.data.dataloader import default_collate

from .data import Data
from .preprocessing import ScalerModule


def _collate_scaler_modules(batch: List[Mapping[str, Any]]):
    transform = batch[0]
    for k, v in transform.items():
        # scaler params are supposed to be the same for all elements in
        # minibatch, just add a fake, 1-sized, batch dimension
        transform[k] = ScalerModule(bias=transform[k].bias[None],
                                    scale=transform[k].scale[None])
        if transform[k].pattern is not None:
            transform[k].pattern = 'b ' + transform[k].pattern
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
            if 't' in pattern:
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

    out.__dict__['batch_size'] = len(batch)
    return out


class Batch(Data):
    _collate_fn: Callable = static_graph_collate

    def __init__(self, input: Optional[Mapping] = None,
                 target: Optional[Mapping] = None,
                 mask: Optional[Tensor] = None,
                 transform: Optional[Mapping] = None,
                 pattern: Optional[Mapping] = None,
                 size: Optional[int] = None,
                 **kwargs):
        super(Batch, self).__init__(input=input,
                                    target=target,
                                    mask=mask,
                                    transform=transform,
                                    pattern=pattern,
                                    **kwargs)
        self._batch_size = size

    @property
    def batch_size(self) -> int:
        if self._batch_size is not None:
            return self._batch_size
        if self.pattern is not None:
            for key, pattern in self.pattern.items():
                if pattern.startswith('b'):
                    return self[key].size(0)

    @classmethod
    def from_data_list(cls, data_list: List[Data]):
        r"""Constructs a :class:`~tsl.data.Batch` object from a Python list of
         :class:`~tsl.data.Data`, representing temporal signals on static
         graphs."""

        batch = cls._collate_fn(data_list, cls)

        batch.__dict__['batch_size'] = len(data_list)

        return batch
