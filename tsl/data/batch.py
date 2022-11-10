from typing import (Optional, Any, Union, List, Mapping, Callable)

import torch
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.data.collate import collate
from torch_geometric.data.separate import separate

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
    r"""A batch of :class:`tsl.data.Data` objects for multiple spatiotemporal
    graphs sharing the same topology.

    The batch object extends :class:`~tsl.data.Data`, thus preserving
    all its functionalities.

    Args:
        input (Mapping, optional): Named mapping of :class:`~torch.Tensor` to be
            used as input to the model.
            (default: :obj:`None`)
        target (Mapping, optional): Named mapping of :class:`~torch.Tensor` to be
            used as target of the task.
            (default: :obj:`None`)
        mask (Tensor, optional): The optional mask associated with the target.
            (default: :obj:`None`)
        transform (Mapping, optional): Named mapping of
            :class:`~tsl.data.preprocessing.Scaler` associated with entries in
            :attr:`input` or :attr:`output`.
            (default: :obj:`None`)
        pattern (Mapping, optional): Map of the pattern of each entry in
            :attr:`input` or :attr:`output`.
            (default: :obj:`None`)
        size (int, optional): The batch size, i.e., the number of spatiotemporal
            graphs in the batch. The different samples in the batch share all
            the same topology, such that there is (at most) only one
            :obj:`edge_index` and :obj:`edge_weight`. If :obj:`None`, then the
            batch size is inferred from data (if possible).
            (default: :obj:`None`)
        **kwargs: Any keyword argument for :class:`~torch_geometric.data.Data`.
    """
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
        """The batch size, i.e., the number of spatiotemporal graphs in the
        batch."""
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


def collate_scaler_params(values, cat_dim=None, batch_index=None):
    elem = values[0]
    # Stack scaler params on new batch dimension.
    if cat_dim is None:
        value = torch.stack(values, dim=0)
    # Concatenate a list of `torch.Tensor` along the `cat_dim`.
    else:
        value = torch.cat(values, dim=cat_dim)

    if batch_index is not None and (cat_dim is None or elem.size(cat_dim) == 1):
        value = value.index_select(cat_dim or 0, batch_index)
        return value, True

    return value, False


def separate_scaler_params(value, slices, idx, is_repeated, cat_dim=None):
    if not is_repeated:
        start, end = slices[idx], slices[idx + 1]
        out = value.narrow(cat_dim or 0, start, end - start)
    else:
        out = value.index_select(cat_dim or 0, slices[idx])
    return out


def collate_scalers(data_list, batch_index=None):
    elem = data_list[0]
    transform = data_list[0].transform

    out = dict()
    for key, scaler in transform.items():
        cat_dim = elem.__cat_dim__(key, elem[key])
        # collate each param in the scaler
        params, rep_params = dict(), dict()
        for param in scaler.params():
            param_list = [getattr(item.transform[key], param)
                          for item in data_list]
            value, is_repeated = collate_scaler_params(param_list,
                                                       cat_dim=cat_dim,
                                                       batch_index=batch_index)
            params[param] = value
            rep_params[param] = is_repeated

        # set pattern of new collated scaler
        if scaler.pattern is not None:
            pattern = scaler.pattern
        else:
            pattern = elem.pattern.get(key)
        if cat_dim is None and pattern is not None:
            pattern = 'n ' + pattern

        out[key] = ScalerModule(**params, pattern=pattern)
        out[key]._repeated_params = rep_params

    return out


def get_example_scalers(batch, idx):
    transform = batch.transform

    out = dict()
    for key, scaler in transform.items():
        if not hasattr(scaler, '_repeated_params'):
            raise RuntimeError("Cannot separate 'ScalerModule' because "
                               "was not created via 'collate_scalers()'.")
        cat_dim = batch.__cat_dim__(key, batch[key])
        # collate each param in the scaler
        params = {
            param: separate_scaler_params(getattr(scaler, param),
                                          slices=batch._slice_dict[key],
                                          idx=idx, cat_dim=cat_dim,
                                          is_repeated=scaler._repeated_params[
                                              param])
            for param in scaler.params()
        }

        # set pattern of new collated scaler
        if scaler.pattern is not None:
            pattern = scaler.pattern
        else:
            pattern = batch.pattern.get(key)
        if cat_dim is None and pattern is not None:
            pattern = pattern[2:]

        out[key] = ScalerModule(**params, pattern=pattern)
    return out


class DisjointBatch(PyGBatch):
    r"""A data object describing a batch of graphs as one big (disconnected)
    graph.

    Inherits from :class:`tsl.data.Data`. In addition, single graphs can be
    identified via the assignment vector :obj:`batch`, which maps each node to
    its respective graph identifier.
    """

    @classmethod
    def from_data_list(cls, data_list: List[Data],
                       follow_batch: Optional[List[str]] = None,
                       exclude_keys: Optional[List[str]] = None,
                       graph_attributes: Optional[List[str]] = None):
        r"""Constructs a :class:`~tsl.data.DisjointBatch` object from a list of
        :class:`~tsl.data.Data` objects.

        The assignment vector :obj:`batch` is created on the fly.
        In addition, creates assignment vectors for each key in
        :obj:`follow_batch`.
        Will exclude any keys given in :obj:`exclude_keys`.

        Args:
            data_list (list): The list of :class:`tsl.data.Data` objects.
            follow_batch (list, optional): Create an assignment vector for each
                key in :attr:`follow_batch`.
                (default: :obj:`None`)
            exclude_keys (list, optional): Exclude the keys in
                :attr:`exclude_keys` from collate.
                (default: :obj:`None`)
            graph_attributes: Keys in :attr:`graph_attributes` with no node
                dimension will be added to the batch as graph attributes, i.e.,
                the tensors will be stacked on a new dimension (the first one).
                Note that all graph attributes indexed by a key which is not in
                this list are repeated along a new node dimension (the second
                one, if the attribute is time-varying, otherwise the first one).
                (default: :obj:`None`)
        """

        if graph_attributes is None:
            graph_attributes = []
        if exclude_keys is None:
            exclude_keys = []
        exclude_keys.append('transform')

        batch, slice_dict, inc_dict = collate(
            cls,
            data_list=data_list,
            increment=True,
            add_batch=not isinstance(data_list[0], Batch),
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        scalers = collate_scalers(data_list, batch_index=batch.batch)
        batch.transform.update(scalers)

        # repeat node-invariant item element along node dimension to not lose
        # coupling with item in the batch
        repeated_keys = []
        for key, value in batch:
            if batch.__cat_dim__(key, value) is None:
                if key not in graph_attributes:
                    batch[key] = value[batch.batch]
                    slice_dict[key] = batch.ptr
                    repeated_keys.append(key)
                if key in batch.pattern:
                    dims = batch.pattern[key].split(' ')
                    # '... -> b ...'
                    if key in graph_attributes:
                        dims.insert(0, 'b')
                    # 'n t ... -> t n ...'
                    elif 't' in dims:
                        batch[key] = torch.transpose(batch[key], 0, 1). \
                            contiguous()
                        dims.insert(1, 'n')
                    # '... -> n ...'
                    else:
                        dims.insert(0, 'n')
                    batch.pattern[key] = ' '.join(dims)

        batch._num_graphs = len(data_list)
        batch._slice_dict = slice_dict
        batch._inc_dict = inc_dict
        batch._repeated_keys = repeated_keys

        return batch

    def get_example(self, idx: int) -> Data:
        r"""Gets the :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object at index :obj:`idx`.
        The :class:`~torch_geometric.data.Batch` object must have been created
        via :meth:`from_data_list` in order to be able to reconstruct the
        initial object."""

        if not hasattr(self, '_slice_dict') and \
                not hasattr(self, '_repeated_keys'):
            raise RuntimeError(
                ("Cannot reconstruct 'Data' object from 'DynamicBatch' because "
                 "'Batch' was not created via 'DynamicBatch.from_data_list()'"))

        data = separate(
            cls=self.__class__.__bases__[-1],
            batch=self,
            idx=idx,
            slice_dict=self._slice_dict,
            inc_dict=self._inc_dict,
            decrement=True,
        )

        scalers = get_example_scalers(self, idx=idx)
        data.transform.update(scalers)

        for key in self._repeated_keys:
            data[key] = data[key][0]
            if key in data.pattern:
                # was 'n ' + previous_pattern, remove 'n '
                data.pattern[key] = data.pattern[key][2:]

        return data
