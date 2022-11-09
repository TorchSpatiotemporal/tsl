from functools import partial
from typing import Optional, Union, List

import torch
from torch.utils import data
from torch_geometric.data import Batch
from torch_geometric.data.collate import collate
from torch_geometric.data.separate import separate

from .. import Data
from ..preprocessing import ScalerModule
from ..spatiotemporal_dataset import SpatioTemporalDataset


def collate_scaler_params(values, cat_dim=None, batch_index=None):
    elem = values[0]
    # Stack scaler params on new batch dimension.
    if cat_dim is None:
        value = torch.stack(values, dim=0)
    # Concatenate a list of `torch.Tensor` along the `cat_dim`.
    else:
        value = torch.cat(values, dim=cat_dim)
        if elem.size(cat_dim) == 1 and batch_index is not None:
            value = value.index_select(cat_dim, batch_index)
    return value


def separate_scaler_params(value, slices, idx, cat_dim=None):
    if cat_dim is None:
        out = value[idx]
    else:
        start, end = slices[idx], slices[idx + 1]
        out = value.narrow(cat_dim or 0, start, end - start)
    return out


def collate_scalers(data_list, batch_index=None):
    elem = data_list[0]
    transform = data_list[0].transform
    for key in transform.keys():
        cat_dim = elem.__cat_dim__(key, elem[key])
        # collate each param in the scaler
        params = {
            param: collate_scaler_params(
                [getattr(item.transform[key], param) for item in data_list],
                cat_dim=cat_dim, batch_index=batch_index)
            for param in transform[key].params()
        }

        # set pattern of new collated scaler
        if transform[key].pattern is not None:
            pattern = transform[key].pattern
        else:
            pattern = elem.pattern.get(key)
        if cat_dim is None and pattern is not None:
            pattern = 'b ' + pattern

        transform[key] = ScalerModule(**params,
                                      pattern=pattern)
    return transform


def get_example_scalers(batch, idx):
    transform = batch.transform
    for key, scaler in transform.items():
        cat_dim = batch.__cat_dim__(key, batch[key])
        # collate each param in the scaler
        params = {
            param: separate_scaler_params(getattr(scaler, param),
                                          slices=batch._slice_dict[key],
                                          idx=idx, cat_dim=cat_dim)
            for param in scaler.params()
        }

        # set pattern of new collated scaler
        if scaler.pattern is not None:
            pattern = scaler.pattern
        else:
            pattern = batch.pattern.get(key)
        if cat_dim is None and pattern is not None:
            pattern = pattern[2:]

        transform[key] = ScalerModule(**params,
                                      pattern=pattern)
    return transform


class DynamicBatch(Batch):
    r"""A data object describing a batch of graphs as one big (disconnected)
    graph.
    Inherits from :class:`torch_geometric.data.Data` or
    :class:`torch_geometric.data.HeteroData`.
    In addition, single graphs can be identified via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    @classmethod
    def from_data_list(cls, data_list: List[Data],
                       follow_batch: Optional[List[str]] = None,
                       exclude_keys: Optional[List[str]] = None):
        r"""Constructs a :class:`~torch_geometric.data.Batch` object from a
        Python list of :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` objects.
        The assignment vector :obj:`batch` is created on the fly.
        In addition, creates assignment vectors for each key in
        :obj:`follow_batch`.
        Will exclude any keys given in :obj:`exclude_keys`."""

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

        for key in batch.pattern:
            if batch.__cat_dim__(key, batch[key]) is None:
                batch.pattern[key] = 'b ' + batch.pattern[key]

        batch._num_graphs = len(data_list)
        batch._slice_dict = slice_dict
        batch._inc_dict = inc_dict

        return batch

    def get_example(self, idx: int) -> Data:
        r"""Gets the :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object at index :obj:`idx`.
        The :class:`~torch_geometric.data.Batch` object must have been created
        via :meth:`from_data_list` in order to be able to reconstruct the
        initial object."""

        if not hasattr(self, '_slice_dict'):
            raise RuntimeError(
                ("Cannot reconstruct 'Data' object from 'Batch' because "
                 "'Batch' was not created via 'Batch.from_data_list()'"))

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

        for key in data.pattern:
            if data.pattern[key].startswith('b'):
                data.pattern[key] = data.pattern[key][2:]

        return data


class DynamicGraphLoader(data.DataLoader):
    r"""A data loader for merging temporal graph signals of type
    :class:`~tsl.data.Data` with different topologies.

    Args:
        dataset (SpatioTemporalDataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If :obj:`True`, then data will be
            reshuffled at every epoch.
            (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list.
            (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`num_workers`.
    """

    def __init__(self, dataset: Union[SpatioTemporalDataset, List[Data]],
                 batch_size: int = 1,
                 shuffle: bool = False,
                 follow_batch: List[str] = None,
                 exclude_keys: List[str] = None,
                 **kwargs):

        if follow_batch is None:
            follow_batch = []

        if exclude_keys is None:
            exclude_keys = []

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning...
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        collate_fn = partial(DynamicBatch.from_data_list,
                             follow_batch=follow_batch,
                             exclude_keys=exclude_keys)

        super().__init__(dataset,
                         shuffle=shuffle,
                         batch_size=batch_size,
                         collate_fn=collate_fn,
                         **kwargs)
