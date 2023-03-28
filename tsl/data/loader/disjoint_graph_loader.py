from functools import partial
from typing import List, Union

from torch.utils import data

from ..batch import DisjointBatch
from ..data import Data
from ..spatiotemporal_dataset import SpatioTemporalDataset


class DisjointGraphLoader(data.DataLoader):
    r"""A data loader for merging temporal graph signals of type
    :class:`~tsl.data.Data` with different topologies.

    Args:
        dataset (SpatioTemporalDataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If :obj:`True`, then data will be
            reshuffled at every epoch.
            (default: :obj:`False`)
        force_batch (bool): If :obj:`True`, then add add dummy batch
            dimension for time-varying elements.
            (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list.
            (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`num_workers`.
    """

    def __init__(
        self,
        dataset: Union[SpatioTemporalDataset, List[Data]],
        batch_size: int = 1,
        shuffle: bool = False,
        force_batch: bool = False,
        follow_batch: List[str] = None,
        exclude_keys: List[str] = None,
        **kwargs
    ):
        if follow_batch is None:
            follow_batch = []

        if exclude_keys is None:
            exclude_keys = []

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning...
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        collate_fn = partial(
            DisjointBatch.from_data_list,
            force_batch=force_batch,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        super().__init__(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=collate_fn,
            **kwargs
        )
