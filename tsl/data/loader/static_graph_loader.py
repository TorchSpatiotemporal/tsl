from typing import Optional

from torch.utils.data import (DataLoader,
                              Sampler,
                              RandomSampler,
                              SequentialSampler,
                              BatchSampler)

from ..spatiotemporal_dataset import SpatioTemporalDataset


class StaticGraphLoader(DataLoader):
    r"""A data loader for getting temporal graph signals of type
    :class:`~tsl.data.Batch` on a shared (static) topology.

    This loader exploits the efficient indexing of
    :class:`~tsl.data.SpatioTemporalDataset` to get multiple items at once,
    by leveraging on a :class:`~torch.utils.data.BatchSampler`.

    Args:
        dataset (SpatioTemporalDataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        sampler (Sampler or Iterable, optional): defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented. If specified, :attr:`shuffle` must not be specified.
        shuffle (bool, optional): If :obj:`True`, then data will be
            reshuffled at every epoch.
            (default: :obj:`False`)
        drop_last (bool, optional): If :obj:`True`, then drop the last
            incomplete batch if the dataset size is not divisible by the batch
            size (which will be smaller otherwise).
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(self, dataset: SpatioTemporalDataset,
                 batch_size: Optional[int] = 1,
                 sampler: Optional[Sampler] = None,
                 shuffle: bool = False,
                 drop_last: bool = False,
                 **kwargs):
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        if sampler is None:
            sampler_cls = RandomSampler if shuffle else SequentialSampler
            sampler = sampler_cls(dataset)
        sampler = BatchSampler(sampler,
                               batch_size=batch_size,
                               drop_last=drop_last)
        super().__init__(dataset,
                         sampler=sampler,
                         batch_size=None,
                         collate_fn=lambda x: x,
                         **kwargs)
