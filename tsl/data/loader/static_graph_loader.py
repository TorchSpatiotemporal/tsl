from typing import Optional

from torch.utils.data import DataLoader

from ..spatiotemporal_dataset import SpatioTemporalDataset


class StaticGraphLoader(DataLoader):
    r"""A data loader for getting temporal graph signals of type
    :class:`~tsl.data.Batch` on a shared (static) topology.

    This loader exploits the efficient indexing of
    :class:`~tsl.data.SpatioTemporalDataset` to get multiple items at once,
    by using a :class:`torch.utils.data.BatchSampler`.

    Args:
        dataset (SpatioTemporalDataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
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
                 shuffle: bool = False,
                 drop_last: bool = False,
                 **kwargs):
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        super().__init__(dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         drop_last=drop_last,
                         collate_fn=lambda x: x,
                         **kwargs)

    @property
    def _auto_collation(self):
        return False

    @property
    def _index_sampler(self):
        return self.batch_sampler
