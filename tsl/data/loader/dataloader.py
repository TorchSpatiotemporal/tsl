from typing import Optional

from torch.utils import data

from ..spatiotemporal_dataset import SpatioTemporalDataset
from ..batch import Batch


class StaticGraphLoader(data.DataLoader):

    def __init__(self, dataset: SpatioTemporalDataset,
                 batch_size: Optional[int] = 1,
                 shuffle: bool = False,
                 num_workers: int = 0,
                 **kwargs):
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        super().__init__(dataset,
                         shuffle=shuffle,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=Batch.from_data_list,
                         **kwargs)
