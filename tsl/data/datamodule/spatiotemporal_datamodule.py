from typing import Optional, Mapping, Literal

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, Subset, DataLoader, RandomSampler, \
    SequentialSampler, BatchSampler

import tsl
from .splitters import Splitter
from ..spatiotemporal_dataset import SpatioTemporalDataset
from ...typing import Index


class SpatioTemporalDataModule(LightningDataModule):
    r"""Base :class:`~pytorch_lightning.core.LightningDataModule` for
    :class:`~tsl.data.SpatioTemporalDataset`.

    Args:
        dataset (SpatioTemporalDataset): The complete dataset.
        scalers (dict, optional): Named mapping of :class:`~tsl.data.preprocessing.scalers.Scaler`
            to be used for data rescaling after splitting. Every scaler is given
            as input the attribute of the dataset named as the scaler's key.
            If :obj:`None`, no scaling is performed.
            (default :obj:`None`)
        mask_scaling (bool): Whether to compute statistics for data scaler (if
            any) by considering only valid values (according to :obj:`dataset.mask`).
            (default :obj:`True`)
        splitter (Splitter, optional): :class:`~tsl.data.datamodule.splitters.Splitter` to
            be used for splitting :obj:`dataset` into training/validation/testing.
            (default :obj:`None`)
        batch_size (int): Size of batches for training/validation/testing splits.
            (default :obj:`32`)
        workers (int): Number of workers to use in DataLoaders.
            (default :obj:`0`)
        pin_memory (bool): Whether to enable pinned GPU memory for the train dataloader.
            (default :obj:`False`)
    """

    def __init__(self, dataset: SpatioTemporalDataset,
                 scalers: Optional[Mapping] = None,
                 mask_scaling: bool = True,
                 splitter: Optional[Splitter] = None,
                 batch_size: int = 32,
                 workers: int = 0,
                 pin_memory: bool = False):
        super(SpatioTemporalDataModule, self).__init__()
        self.torch_dataset = dataset
        # splitting
        self.splitter = splitter
        self.trainset = self.valset = self.testset = None
        # scaling
        if scalers is None:
            self.scalers = dict()
        else:
            self.scalers = scalers
        self.mask_scaling = mask_scaling
        # data loaders
        self.batch_size = batch_size
        self.workers = workers
        self.pin_memory = pin_memory

    def __getattr__(self, item):
        ds = self.__dict__.get('torch_dataset')
        if ds is not None and hasattr(ds, item):
            return getattr(ds, item)
        else:
            raise AttributeError(item)

    def __repr__(self):
        return "{}(train_len={}, val_len={}, test_len={}, " \
               "scalers=[{}], batch_size={})" \
            .format(self.__class__.__name__,
                    self.train_len, self.val_len, self.test_len,
                    ', '.join(self.scalers.keys()), self.batch_size)

    @property
    def trainset(self):
        return self._trainset

    @property
    def valset(self):
        return self._valset

    @property
    def testset(self):
        return self._testset

    @trainset.setter
    def trainset(self, value):
        self._add_set('train', value)

    @valset.setter
    def valset(self, value):
        self._add_set('val', value)

    @testset.setter
    def testset(self, value):
        self._add_set('test', value)

    @property
    def train_len(self):
        return len(self.trainset) if self.trainset is not None else None

    @property
    def val_len(self):
        return len(self.valset) if self.valset is not None else None

    @property
    def test_len(self):
        return len(self.testset) if self.testset is not None else None

    @property
    def train_slice(self):
        return self._train_slice if hasattr(self, '_train_slice') else None

    @property
    def val_slice(self):
        return self._val_slice if hasattr(self, '_val_slice') else None

    @property
    def test_slice(self):
        return self._test_slice if hasattr(self, '_test_slice') else None

    def _add_set(self, split_type, _set):
        assert split_type in ['train', 'val', 'test']
        split_type = '_' + split_type
        name = split_type + 'set'
        if _set is None or isinstance(_set, Dataset):
            setattr(self, name, _set)
        else:
            indices = _set
            assert isinstance(indices, Index.__args__), \
                f"type {type(indices)} of `{name}` is not a valid type. " \
                "It must be a dataset or a sequence of indices."
            _set = Subset(self.torch_dataset, indices)
            _slice = self.torch_dataset.expand_indices(_set.indices,
                                                       merge=True)
            setattr(self, name, _set)
            slice_name = split_type + '_slice'  # e.g. trainset > _train_slice
            setattr(self, slice_name, _slice)

    def setup(self, stage=None):

        # splitting
        if self.splitter is not None:
            self.splitter.split(self.torch_dataset)
            self.trainset = self.splitter.train_idxs
            self.valset = self.splitter.val_idxs
            self.testset = self.splitter.test_idxs

        for k, scaler, in self.scalers.items():
            train = getattr(self.torch_dataset, k)[self.train_slice]
            train_mask = None
            if k == 'data' and self.mask_scaling:
                if self.torch_dataset.mask is not None:
                    train_mask = self.torch_dataset.mask[self.train_slice]
            scaler = scaler.fit(train, mask=train_mask, keepdims=True)
            tsl.logger.info('Scaler for {}: {}'.format(k, scaler))
            self.torch_dataset.add_scaler(k, scaler)

    def get_dataloader(self, split: Literal['train', 'val', 'test'],
                       shuffle: bool = False,
                       batch_size: int = None):
        if split not in ['train', 'val', 'test']:
            raise ValueError("Argument `split` must be one of "
                             "'train', 'val', or 'test'.")
        dataset = getattr(self, f'{split}set')
        if dataset is None:
            return None
        pin_memory = self.pin_memory if split == 'train' else None
        sampler_cls = RandomSampler if shuffle else SequentialSampler
        sampler = BatchSampler(sampler_cls(dataset),
                               batch_size=batch_size or self.batch_size,
                               drop_last=split == 'train')
        return DataLoader(dataset,
                          sampler=sampler,
                          batch_size=None,
                          collate_fn=lambda x: x,
                          num_workers=self.workers,
                          pin_memory=pin_memory)

    def train_dataloader(self, shuffle=True, batch_size=None):
        return self.get_dataloader('train', shuffle, batch_size)

    def val_dataloader(self, shuffle=False, batch_size=None):
        return self.get_dataloader('val', shuffle, batch_size)

    def test_dataloader(self, shuffle=False, batch_size=None):
        return self.get_dataloader('test', shuffle, batch_size)

    @staticmethod
    def add_argparse_args(parser, **kwargs):
        parser.add_argument('--batch-size', type=int, default=64)
        parser.add_argument('--mask-scaling', type=bool, default=True)
        parser.add_argument('--workers', type=int, default=0)
        return parser
