from . import splitters
from .spatiotemporal_datamodule import SpatioTemporalDataModule
from .splitters import *

datamodule_classes = ['SpatioTemporalDataModule']
splitter_classes = splitters.__all__

__all__ = datamodule_classes + splitter_classes
