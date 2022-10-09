from . import stgn, temporal
from .base_model import BaseModel
from .stgn import *
from .temporal import *

classes = ['BaseModel'] + stgn.classes + temporal.classes

__all__ = ['stgn', 'temporal'] + classes
