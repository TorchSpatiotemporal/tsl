from .base_model import BaseModel
from .stgn import *
from .temporal import *

__all__ = ['BaseModel'] + stgn.classes + temporal.classes  # noqa

classes = __all__
