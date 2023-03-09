from .base_model import BaseModel
from .stgn import *  # noqa
from .temporal import *  # noqa

__all__ = ['BaseModel'] + stgn.classes + temporal.classes

classes = __all__
