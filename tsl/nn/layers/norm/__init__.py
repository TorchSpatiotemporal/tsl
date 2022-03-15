from .layer_norm import LayerNorm
from .instance_norm import InstanceNorm
from .batch_norm import BatchNorm
from .norm import Norm

__all__ = [
    'Norm',
    'LayerNorm',
    'InstanceNorm',
    'BatchNorm'
]

classes = __all__
