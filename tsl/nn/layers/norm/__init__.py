from .batch_norm import BatchNorm
from .instance_norm import InstanceNorm
from .layer_norm import LayerNorm
from .norm import Norm

__all__ = [
    'Norm',
    'LayerNorm',
    'InstanceNorm',
    'BatchNorm',
]

classes = __all__
