from .conv import MultiConv1d
from .dense import MultiDense
from .linear import MultiLinear
from .recurrent import MultiGRUCell, MultiLSTMCell

__all__ = [
    'MultiLinear',
    'MultiDense',
    'MultiConv1d',
    'MultiGRUCell',
    'MultiLSTMCell',
]

classes = __all__
