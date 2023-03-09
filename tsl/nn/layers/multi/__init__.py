from .conv import MultiConv1d
from .dense import MultiDense
from .linear import MultiLinear
from .recurrent import MultiGRUCell, MultiLSTMCell, MultiRNN
from .mlp import MultiMLP

__all__ = [
    'MultiLinear',
    'MultiDense',
    'MultiMLP',
    'MultiConv1d',
    'MultiGRUCell',
    'MultiLSTMCell',
    'MultiRNN'
]

classes = __all__
