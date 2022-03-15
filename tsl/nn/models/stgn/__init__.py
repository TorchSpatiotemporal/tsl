from .dcrnn_model import DCRNNModel
from .graph_wavenet_model import GraphWaveNetModel
from .rnn2gcn_model import RNNEncGCNDecModel
from .stcn_model import STCNModel

__all__ = [
    'DCRNNModel',
    'GraphWaveNetModel',
    'RNNEncGCNDecModel',
    'STCNModel'
]

classes = __all__
