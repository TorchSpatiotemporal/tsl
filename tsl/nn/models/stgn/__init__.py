from .dcrnn_model import DCRNNModel
from .gated_gn_model import GatedGraphNetworkModel
from .graph_wavenet_model import GraphWaveNetModel
from .grin_model import GRINModel
from .rnn2gcn_model import RNNEncGCNDecModel
from .stcn_model import STCNModel

__all__ = [
    'DCRNNModel',
    'GraphWaveNetModel',
    'GatedGraphNetworkModel',
    'RNNEncGCNDecModel',
    'STCNModel',
    'GRINModel'
]

classes = __all__
