from .conditional import ConditionalBlock, ConditionalTCNBlock
from .dcrnn import DCRNNCell, DCRNN
from .dense_dcrnn import DenseDCRNNCell, DenseDCRNN
from .gcgru import GraphConvGRUCell, GraphConvGRU
from .gclstm import GraphConvLSTMCell, GraphConvLSTM
from .mlp import MLP, ResidualMLP
from .rnn import RNN
from .stcn import SpatioTemporalConvNet
from .tcn import TemporalConvNet
from .transformer import (TransformerLayer, SpatioTemporalTransformerLayer,
                          Transformer)

general_classes = [
    'ConditionalBlock',
    'ConditionalTCNBlock',
    'MLP',
    'ResidualMLP',
    'RNN',
]

cell_classes = [
    'DCRNNCell',
    'DenseDCRNNCell',
    'GraphConvGRUCell',
    'GraphConvLSTMCell'
]

grnn_classes = [
    'DCRNN',
    'DenseDCRNN',
    'GraphConvGRU',
    'GraphConvLSTM'
]

conv_classes = [
    'TemporalConvNet',
    'SpatioTemporalConvNet'
]

transformer_classes = [
    'TransformerLayer',
    'SpatioTemporalTransformerLayer',
    'Transformer'
]

classes = [
    *general_classes,
    *cell_classes,
    *grnn_classes,
    *conv_classes,
    *transformer_classes
]

__all__ = classes
