from tsl.nn.base import GraphConv
from tsl.nn.blocks.encoders.gcrnn import _GraphGRUCell, _GraphRNN

from torch import nn


class GraphConvGRUCell(_GraphGRUCell):
    r"""
    Gate Recurrent Unit with `GraphConv` gates.
    Loosely based on Seo et al., ”Structured Sequence Modeling with Graph Convolutional Recurrent Networks”, ICONIP 2017

    Args:
        input_size: Size of the input.
        out_size: Number of units in the hidden state.
        root_weight: Whether to learn a separate transformation for the central node.
    """
    def __init__(self, in_size, out_size, root_weight=True):
        super(GraphConvGRUCell, self).__init__()
        # instantiate gates
        self.forget_gate = GraphConv(in_size + out_size, out_size, root_weight=root_weight)
        self.update_gate = GraphConv(in_size + out_size, out_size, root_weight=root_weight)
        self.candidate_gate = GraphConv(in_size + out_size, out_size, root_weight=root_weight)


class GraphConvGRU(_GraphRNN):
    r"""
    GraphConv GRU network.

    Loosely based on Seo et al., ”Structured Sequence Modeling with Graph Convolutional Recurrent Networks”, ICONIP 2017

    Args:
        input_size (int): Size of the input.
        hidden_size (int): Number of units in the hidden state.
        n_layers (int, optional): Number of hidden layers.
        root_weight (bool, optional): Whether to learn a separate transformation for the central node.
    """
    _n_states = 1

    def __init__(self,
                 input_size,
                 hidden_size,
                 n_layers=1,
                 root_weight=True):
        super(GraphConvGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn_cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(GraphConvGRUCell(in_size=self.input_size if i == 0 else self.hidden_size,
                                                   out_size=self.hidden_size,
                                                   root_weight=root_weight))
