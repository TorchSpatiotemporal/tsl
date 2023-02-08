from tsl.nn.base import GraphConv
from tsl.nn.base.recurrent import GraphLSTMCell, RNNBase


class GraphConvLSTMCell(GraphLSTMCell):
    r"""
    LSTM with `GraphConv` gates.
    Loosely based on Seo et al., ”Structured Sequence Modeling with Graph Convolutional Recurrent Networks”, ICONIP 2017

    Args:
        input_size: Size of the input.
        out_size: Number of units in the hidden state.
        root_weight: Whether to learn a separate transformation for the central node.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 bias: bool = True,
                 asymmetric_norm: bool = True,
                 root_weight: bool = True,
                 activation='linear',
                 cached: bool = False,
                 **kwargs):
        self.input_size = input_size
        # instantiate gates
        input_gate = GraphConv(input_size + hidden_size, hidden_size,
                               asymmetric_norm=asymmetric_norm,
                               root_weight=root_weight,
                               activation=activation,
                               bias=bias, cached=cached,
                               **kwargs)
        forget_gate = GraphConv(input_size + hidden_size, hidden_size,
                                asymmetric_norm=asymmetric_norm,
                                root_weight=root_weight,
                                activation=activation,
                                bias=bias, cached=cached,
                                **kwargs)
        cell_gate = GraphConv(input_size + hidden_size, hidden_size,
                              asymmetric_norm=asymmetric_norm,
                              root_weight=root_weight,
                              activation=activation,
                              bias=bias, cached=cached,
                              **kwargs)
        output_gate = GraphConv(input_size + hidden_size, hidden_size,
                                asymmetric_norm=asymmetric_norm,
                                root_weight=root_weight,
                                activation=activation,
                                bias=bias, cached=cached,
                                **kwargs)
        super(GraphConvLSTMCell, self).__init__(hidden_size=hidden_size,
                                                input_gate=input_gate,
                                                forget_gate=forget_gate,
                                                cell_gate=cell_gate,
                                                output_gate=output_gate)


class GraphConvLSTM(RNNBase):
    r"""
        GraphConv LSTM network.

        Loosely based on Seo et al., ”Structured Sequence Modeling with Graph Convolutional Recurrent Networks”, ICONIP 2017

        Args:
            input_size (int): Size of the input.
            hidden_size (int): Number of units in the hidden state.
            n_layers (int, optional): Number of hidden layers.
            root_weight (bool, optional): Whether to learn a separate transformation for the central node.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 n_layers: int = 1, cat_states_layers: bool = False,
                 return_only_last_state: bool = False,
                 bias: bool = True,
                 asymmetric_norm: bool = True,
                 root_weight: bool = True,
                 activation='linear',
                 cached: bool = False,
                 **kwargs):
        self.input_size = input_size
        self.hidden_size = hidden_size
        rnn_cells = [
            GraphConvLSTMCell(input_size if i == 0 else hidden_size,
                              hidden_size, asymmetric_norm=asymmetric_norm,
                              root_weight=root_weight, activation=activation,
                              bias=bias, cached=cached,
                              **kwargs)
            for i in range(n_layers)
        ]
        super(GraphConvLSTM, self).__init__(rnn_cells, cat_states_layers,
                                            return_only_last_state)
