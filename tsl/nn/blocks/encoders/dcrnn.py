from tsl.nn.base.recurrent import GraphGRUCell, RNNBase
from tsl.nn.layers.graph_convs.diff_conv import DiffConv


class DCRNNCell(GraphGRUCell):
    """The Diffusion Convolutional Recurrent cell from the paper
    `"Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic
    Forecasting" <https://arxiv.org/abs/1707.01926>`_ (Li et al., ICLR 2018).

    Args:
        input_size: Size of the input.
        hidden_size: Number of units in the hidden state.
        k: Size of the diffusion kernel.
        root_weight: Whether to learn a separate transformation for the central
            node.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 k: int = 2,
                 root_weight: bool = True,
                 add_backward: bool = True,
                 bias: bool = True):
        # instantiate gates
        forget_gate = DiffConv(input_size + hidden_size, hidden_size, k=k,
                               root_weight=root_weight,
                               add_backward=add_backward,
                               bias=bias)
        update_gate = DiffConv(input_size + hidden_size, hidden_size, k=k,
                               root_weight=root_weight,
                               add_backward=add_backward,
                               bias=bias)
        candidate_gate = DiffConv(input_size + hidden_size, hidden_size, k=k,
                                  root_weight=root_weight,
                                  add_backward=add_backward,
                                  bias=bias)
        super(DCRNNCell, self).__init__(hidden_size=hidden_size,
                                        forget_gate=forget_gate,
                                        update_gate=update_gate,
                                        candidate_gate=candidate_gate)


class DCRNN(RNNBase):
    """The Diffusion Convolutional Recurrent Neural Network from the paper
    `"Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic
    Forecasting" <https://arxiv.org/abs/1707.01926>`_ (Li et al., ICLR 2018).

    Args:
        input_size: Size of the input.
        hidden_size: Number of units in the hidden state.
        n_layers: Number of layers.
        k: Size of the diffusion kernel.
        root_weight: Whether to learn a separate transformation for the central
            node.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 n_layers: int = 1, cat_states_layers: bool = False,
                 return_only_last_state: bool = False,
                 k: int = 2,
                 root_weight: bool = True,
                 add_backward: bool = True,
                 bias: bool = True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        rnn_cells = [
            DCRNNCell(input_size if i == 0 else hidden_size, hidden_size, k=k,
                      root_weight=root_weight,
                      add_backward=add_backward,
                      bias=bias)
            for i in range(n_layers)
        ]
        super(DCRNN, self).__init__(rnn_cells, cat_states_layers,
                                    return_only_last_state)
