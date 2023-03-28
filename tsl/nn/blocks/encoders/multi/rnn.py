from tsl.nn.blocks.encoders.recurrent import RNNBase
from tsl.nn.layers.multi import MultiGRUCell, MultiLSTMCell


class MultiRNN(RNNBase):
    """A Recurrent Neural Network whose cells' weights are not shared among
    the different instances."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_instances: int,
                 n_layers: int = 1,
                 cat_states_layers: bool = False,
                 return_only_last_state: bool = False,
                 cell: str = 'gru',
                 bias: bool = True,
                 **kwargs):

        if cell == 'gru':
            cell = MultiGRUCell
        elif cell == 'lstm':
            cell = MultiLSTMCell
        else:
            raise NotImplementedError(f'"{cell}" cell not implemented.')

        rnn_cells = [
            cell(input_size if i == 0 else hidden_size,
                 hidden_size,
                 n_instances,
                 bias=bias,
                 **kwargs) for i in range(n_layers)
        ]
        super(MultiRNN, self).__init__(rnn_cells, cat_states_layers,
                                       return_only_last_state)
