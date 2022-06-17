import torch

from tsl.nn.layers.graph_convs.diff_conv import DiffConv
from tsl.nn.blocks.encoders.gcrnn import _GraphGRUCell, _GraphRNN


class DCRNNCell(_GraphGRUCell):
    """
    Diffusion Convolutional Recurrent Cell.

    Args:
         input_size: Size of the input.
         output_size: Number of units in the hidden state.
         k: Size of the diffusion kernel.
         root_weight: Whether to learn a separate transformation for the central node.
    """

    def __init__(self, input_size, output_size, k=2, root_weight=True):
        super(DCRNNCell, self).__init__()
        # instantiate gates
        self.forget_gate = DiffConv(input_size + output_size, output_size, k=k,
                                    root_weight=root_weight)
        self.update_gate = DiffConv(input_size + output_size, output_size, k=k,
                                    root_weight=root_weight)
        self.candidate_gate = DiffConv(input_size + output_size, output_size,
                                       k=k, root_weight=root_weight)


class DCRNN(_GraphRNN):
    r"""Diffusion Convolutional Recurrent Network, from the paper
    `"Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_.

    Args:
         input_size: Size of the input.
         hidden_size: Number of units in the hidden state.
         n_layers: Number of layers.
         k: Size of the diffusion kernel.
         root_weight: Whether to learn a separate transformation for the central node.
    """
    _n_states = 1

    def __init__(self,
                 input_size,
                 hidden_size,
                 n_layers=1,
                 k=2,
                 root_weight=True):
        super(DCRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.k = k
        self.rnn_cells = torch.nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(DCRNNCell(
                input_size=self.input_size if i == 0 else self.hidden_size,
                output_size=self.hidden_size, k=self.k,
                root_weight=root_weight))
