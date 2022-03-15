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
    def __init__(self, input_size, output_size, k=2, root_weight=False):
        super(DCRNNCell, self).__init__()
        # instantiate gates
        self.forget_gate = DiffConv(input_size + output_size, output_size, k=k, root_weight=root_weight)
        self.update_gate = DiffConv(input_size + output_size, output_size, k=k, root_weight=root_weight)
        self.candidate_gate = DiffConv(input_size + output_size, output_size, k=k, root_weight=root_weight)


class DCRNN(_GraphRNN):
    r"""
        Diffusion Convolutional Recurrent Network.

        From Li et al., ”Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting”, ICLR 2018

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
                 root_weight=False):
        super(DCRNN, self).__init__()
        self.d_in = input_size
        self.d_model = hidden_size
        self.n_layers = n_layers
        self.k = k
        self.rnn_cells = torch.nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(DCRNNCell(input_size=self.d_in if i == 0 else self.d_model,
                                            output_size=self.d_model, k=self.k, root_weight=root_weight))
