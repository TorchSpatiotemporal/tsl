import torch

from tsl.nn.layers.graph_convs.dense_spatial_conv import SpatialConvOrderK
from tsl.nn.blocks.encoders.gcrnn import _GraphGRUCell, _GraphRNN


class DenseDCRNNCell(_GraphGRUCell):
    r"""
    Diffusion Convolutional Recurrent Cell.

    Args:
         input_size: Size of the input.
         output_size: Number of units in the hidden state.
         k: Size of the diffusion kernel.
         root_weight: Whether to learn a separate transformation for the central node.
    """
    def __init__(self, input_size, output_size, k=2, root_weight=False):
        super(DenseDCRNNCell, self).__init__()
        # instantiate gates
        self.forget_gate = SpatialConvOrderK(input_size=input_size + output_size,
                                             output_size=output_size,
                                             support_len=2,
                                             order=k,
                                             include_self=root_weight,
                                             channel_last=True)
        self.update_gate = SpatialConvOrderK(input_size=input_size + output_size,
                                             output_size=output_size,
                                             support_len=2,
                                             order=k,
                                             include_self=root_weight,
                                             channel_last=True)
        self.candidate_gate = SpatialConvOrderK(input_size=input_size + output_size,
                                                output_size=output_size,
                                                support_len=2,
                                                order=k,
                                                include_self=root_weight,
                                                channel_last=True)


class DenseDCRNN(_GraphRNN):
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
        super(DenseDCRNN, self).__init__()
        self.d_in = input_size
        self.d_model = hidden_size
        self.n_layers = n_layers
        self.k = k
        self.rnn_cells = torch.nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(DenseDCRNNCell(input_size=self.d_in if i == 0 else self.d_model,
                                                 output_size=self.d_model, k=self.k, root_weight=root_weight))

    def forward(self, x, adj, h=None):
        support = SpatialConvOrderK.compute_support(adj)
        return super(DenseDCRNN, self).forward(x, support, h=h)
