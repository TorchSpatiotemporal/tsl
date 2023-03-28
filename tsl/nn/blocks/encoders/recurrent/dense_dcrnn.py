from typing import Optional

from torch import Tensor

from tsl.nn.layers.graph_convs.dense_graph_conv import DenseGraphConvOrderK
from tsl.nn.layers.recurrent import DenseDCRNNCell

from .base import RNNBase


class DenseDCRNN(RNNBase):
    """Dense implementation of the Diffusion Convolutional Recurrent Neural
    Network from the paper `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_
    (Li et al., ICLR 2018).

    In this implementation, the adjacency matrix is dense and the convolution is
    performed with matrix multiplication.

    Args:
        input_size: Size of the input.
        hidden_size: Number of units in the hidden state.
        n_layers: Number of layers.
        k: Size of the diffusion kernel.
        root_weight: Whether to learn a separate transformation for the central
            node.
    """
    _n_states = 1

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_layers: int = 1,
                 cat_states_layers: bool = False,
                 return_only_last_state: bool = False,
                 k: int = 2,
                 root_weight: bool = False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        rnn_cells = [
            DenseDCRNNCell(input_size if i == 0 else hidden_size,
                           hidden_size,
                           k=k,
                           root_weight=root_weight) for i in range(n_layers)
        ]
        super(DenseDCRNN, self).__init__(rnn_cells, cat_states_layers,
                                         return_only_last_state)

    def forward(self, x: Tensor, adj, h: Optional[Tensor] = None, **kwargs):
        """"""
        support = DenseGraphConvOrderK.compute_support(adj)
        return super(DenseDCRNN, self).forward(x, h=h, support=support)
