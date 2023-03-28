from tsl.nn.layers.graph_convs.diff_conv import DiffConv
from tsl.nn.layers.recurrent.base import GraphGRUCellBase


class DCRNNCell(GraphGRUCellBase):
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

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 k: int = 2,
                 root_weight: bool = True,
                 add_backward: bool = True,
                 bias: bool = True):
        # instantiate gates
        forget_gate = DiffConv(input_size + hidden_size,
                               hidden_size,
                               k=k,
                               root_weight=root_weight,
                               add_backward=add_backward,
                               bias=bias)
        update_gate = DiffConv(input_size + hidden_size,
                               hidden_size,
                               k=k,
                               root_weight=root_weight,
                               add_backward=add_backward,
                               bias=bias)
        candidate_gate = DiffConv(input_size + hidden_size,
                                  hidden_size,
                                  k=k,
                                  root_weight=root_weight,
                                  add_backward=add_backward,
                                  bias=bias)
        super(DCRNNCell, self).__init__(hidden_size=hidden_size,
                                        forget_gate=forget_gate,
                                        update_gate=update_gate,
                                        candidate_gate=candidate_gate)
