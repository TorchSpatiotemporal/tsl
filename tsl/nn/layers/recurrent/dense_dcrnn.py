from tsl.nn.layers.graph_convs.dense_graph_conv import DenseGraphConvOrderK
from tsl.nn.layers.recurrent.base import GraphGRUCellBase


class DenseDCRNNCell(GraphGRUCellBase):
    r"""Dense implementation of the Diffusion Convolutional Recurrent cell from
    the paper `"Diffusion Convolutional Recurrent Neural Network: Data-Driven
    Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_ (Li et al., ICLR
    2018).

    In this implementation, the adjacency matrix is dense and the convolution is
    performed with matrix multiplication.

    Args:
         input_size: Size of the input.
         hidden_size: Number of units in the hidden state.
         k: Size of the diffusion kernel.
         root_weight (bool): Whether to learn a separate transformation for the
            central node.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 k: int = 2,
                 root_weight: bool = False):
        # instantiate gates
        forget_gate = DenseGraphConvOrderK(input_size + hidden_size,
                                           hidden_size,
                                           support_len=2,
                                           order=k,
                                           include_self=root_weight,
                                           channel_last=True)
        update_gate = DenseGraphConvOrderK(input_size + hidden_size,
                                           hidden_size,
                                           support_len=2,
                                           order=k,
                                           include_self=root_weight,
                                           channel_last=True)
        candidate_gate = DenseGraphConvOrderK(input_size + hidden_size,
                                              hidden_size,
                                              support_len=2,
                                              order=k,
                                              include_self=root_weight,
                                              channel_last=True)

        super(DenseDCRNNCell, self).__init__(hidden_size=hidden_size,
                                             forget_gate=forget_gate,
                                             update_gate=update_gate,
                                             candidate_gate=candidate_gate)
