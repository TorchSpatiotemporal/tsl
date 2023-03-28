from tsl.nn.layers.graph_convs.adaptive_graph_conv import AdaptiveGraphConv
from tsl.nn.layers.recurrent.base import GraphGRUCellBase


class AGCRNCell(GraphGRUCellBase):
    """The Adaptive Graph Convolutional cell from the paper `"Adaptive Graph
    Convolutional Recurrent Network for Traffic Forecasting"
    <https://arxiv.org/abs/2007.02842>`_ (Bai et al., NeurIPS 2020).

    Args:
        input_size: Size of the input.
        emb_size: Size of the input node embeddings.
        hidden_size: Output size.
        num_nodes: Number of nodes in the input graph.
    """

    def __init__(self,
                 input_size: int,
                 emb_size: int,
                 hidden_size: int,
                 num_nodes: int,
                 bias: bool = True):
        self.input_size = input_size
        self.emb_size = emb_size
        self.num_nodes = num_nodes
        # instantiate gates
        forget_gate = AdaptiveGraphConv(input_size + hidden_size,
                                        emb_size,
                                        output_size=hidden_size,
                                        num_nodes=num_nodes,
                                        bias=bias)
        update_gate = AdaptiveGraphConv(input_size + hidden_size,
                                        emb_size,
                                        output_size=hidden_size,
                                        num_nodes=num_nodes,
                                        bias=bias)
        candidate_gate = AdaptiveGraphConv(input_size + hidden_size,
                                           emb_size,
                                           output_size=hidden_size,
                                           num_nodes=num_nodes,
                                           bias=bias)
        super(AGCRNCell, self).__init__(hidden_size=hidden_size,
                                        forget_gate=forget_gate,
                                        update_gate=update_gate,
                                        candidate_gate=candidate_gate)
