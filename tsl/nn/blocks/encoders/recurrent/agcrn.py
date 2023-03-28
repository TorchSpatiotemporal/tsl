from typing import Optional

from torch import Tensor

from tsl.nn.layers.base import NodeEmbedding
from tsl.nn.layers.graph_convs.adaptive_graph_conv import AdaptiveGraphConv
from tsl.nn.layers.recurrent import AGCRNCell

from .base import RNNBase


class AGCRN(RNNBase):
    r"""The Adaptive Graph Convolutional Recurrent Network from the paper
    `"Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting"
    <https://arxiv.org/abs/2007.02842>`_ (Bai et al., NeurIPS 2020).

    Args:
        input_size: Size of the input.
        emb_size: Size of the input node embeddings.
        hidden_size: Output size.
        num_nodes: Number of nodes in the input graph.
        n_layers: Number of recurrent layers.
    """

    def __init__(self,
                 input_size: int,
                 emb_size: int,
                 hidden_size: int,
                 num_nodes: int,
                 n_layers: int = 1,
                 cat_states_layers: bool = False,
                 return_only_last_state: bool = False,
                 bias: bool = True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        rnn_cells = [
            AGCRNCell(input_size if i == 0 else hidden_size,
                      emb_size=emb_size,
                      hidden_size=hidden_size,
                      num_nodes=num_nodes,
                      bias=bias) for i in range(n_layers)
        ]
        super(AGCRN, self).__init__(rnn_cells, cat_states_layers,
                                    return_only_last_state)
        self.node_emb = NodeEmbedding(num_nodes, emb_size)

    def forward(self, x: Tensor, h: Optional[Tensor] = None):
        """"""
        emb = self.node_emb()
        adj = AdaptiveGraphConv.compute_adj(emb)
        return super(AGCRN, self).forward(x, h=h, adj=adj, e=emb)
