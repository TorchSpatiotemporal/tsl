from tsl.nn.base import StaticGraphEmbedding
from tsl.nn.layers.graph_convs.adap_graph_conv import AdaptiveGraphConv
from tsl.nn.blocks.encoders.gcrnn import _GraphGRUCell, _GraphRNN

from torch import nn


class AGCRNCell(_GraphGRUCell):
    r"""
    Adaptive Graph Convolutional Cell.
    Based on Bai et al. "Adaptive Graph Convolutional Recurrent Network for Trafﬁc Forecasting", NeurIPS 2020

    Args:
        in_size: Size of the input.
        emb_size: Size of the input node embeddings.
        out_size: Output size.
        num_nodes: Number of nodes in the input graph.
    """
    def __init__(self, in_size, emb_size, out_size, num_nodes):
        super(AGCRNCell, self).__init__()
        # instantiate gates
        self.forget_gate = AdaptiveGraphConv(in_size + out_size, emb_size,  out_size, num_nodes)
        self.update_gate = AdaptiveGraphConv(in_size + out_size, emb_size,  out_size, num_nodes)
        self.candidate_gate = AdaptiveGraphConv(in_size + out_size, emb_size,  out_size, num_nodes)


class AGCRN(_GraphRNN):
    r"""
    Adaptive Graph Convolutional Recurrent Network.
    Based on Bai et al. "Adaptive Graph Convolutional Recurrent Network for Trafﬁc Forecasting", NeurIPS 2020

    Args:
        input_size: Size of the input.
        emb_size: Size of the input node embeddings.
        hidden_size: Output size.
        num_nodes: Number of nodes in the input graph.
        n_layers: Number of recurrent layers.
    """
    _n_states = 1

    def __init__(self,
                 input_size,
                 emb_size,
                 hidden_size,
                 num_nodes,
                 n_layers=1):
        super(AGCRN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn_cells = nn.ModuleList()
        self.node_emb = StaticGraphEmbedding(num_nodes, emb_size)
        for i in range(self.n_layers):
            self.rnn_cells.append(AGCRNCell(in_size=self.input_size if i == 0 else self.hidden_size,
                                            emb_size=emb_size,
                                            out_size=self.hidden_size,
                                            num_nodes=num_nodes))

    def forward(self, x, *args, h=None, **kwargs):
        emb = self.node_emb()
        adj = AdaptiveGraphConv.compute_adj(emb)
        return super(AGCRN, self).forward(x, h=h, adj=adj, e=emb)