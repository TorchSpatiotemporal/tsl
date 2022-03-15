from torch import nn
from torch.nn import functional as F

from tsl.nn.base.graph_conv import GraphConv
from tsl.nn.blocks.decoders.mlp_decoder import MLPDecoder


class GCNDecoder(nn.Module):
    r"""
    GCN decoder for multi-step forecasting.
    Applies multiple graph convolutional layers followed by a feed-forward layer amd a linear readout.

    If the input representation has a temporal dimension, this model will simply take as input the representation
    corresponding to the last step.

    Args:
        input_size (int): Input size.
        hidden_size (int): Hidden size.
        output_size (int): Output size.
        horizon (int): Output steps.
        n_layers (int, optional): Number of layers in the decoder. (default: 1)
        activation (str, optional): Activation function to use.
        dropout (float, optional): Dropout probability applied in the hidden layers.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 horizon=1,
                 n_layers=1,
                 activation='relu',
                 dropout=0.):
        super(GCNDecoder, self).__init__()
        graph_convs = []
        for l in range(n_layers):
            graph_convs.append(
                GraphConv(input_size=input_size if l == 0 else hidden_size,
                          output_size=hidden_size)
            )
        self.convs = nn.ModuleList(graph_convs)
        self.activation = getattr(F, activation)
        self.dropout = nn.Dropout(dropout)
        self.readout = MLPDecoder(input_size=hidden_size,
                                  hidden_size=hidden_size,
                                  output_size=output_size,
                                  activation=activation,
                                  horizon=horizon)

    def forward(self, h, edge_index, edge_weight=None):
        """"""
        # h: [batches (steps) nodes features]
        if h.dim() == 4:
            # take last step representation
            h = h[:, -1]
        for conv in self.convs:
            h = self.dropout(self.activation(conv(h, edge_index, edge_weight)))
        return self.readout(h)
