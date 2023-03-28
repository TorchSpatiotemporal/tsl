from torch import nn

from tsl.nn.layers.graph_convs import GraphConv
from tsl.nn.utils import get_functional_activation

from .mlp_decoder import MLPDecoder


class GCNDecoder(nn.Module):
    r"""GCN decoder for multistep forecasting.

    Applies multiple graph convolutional layers followed by a feed-forward layer
    and a linear readout. If the input representation has a temporal dimension,
    this model will simply take as input the representation corresponding to the
    last step.

    Args:
        input_size (int): Input size.
        hidden_size (int): Hidden size.
        output_size (int): Output size.
        horizon (int): Number of time steps in the prediction horizon.
            (default: ``1``)
        n_layers (int): Number of layers in the decoder.
            (default: ``1``)
        activation (str, optional): Activation function to be used.
            (default: ``'relu'``)
        dropout (float, optional): Dropout probability applied in the hidden
            layers.
            (default: ``0``)
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 horizon: int = 1,
                 n_layers: int = 1,
                 activation: str = 'relu',
                 dropout: float = 0.):
        super(GCNDecoder, self).__init__()
        graph_convs = []
        for i in range(n_layers):
            graph_convs.append(
                GraphConv(input_size=input_size if i == 0 else hidden_size,
                          output_size=hidden_size))
        self.convs = nn.ModuleList(graph_convs)
        self.activation = get_functional_activation(activation)
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
