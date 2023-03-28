from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from tsl.nn.blocks.encoders.mlp import MLP


class MLPDecoder(nn.Module):
    r"""Simple MLP decoder for multistep forecasting.

    If the input representation has a temporal dimension, this model will take
    the flattened representations corresponding to the last
    ``'receptive_field'`` time steps.

    Args:
        input_size (int): Input size.
        hidden_size (int): Hidden size.
        output_size (int): Output size.
        horizon (int): Number of steps to predict.
            (default: :obj:`1`)
        n_layers (int): Number of hidden layers in the decoder.
            (default: ``1``)
        receptive_field (int): Number of steps to consider for decoding.
            (default: :obj:`1`)
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
                 receptive_field: int = 1,
                 activation: str = 'relu',
                 dropout: float = 0.):
        super(MLPDecoder, self).__init__()

        self.receptive_field = receptive_field
        self.readout = MLP(input_size=receptive_field * input_size,
                           hidden_size=hidden_size,
                           output_size=output_size * horizon,
                           n_layers=n_layers,
                           dropout=dropout,
                           activation=activation)
        self.rearrange = Rearrange('b n (h f) -> b h n f',
                                   f=output_size,
                                   h=horizon)

    def forward(self, h):
        """"""
        # h: [batches (steps) nodes features]
        if h.dim() == 4:
            # take last step representation
            h = rearrange(h[:, -self.receptive_field:], 'b t n f -> b n (t f)')
        else:
            assert self.receptive_field == 1
        out = self.readout(h)
        return self.rearrange(out)
