from torch import nn

from tsl.nn.blocks.encoders.mlp import MLP
from einops.layers.torch import Rearrange

from einops import rearrange


class MLPDecoder(nn.Module):
    r"""
    Simple MLP decoder for multi-step forecasting.

    If the input representation has a temporal dimension, this model will take the flatten representations corresponding
    to the last `receptive_field` time steps.

    Args:
        input_size (int): Input size.
        hidden_size (int): Hidden size.
        output_size (int): Output size.
        horizon (int): Output steps.
        n_layers (int, optional): Number of layers in the decoder. (default: 1)
        receptive_field (int, optional): Number of steps to consider for decoding. (default: 1)
        activation (str, optional): Activation function to use.
        dropout (float, optional): Dropout probability applied in the hidden layers.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 horizon=1,
                 n_layers=1,
                 receptive_field=1,
                 activation='relu',
                 dropout=0.):
        super(MLPDecoder, self).__init__()

        self.receptive_field = receptive_field
        self.readout = nn.Sequential(
            MLP(input_size=receptive_field * input_size,
                hidden_size=hidden_size,
                output_size=output_size * horizon,
                n_layers=n_layers,
                dropout=dropout,
                activation=activation),
            Rearrange('b n (h c) -> b h n c', c=output_size, h=horizon)
        )

    def forward(self, h):
        # h: [batches (steps) nodes features]
        if h.dim() == 4:
            # take last step representation
            h = rearrange(h[:, -self.receptive_field:], 'b s n c -> b n (s c)')
        else:
            assert self.receptive_field == 1
        return self.readout(h)
