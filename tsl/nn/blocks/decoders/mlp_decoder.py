from torch import nn

from tsl.nn.blocks.encoders.mlp import MLP
from einops.layers.torch import Rearrange


class MLPDecoder(nn.Module):
    r"""
    Simple MLP decoder for multi-step forecasting.

    If the input representation has a temporal dimension, this model will simply take the representation corresponding
    to the last step.

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
        super(MLPDecoder, self).__init__()

        self.readout = nn.Sequential(
            MLP(input_size=input_size,
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
            h = h[:, -1]
        return self.readout(h)
