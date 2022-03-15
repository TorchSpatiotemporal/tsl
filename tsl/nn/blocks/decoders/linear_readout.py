from torch import nn
from einops.layers.torch import Rearrange


class LinearReadout(nn.Module):
    r"""
    Simple linear readout for multi-step forecasting.

    If the input representation has a temporal dimension, this model will simply take the representation corresponding
    to the last step.

    Args:
        input_size (int): Input size.
        output_size (int): Output size.
        horizon(int): Number of steps predict.
    """
    def __init__(self,
                 input_size,
                 output_size,
                 horizon=1):
        super(LinearReadout, self).__init__()

        self.readout = nn.Sequential(
            nn.Linear(input_size, output_size * horizon),
            Rearrange('b n (h * c) -> b h n c', c=output_size, h=horizon)
        )

    def forward(self, h):
        # h: [batches (steps) nodes features]
        if h.dim() == 4:
            # take last step representation
            h = h[:, -1]
        return self.readout(h)
