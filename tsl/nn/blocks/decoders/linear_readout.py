from einops.layers.torch import Rearrange
from torch import Tensor, nn


class LinearReadout(nn.Module):
    r"""Simple linear readout for multistep forecasting.

    If the input representation has a temporal dimension, this model will simply
    take the representation corresponding to the last step.

    Args:
        input_size (int): Input size.
        output_size (int): Output size.
        horizon (int): Number of steps to predict.
            (default: :obj:`1`)
        bias (bool): Whether to add a learnable bias.
            (default: :obj:`True`)
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 horizon: int = 1,
                 bias: bool = True):
        super(LinearReadout, self).__init__()

        self.readout = nn.Linear(input_size, output_size * horizon, bias=bias)
        self.rearrange = Rearrange('b n (h f) -> b h n f',
                                   f=output_size,
                                   h=horizon)

    def forward(self, h: Tensor) -> Tensor:
        """"""
        # h: [batches (steps) nodes features]
        if h.dim() == 4:
            # take last step representation
            h = h[:, -1]
        out = self.readout(h)
        return self.rearrange(out)
