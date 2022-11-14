import torch
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
        horizon (int): Number of steps predict.
        bias (bool): Whether to add a learnable bias.
    """

    def __init__(self,
                 input_size,
                 output_size,
                 horizon=1,
                 bias=True):
        super(LinearReadout, self).__init__()

        self.readout = nn.Sequential(
            nn.Linear(input_size, output_size * horizon, bias=bias),
            Rearrange('b n (h c) -> b h n c', c=output_size, h=horizon)
        )
    #
    #     if bias:
    #         self.bias = nn.Parameter(torch.Tensor(output_size))
    #     else:
    #         self.register_parameter('bias', None)
    #     self.reset_parameters()
    #
    # def reset_parametesr(self):
    #     self.readout[0].reset_parameters()
    #     if self.bias is not None:
    #         nn.init.zeros_(self.bias)

    def forward(self, h):
        # h: [batches (steps) nodes features]
        if h.dim() == 4:
            # take last step representation
            h = h[:, -1]
        return self.readout(h)