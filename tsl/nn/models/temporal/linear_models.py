from typing import Optional

from einops import rearrange
from torch import Tensor

from tsl.nn.blocks.decoders import LinearReadout
from tsl.nn.models.base_model import BaseModel
from tsl.nn.utils import maybe_cat_exog


class ARModel(BaseModel):
    r"""Simple univariate linear AR model for multistep forecasting.

    Args:
        input_size (int): Size of the input.
        temporal_order (int): Number of units in the recurrent cell.
        output_size (int): Number of output channels.
        exog_size (int): Size of the exogenous variables.
        horizon (int): Forecasting horizon.
    """

    def __init__(self,
                 input_size: int,
                 temporal_order: int,
                 output_size: int,
                 horizon: int,
                 exog_size: int = 0,
                 bias: bool = True):
        super(ARModel, self).__init__(return_type=Tensor)

        input_size += exog_size
        self.linear = LinearReadout(input_size=input_size * temporal_order,
                                    output_size=output_size,
                                    horizon=horizon,
                                    bias=bias)
        self.temporal_order = temporal_order

    def forward(self, x: Tensor, u: Optional[Tensor] = None) -> Tensor:
        """"""
        # x: [batches steps nodes features]
        # u: [batches steps (nodes) features]
        x = maybe_cat_exog(x, u)
        x = x[:, -self.temporal_order:]
        x = rearrange(x, 'b s n f -> b n (s f)')
        return self.linear(x)


class VARModel(ARModel):
    r"""A simple VAR model for multistep forecasting.

    Args:
        input_size (int): Size of the input.
        n_nodes (int): Number of nodes.
        temporal_order (int): Number of units in the recurrent cell.
        output_size (int): Number of output channels.
        exog_size (int): Size of the exogenous variables.
        horizon (int): Forecasting horizon.
    """

    def __init__(self,
                 input_size: int,
                 temporal_order: int,
                 output_size: int,
                 horizon: int,
                 n_nodes: int,
                 exog_size: int = 0,
                 bias: bool = True):

        super(VARModel, self).__init__(input_size=input_size * n_nodes,
                                       temporal_order=temporal_order,
                                       output_size=output_size * n_nodes,
                                       horizon=horizon,
                                       exog_size=exog_size,
                                       bias=bias)

    def forward(self, x: Tensor, u: Optional[Tensor] = None) -> Tensor:
        """"""
        # x: [batches, steps, nodes, features]
        # u: [batches, steps, (nodes), features]
        *_, n, _ = x.size()
        x = rearrange(x, 'b t n f -> b t 1 (n f)')
        if u is not None and u.dim() == 4:
            u = rearrange(u, 'b t n f -> b t 1 (n f)')
        x = super(VARModel, self).forward(x, u)
        # [b, h, 1, (n f)]
        return rearrange(x, 'b h 1 (n f) -> b h n f', n=n)
