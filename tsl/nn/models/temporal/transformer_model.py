from typing import Optional

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from tsl.nn.blocks.encoders import ConditionalBlock
from tsl.nn.blocks.encoders.mlp import MLP
from tsl.nn.blocks.encoders.transformer import Transformer
from tsl.nn.layers import PositionalEncoding
from tsl.nn.layers.ops import Select
from tsl.nn.models.base_model import BaseModel


class TransformerModel(BaseModel):
    r"""A Transformer from the paper `"Attention Is All You Need"
    <https://arxiv.org/abs/1706.03762>`_ (Vaswani et al., NeurIPS 2017) for
    multistep time series forecasting.

    Args:
        input_size (int): Input size.
        hidden_size (int): Dimension of the learned representations.
        output_size (int): Dimension of the output.
        ff_size (int): Units in the MLP after self attention.
        exog_size (int): Dimension of the exogenous variables.
        horizon (int): Number of forecasting steps.
        n_heads (int, optional): Number of parallel attention heads.
        n_layers (int, optional): Number of layers.
        dropout (float, optional): Dropout probability.
        axis (str, optional): Dimension on which to apply attention to update
            the representations. Can be either, 'time', 'nodes', or 'both'.
            (default: :obj:`'time'`)
        activation (str, optional): Activation function.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 horizon: int,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 ff_size: int = 32,
                 n_heads: int = 1,
                 n_layers: int = 1,
                 dropout: float = 0.,
                 axis: str = 'time',
                 activation: str = 'elu'):
        super(TransformerModel, self).__init__(return_type=Tensor)

        if exog_size:
            self.input_encoder = ConditionalBlock(input_size=input_size,
                                                  exog_size=exog_size,
                                                  output_size=hidden_size,
                                                  activation=activation)
        else:
            self.input_encoder = nn.Linear(input_size, hidden_size)

        self.pe = PositionalEncoding(hidden_size, max_len=100)

        self.transformer_encoder = nn.Sequential(
            Transformer(input_size=hidden_size,
                        hidden_size=hidden_size,
                        ff_size=ff_size,
                        n_heads=n_heads,
                        n_layers=n_layers,
                        activation=activation,
                        dropout=dropout,
                        axis=axis), Select(1, -1))

        self.readout = nn.Sequential(
            MLP(input_size=hidden_size,
                hidden_size=ff_size,
                output_size=output_size * horizon,
                dropout=dropout),
            Rearrange('b n (h f) -> b h n f', f=output_size, h=horizon))

    def forward(self, x: Tensor, u: Optional[Tensor] = None) -> Tensor:
        """"""
        # x: [batches steps nodes features]
        # u: [batches steps (nodes) features]
        b, *_ = x.size()
        if u is not None:
            if u.dim() == 3:
                u = rearrange(u, 'b t f -> b t 1 f')
            x = self.input_encoder(x, u)
        else:
            x = self.input_encoder(x)
        x = self.pe(x)
        x = self.transformer_encoder(x)

        return self.readout(x)
