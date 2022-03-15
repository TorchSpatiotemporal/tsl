import torch
from torch import nn

from tsl.nn.blocks.encoders.mlp import MLP
from einops import rearrange, repeat


class MultiHorizonMLPDecoder(nn.Module):
    r"""
    Decoder for multistep forecasting based on

    Wen et al., "A Multi-Horizon Quantile Recurrent Forecaster", 2018.

    It requires exogenous variables synched with the forecasting horizon.

    Args:
        input_size (int): Size of the input.
        exog_size (int): Size of the horizon exogenous variables.
        hidden_size (int): Number of hidden units.
        context_size (int): Number of units used to condition the forecasting of each step.
        output_size (int): Output channels.
        n_layers (int): Number of hidden layers.
        horizon (int): Forecasting horizon.
        activation (str, optional): Activation function.
        dropout (float, optional): Dropout probability.
    """
    def __init__(self,
                 input_size,
                 exog_size,
                 hidden_size,
                 context_size,
                 output_size,
                 n_layers,
                 horizon,
                 activation='relu',
                 dropout=0.):
        super(MultiHorizonMLPDecoder, self).__init__()
        global_d_out = horizon * context_size + context_size
        self.d_context = context_size
        self.horizon = horizon
        self.global_mlp = MLP(input_size=input_size, hidden_size=hidden_size, output_size=global_d_out,
                              n_layers=n_layers, activation=activation, dropout=dropout)
        self.local_mlp = MLP(input_size=exog_size + 2 * context_size, hidden_size=hidden_size, output_size=output_size,
                             n_layers=n_layers, activation=activation, dropout=dropout)

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        """"""
        # x: [batch, (steps), nodes, features]
        # u: [batch, horizon, (nodes), features]
        if x.dim() == 4:
            x = x[:, -1]
        n = x.size(1)

        if u.dim() == 3:
            u = repeat(u, 'b h c -> b h n c', n=n)
        out = self.global_mlp.f(self.global_mlp(x))
        global_context, contexts = torch.split(out, [self.d_context, self.horizon * self.d_context], -1)
        global_context = repeat(global_context, 'b n c -> b n h c', h=self.horizon)
        contexts = rearrange(contexts, 'b n (h c) -> b n h c', c=self.d_context, h=self.horizon)
        x_dec = torch.cat([contexts, global_context, u], -1)
        return self.local_mlp(x_dec)

    def reset_parameters(self):
        self.global_mlp.reset_parameters()
        self.local_mlp.reset_parameters()
