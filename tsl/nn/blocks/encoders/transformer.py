from torch import nn

from tsl.nn.base.attention import MultiHeadAttention
from tsl.nn.layers.norm import LayerNorm
from tsl.nn.utils.utils import get_layer_activation
from functools import partial

import torch.nn.functional as F


class TransformerLayer(nn.Module):
    r"""
    A TransformerLayer which can be instantiated to attent the temporal or spatial dimension.

    Args:
        input_size (int): Input size.
        hidden_size (int): Dimension of the learned representations.
        ff_size (int): Units in the MLP after self attention.
        n_heads (int, optional): Number of parallel attention heads.
        axis (str, optional): Dimension on which to apply attention to update the representations.
        causal (bool, optional): Whether to causally mask the attention scores (can be `True` only if `axis` is `steps`).
        activation (str, optional): Activation function.
        dropout (float, optional): Dropout probability.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size=None,
                 n_heads=1,
                 axis='steps',
                 causal=True,
                 activation='elu',
                 dropout=0.):
        super(TransformerLayer, self).__init__()
        self.att = MultiHeadAttention(embed_dim=hidden_size,
                                      qdim=input_size,
                                      kdim=input_size,
                                      vdim=input_size,
                                      heads=n_heads,
                                      axis=axis,
                                      causal=causal)

        if input_size != hidden_size:
            self.skip_conn = nn.Linear(input_size, hidden_size)
        else:
            self.skip_conn = nn.Identity()

        self.norm1 = LayerNorm(input_size)

        self.mlp = nn.Sequential(
            LayerNorm(hidden_size),
            nn.Linear(hidden_size, ff_size),
            get_layer_activation(activation)(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

        self.activation = getattr(F, activation)

    def forward(self, x, mask=None):
        """"""
        # x: [batch, steps, nodes, features]
        x = self.skip_conn(x) + self.dropout(self.att(self.norm1(x), attn_mask=mask)[0])
        x = x + self.mlp(x)
        return x


class SpatioTemporalTransformerLayer(nn.Module):
    r"""
    A TransformerLayer which attend both the spatial and temporal dimensions by stacking two `MultiHeadAttention` layers.

    Args:
        input_size (int): Input size.
        hidden_size (int): Dimension of the learned representations.
        ff_size (int): Units in the MLP after self attention.
        n_heads (int, optional): Number of parallel attention heads.
        causal (bool, optional): Whether to causally mask the attention scores (can be `True` only if `axis` is `steps`).
        activation (str, optional): Activation function.
        dropout (float, optional): Dropout probability.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size=None,
                 n_heads=1,
                 causal=True,
                 activation='elu',
                 dropout=0.):
        super(SpatioTemporalTransformerLayer, self).__init__()
        self.temporal_att = MultiHeadAttention(embed_dim=hidden_size,
                                               qdim=input_size,
                                               kdim=input_size,
                                               vdim=input_size,
                                               heads=n_heads,
                                               axis='steps',
                                               causal=causal)

        self.spatial_att = MultiHeadAttention(embed_dim=hidden_size,
                                              qdim=hidden_size,
                                              kdim=hidden_size,
                                              vdim=hidden_size,
                                              heads=n_heads,
                                              axis='nodes',
                                              causal=False)

        self.skip_conn = nn.Linear(input_size, hidden_size)

        self.norm1 = LayerNorm(input_size)
        self.norm2 = LayerNorm(hidden_size)

        self.mlp = nn.Sequential(
            LayerNorm(hidden_size),
            nn.Linear(hidden_size, ff_size),
            get_layer_activation(activation)(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """"""
        # x: [batch, steps, nodes, features]

        x = self.skip_conn(x) + self.dropout(self.temporal_att(self.norm1(x), attn_mask=mask)[0])
        x = x + self.dropout(self.spatial_att(self.norm2(x), attn_mask=mask)[0])
        x = x + self.mlp(x)
        return x


class Transformer(nn.Module):
    r"""
    A stack of Transformer layers.

    Args:
        input_size (int): Input size.
        hidden_size (int): Dimension of the learned representations.
        ff_size (int): Units in the MLP after self attention.
        output_size (int, optional): Size of an optional linear readout.
        n_layers (int, optional): Number of Transformer layers.
        n_heads (int, optional): Number of parallel attention heads.
        axis (str, optional): Dimension on which to apply attention to update the representations.
        causal (bool, optional): Whether to causally mask the attention scores (can be `True` only if `axis` is `steps`).
        activation (str, optional): Activation function.
        dropout (float, optional): Dropout probability.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size=None,
                 output_size=None,
                 n_layers=1,
                 n_heads=1,
                 axis='steps',
                 causal=True,
                 activation='elu',
                 dropout=0.):
        super(Transformer, self).__init__()
        self.f = getattr(F, activation)

        if ff_size is None:
            ff_size = hidden_size

        if axis in ['steps', 'nodes']:
            transformer_layer = partial(TransformerLayer, axis=axis)
        elif axis == 'both':
            transformer_layer = SpatioTemporalTransformerLayer
        else:
            raise ValueError(f'"{axis}" is not a valid axis.')

        layers = []
        for i in range(n_layers):
            layers.append(transformer_layer(d_in=input_size if i == 0 else hidden_size,
                                            d_model=hidden_size,
                                            d_ff=ff_size,
                                            n_heads=n_heads,
                                            causal=causal,
                                            activation=activation,
                                            dropout=dropout))

        self.net = nn.Sequential(*layers)

        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.register_parameter('readout', None)

    def forward(self, x):
        """"""
        x = self.net(x)
        if self.readout is not None:
            return self.readout(x)
        return x
