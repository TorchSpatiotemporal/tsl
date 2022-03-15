from typing import Optional

from torch import nn, Tensor
from torch_geometric.nn.dense import Linear

from tsl.nn.base.attention.attention import MultiHeadAttention


class SpatialSelfAttention(nn.Module):
    """
    Spatial Self Attention layer.

    Args:
        embed_dim (int): Size of the hidden dimension associeted with each node at each time step.
        num_heads (int): Number of parallel attention heads.
        dropout (float): Dropout probability.
        bias (bool, optional): Whther to add a learnable bias.
        device (optional): Device on which store the model.
        dtype (optional): Data Type of the parameters.
    Examples::
        >>> import torch
        >>> m = SpatialSelfAttention(32, 4, -1)
        >>> input = torch.randn(128, 24, 10, 20)
        >>> output, _ = m(input)
        >>> print(output.size())
        torch.Size([128, 24, 10, 32])
    """

    def __init__(self, embed_dim, num_heads,
                 in_channels=None,
                 dropout=0.,
                 bias=True,
                 device=None,
                 dtype=None) -> None:
        super(SpatialSelfAttention, self).__init__()

        self.embed_dim = embed_dim

        if in_channels is not None:
            self.input_encoder = Linear(in_channels, self.embed_dim)
        else:
            self.input_encoder = nn.Identity()

        self.attention = MultiHeadAttention(embed_dim, num_heads,
                                            axis='nodes',
                                            dropout=dropout,
                                            bias=bias,
                                            device=device,
                                            dtype=dtype)

    def forward(self, x, attn_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True):
        """"""
        # x: [batch, steps, nodes, in_channels]
        x = self.input_encoder(x)  # -> [batch, steps, nodes, embed_dim]
        return self.attention(x,
                              attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask,
                              need_weights=need_weights)
