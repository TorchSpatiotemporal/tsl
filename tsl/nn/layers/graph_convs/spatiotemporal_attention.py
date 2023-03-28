import torch.nn as nn
from einops import rearrange, reduce
from torch.nn import MultiheadAttention


class SpatioTemporalAttention(nn.Module):

    def __init__(self,
                 d_in,
                 d_model,
                 d_ff,
                 n_heads,
                 dropout,
                 pool_size=1,
                 pooling_op='mean'):
        super(SpatioTemporalAttention, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.pool_size = pool_size
        self.pooling_op = pooling_op

        if self.d_in != self.d_model:
            self.input_encoder = nn.Linear(self.d_in, self.d_model)
        else:
            self.input_encoder = nn.Identity()

        self.temporal_attn = MultiheadAttention(self.d_model,
                                                self.n_heads,
                                                dropout=dropout)
        self.spatial_attn = MultiheadAttention(self.d_model,
                                               self.n_heads,
                                               dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(self.d_model, self.d_ff)
        self.linear2 = nn.Linear(self.d_ff, self.d_model)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        # x: [batch, steps, nodes, features]
        # u: [batch, steps, nodes, features]
        b, s, n, f = x.size()
        x = rearrange(x, 'b s n f -> s (b n) f')

        x = self.input_encoder(x)
        if (self.pool_size > 1) and (s >= self.pool_size):
            q = reduce(x,
                       '(s1 s2) m f -> s1 m f',
                       self.pooling_op,
                       s2=self.pool_size)
        else:
            q = x
        # temporal module
        x2 = self.temporal_attn(q, x, x)[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x = rearrange(x, 's (b n) f -> n (b s) f', b=b, n=n)

        # spatial module
        x2 = self.spatial_attn(x, x, x)[0]
        x = x + self.dropout2(x2)
        x = self.norm2(x)

        # feed-forward network
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout3(x2)
        x = self.norm3(x)
        return x
