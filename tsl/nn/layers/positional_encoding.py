import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Implementation of the positional encoding from Vaswani et al. 2017
    """
    def __init__(self, d_model, dropout=0., max_len=5000, affinity=False, batch_first=True):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        if affinity:
            self.affinity = nn.Linear(d_model, d_model)
        else:
            self.affinity = None
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.batch_first = batch_first

    def forward(self, x):
        if self.affinity is not None:
            x = self.affinity(x)
        pe = self.pe[:x.size(1), :] if self.batch_first else self.pe[:x.size(0), :]
        x = x + pe
        return self.dropout(x)