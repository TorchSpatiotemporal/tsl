import torch
from torch import nn

from tsl.nn.utils import get_layer_activation


class LinkPredictor(nn.Module):
    r"""Output a pairwise score for each couple of input elements.

    Can be used as a building block for a graph learning model.

    .. math::
        \mathbf{S} = \left(\text{MLP}_s(\mathbf{E})\right)
        \left(\text{MLP}_t(\mathbf{E})\right)^T

    Args:
        emb_size: Size of the input embeddings.
        ff_size: Size of the hidden layer used to learn the scores.
        dropout: Dropout probability.
        activation: Activation function used in the hidden layer.
    """

    def __init__(self,
                 emb_size,
                 ff_size,
                 hidden_size,
                 dropout=0.,
                 activation='relu'):
        super(LinkPredictor, self).__init__()
        self.source_mlp = nn.Sequential(nn.Linear(emb_size, ff_size),
                                        get_layer_activation(activation)(),
                                        nn.Dropout(dropout),
                                        nn.Linear(ff_size, hidden_size))

        self.target_mlp = nn.Sequential(nn.Linear(emb_size, ff_size),
                                        get_layer_activation(activation)(),
                                        nn.Dropout(dropout),
                                        nn.Linear(ff_size, hidden_size))

    def forward(self, x):
        """"""
        # x: [*, nodes, channels]
        z_s = self.source_mlp(x)
        z_t = self.target_mlp(x)
        # scores = z_s @ z_t.T
        logits = torch.einsum('... ik, ... jk -> ... ij', z_s, z_t)
        return logits
