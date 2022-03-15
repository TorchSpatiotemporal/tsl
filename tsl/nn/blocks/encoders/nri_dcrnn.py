import torch
from torch import nn

from tsl.nn.base.embedding import StaticGraphEmbedding
from tsl.nn.layers.link_predictor import LinkPredictor

from tsl.nn.blocks.encoders.dense_dcrnn import DenseDCRNN

import tsl


class DifferentiableBinarySampler(nn.Module):
    """
    This module exploits the GumbelMax trick to sample from a Bernoulli distribution in differentiable fashion.

    Adapted from https://github.com/yaringal/ConcreteDropout
    """
    def __init__(self):
        super(DifferentiableBinarySampler, self).__init__()

    def forward(self, scores, tau):
        unif_noise = torch.rand_like(scores)
        eps = tsl.epsilon

        logit = torch.log(scores + eps) - torch.log(1 - scores + eps) + \
                torch.log(unif_noise + eps) - torch.log(1 - unif_noise + eps)

        soft_out = torch.sigmoid(logit / tau)
        return soft_out


class NeuRelInfDCRNN(DenseDCRNN):
    r"""
        Diffusion Convolutional Recurrent Network with graph learned through neural relational inference.

        Loosely inspired by:
            - Kipf et al. "Neural relational inference for interacting systems". ICLR 2018.
            - Shang et al. "Discrete graph structure learning for forecasting multiple time series". ICLR 2021.

        Args:
             input_size: Size of the input.
             hidden_size: Number of units in the hidden state.
             n_layers: Number of layers.
             k: Size of the diffusion kernel.
             root_weight: Whether to learn a separate transformation for the central node.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 emb_size,
                 n_nodes,
                 n_layers=1,
                 k=2,
                 root_weight=False):
        super(NeuRelInfDCRNN, self).__init__(input_size=input_size,
                                             hidden_size=hidden_size,
                                             n_layers=n_layers,
                                             k=k,
                                             root_weight=root_weight)

        self.node_emb = StaticGraphEmbedding(n_tokens=n_nodes,
                                             emb_size=emb_size)
        self.link_predictor = LinkPredictor(emb_size=emb_size,
                                            ff_size=hidden_size,
                                            hidden_size=hidden_size
                                            )
        self.sampler = DifferentiableBinarySampler()

    def forward(self, x, h=None, tau=0.25):
        emb = self.node_emb()
        adj_p = torch.sigmoid(self.link_predictor(emb))
        adj = self.sampler(adj_p, tau)
        return super(NeuRelInfDCRNN, self).forward(x, adj, h)
