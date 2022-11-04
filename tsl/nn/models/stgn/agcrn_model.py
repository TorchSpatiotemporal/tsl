from typing import Optional

from einops import rearrange
from torch import nn, Tensor
from torch_geometric.typing import Adj, OptTensor

from tsl.nn.blocks.decoders.mlp_decoder import MLPDecoder
from tsl.nn.blocks.encoders import ConditionalBlock
from tsl.nn.blocks.encoders.agcrn import AGCRN
from ..base_model import BaseModel
from ...blocks.decoders import LinearReadout
from ...utils import utils


class AGCRNModel(BaseModel):
    r"""The Adaptive Graph Convolutional Recurrent Network from the paper
    `"Adaptive Graph Convolutional Recurrent Network for Trafﬁc Forecasting" <https://arxiv.org/abs/2007.02842>`_ (Bai et al., NeurIPS 2020).

    Args:
        input_size (int): Number of features of the input sample.
        output_size (int): Number of output channels.
        horizon (int): Number of future time steps to forecast.
        exog_size (int): Number of features of the input covariate, if any.
        hidden_size (int): Number of hidden units.
        hidden_size (int): Size of the learned node embeddings.
        n_nodes (int): Number of nodes in the input (static) graph.
        n_layers (int): Number of AGCRN cells.
            (default: :obj:`1`)
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 horizon: int,
                 exog_size: int,
                 n_nodes: int,
                 emb_size: int,
                 hidden_size: int,
                 n_layers: int = 1):
        super(AGCRNModel, self).__init__(return_type=Tensor)

        self.input_encoder = nn.Linear(input_size + exog_size, hidden_size)

        self.agrn = AGCRN(input_size=hidden_size,
                          emb_size=emb_size,
                          num_nodes=n_nodes,
                          hidden_size=hidden_size,
                          n_layers=n_layers)

        self.readout = LinearReadout(input_size=hidden_size,
                                     output_size=output_size,
                                     horizon=horizon)

    def forward(self, x: Tensor, u: OptTensor = None, **kwargs) -> Tensor:
        """"""
        x = utils.maybe_cat_exog(x, u)
        x = self.input_encoder(x)
        h, _ = self.agrn(x)
        return self.readout(h)
