from torch import Tensor, nn
from torch_geometric.typing import OptTensor

from tsl.nn.blocks.decoders import LinearReadout
from tsl.nn.blocks.encoders import AGCRN
from tsl.nn.models.base_model import BaseModel
from tsl.nn.utils import maybe_cat_exog


class AGCRNModel(BaseModel):
    r"""The Adaptive Graph Convolutional Recurrent Network from the paper
    `"Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting"
    <https://arxiv.org/abs/2007.02842>`_ (Bai et al., NeurIPS 2020).

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
                 n_nodes: int,
                 hidden_size: int = 64,
                 emb_size: int = 10,
                 exog_size: int = 0,
                 n_layers: int = 1):
        super(AGCRNModel, self).__init__(return_type=Tensor)

        self.input_encoder = nn.Linear(input_size + exog_size, hidden_size)

        self.agrn = AGCRN(input_size=hidden_size,
                          emb_size=emb_size,
                          num_nodes=n_nodes,
                          hidden_size=hidden_size,
                          n_layers=n_layers,
                          return_only_last_state=True)

        self.readout = LinearReadout(input_size=hidden_size,
                                     output_size=output_size,
                                     horizon=horizon)

    def forward(self, x: Tensor, u: OptTensor = None) -> Tensor:
        """"""
        x = maybe_cat_exog(x, u)
        x = self.input_encoder(x)
        out = self.agrn(x)
        return self.readout(out)
