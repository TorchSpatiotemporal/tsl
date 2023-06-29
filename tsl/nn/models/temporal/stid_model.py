from typing import Optional, Sequence, Union

import torch
from einops import rearrange
from torch import Tensor, nn

from tsl.utils import ensure_list

from ...blocks import MLP, LinearReadout
from ...layers import NodeEmbedding
from .. import BaseModel


class STIDModel(BaseModel):
    """The Spatial-Temporal Identity (STID) model from the paper
    `"Spatial-Temporal Identity: A Simple yet Effective Baseline for
    Multivariate Time Series Forecasting" <https://arxiv.org/abs/2208.05233>`_
    (Shao et al., CIKM 2022 short paper).

    Args:
        input_size (int): Size of the input.
        n_nodes (int): Number of nodes.
        window (int): Size of the input window (this model cannot process
            sequences of variable length).
        horizon (int): Forecasting steps.
        n_exog_emb (int or list): Number of embeddings to be set for optional
            exogenous variables. Can be an integer or a list of integers where
            each element is the cardinality of each related covariate.
            (default: :obj:`None`)
        output_size (int): Size of the output. If :obj:`None`, then defaults to
            :obj:`input_size`.
            (default: :obj:`None`)
        hidden_size (int): Number of hidden units in each hidden layer.
            (default: :obj:`32`)
        n_layers (int): Number of layers in the MLP encoder.
            (default: :obj:`3`)
        dropout (float): Dropout probability in MLP hidden layers.
            (default: :obj:`0.15`)
    """

    def __init__(self,
                 input_size: int,
                 n_nodes: int,
                 window: int,
                 horizon: int,
                 n_exog_emb: Union[Sequence[int], int] = None,
                 output_size: int = None,
                 hidden_size: int = 32,
                 n_layers: int = 3,
                 dropout: float = 0.15):
        super().__init__()

        self.input_size = input_size
        self.n_nodes = n_nodes
        self.window = window
        self.horizon = horizon
        self.output_size = output_size or input_size

        self.node_emb = NodeEmbedding(n_nodes, hidden_size)
        mlp_size = 2 * hidden_size

        # temporal embeddings
        if n_exog_emb is not None:
            n_exog_emb = ensure_list(n_exog_emb)
            self.exog_embs = nn.ModuleList(
                [NodeEmbedding(size, hidden_size) for size in n_exog_emb])
            mlp_size += len(n_exog_emb) * hidden_size
        self.exog_size = n_exog_emb

        # embedding layer
        self.input_encoder = nn.Linear(input_size * window, hidden_size)

        # encoding
        self.mlp_list = nn.ModuleList([
            MLP(input_size=mlp_size,
                hidden_size=mlp_size,
                output_size=mlp_size,
                n_layers=1,
                activation="relu",
                dropout=dropout) for _ in range(n_layers)
        ])

        # regression
        self.readout = LinearReadout(mlp_size, self.output_size, horizon)

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.node_emb.emb)
            for exog_emb in self.exog_embs:
                nn.init.xavier_uniform_(exog_emb.emb)
        self.input_encoder.reset_parameters()
        for mlp in self.mlp_list:
            mlp.reset_parameters()
        self.readout.reset_parameters()

    def forward(self, x: Tensor, u: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): The input data.
            u (Tensor, optional): Optional index of exogenous variables. Each
                channel contains the index for the corresponding embeddings.
                If :obj:`u` has a time dimension, then it must be of the same
                length as :obj:`x` and equal to :obj:`self.window`; otherwise,
                it must be synchronized with the last step of :obj:`x`.
                (default: :obj:`None`)

        Shapes:
            x: :math:`(B, T, N, F)`, where :math:`B` is the batch size,
                :math:`T` is the number of time steps in the lookback window,
                :math:`N` is the number of nodes, and :math:`F` is the number
                of features/channels.
            u: :math:`(B, [T,] F)`, where :math:`B` is the batch size,
                :math:`T` is the (optional) number of time steps in the
                lookback window, :math:`F` is the number of covariates.
        """
        # x: [b t n f]
        # u: [b t f]
        b = x.size(0)  # batch size
        # flat time dimension
        assert x.size(1) == self.window
        x = rearrange(x, 'b s n f -> b n (s f)')

        h = self.input_encoder(x)  # h: b n f
        n_emb = self.node_emb(expand=(b, -1, -1))  # emb: b n f

        z = [h, n_emb]

        if u is not None:
            assert (self.exog_size is not None and u.dim() <= 3
                    and u.size(-1) == len(self.exog_size))
            if u.dim() == 3:
                assert u.size(1) == self.window
                u = u[:, -1]  # select only last step: b t f -> b 1 f
            for u_idx, u_emb in enumerate(self.exog_embs):
                t_emb = u_emb(expand=(-1, self.n_nodes, -1),
                              node_index=u[..., u_idx])
                z.append(t_emb)

        z = torch.cat(z, dim=-1)

        # encoding
        for mlp in self.mlp_list:
            z = mlp(z) + z

        # regression
        out = self.readout(z)

        return out
