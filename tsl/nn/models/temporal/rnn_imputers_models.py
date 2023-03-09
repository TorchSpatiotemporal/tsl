from typing import Optional, Union

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, Tensor

from tsl import logger
from tsl.nn.functional import reverse_tensor
from tsl.nn.models.base_model import BaseModel


class RNNImputerModel(BaseModel):
    r"""Fill the blanks with 1-step-ahead predictions of a recurrent network.

    .. math ::
        \bar{x}_{t} = m_{t} \cdot x_{t} + (1 - m_{t}) \cdot \hat{x}_{t}

    Args:
        input_size (int): Number of features of the input sample.
        hidden_size (int): Number of hidden units.
            (default: 64)
        exog_size (int): Number of features of the input covariate, if any.
            (default: :obj:`0`)
        cell (str): Type of recurrent cell to be used, one of [:obj:`gru`,
            :obj:`lstm`].
            (default: :obj:`gru`)
        concat_mask (bool): If :obj:`True`, then the input tensor is
            concatenated to the mask when fed to the RNN cell.
            (default: :obj:`True`)
        fully_connected (bool): If :obj:`True`, then node and feature dimensions
            are flattened together.
            (default: :obj:`False`)
        n_nodes (int, optional): The number of nodes in the input sample, to be
            provided in case :obj:`fully_connected` is :obj:`True`.
            (default: :obj:`None`)
        detach_input (bool): If :obj:`True`, call :meth:`~torch.Tensor.detach`
            on predictions before they are used to fill the gaps, breaking the
            error backpropagation.
            (default: :obj:`False`)
        state_init (str): How to initialize the state of the recurrent cell,
            one of [:obj:`zero`, :obj:`noise`]. With :obj:`noise`, the states
            are drawn from a normal distribution.
            (default: :obj:`zero`)
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 exog_size: int = 0,
                 cell: str = 'gru',
                 concat_mask: bool = True,
                 fully_connected: bool = False,
                 n_nodes: Optional[int] = None,
                 detach_input: bool = False,
                 state_init: str = 'zero'):
        super(RNNImputerModel, self).__init__()

        if fully_connected:
            assert n_nodes is not None
            input_size = input_size * n_nodes
            self._to_pattern = 'b t (n f)'
        else:
            self._to_pattern = '(b n) t f'

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.exog_size = exog_size

        if concat_mask and fully_connected:
            logger.warning("Parameter 'concat_mask' can be True only when "
                           "'fully_connected' is False.")
            concat_mask = False

        self.concat_mask = concat_mask
        self.fully_connected = fully_connected
        self.detach_input = detach_input
        self.state_init = state_init

        if cell == 'gru':
            cell = nn.GRUCell
        elif cell == 'lstm':
            cell = nn.LSTMCell
        else:
            raise NotImplementedError(f'"{cell}" cell not implemented.')

        if concat_mask:
            input_size = 2 * input_size
        input_size = input_size + exog_size
        self.rnn_cell = cell(input_size=input_size, hidden_size=hidden_size)

        self.readout = nn.Linear(hidden_size, self.input_size)

    def init_hidden_state(self, x: Tensor):
        if self.state_init == 'zero':
            return torch.zeros((x.size(0), self.hidden_size), device=x.device,
                               dtype=x.dtype)
        if self.state_init == 'noise':
            return torch.randn(x.size(0), self.hidden_size, device=x.device,
                               dtype=x.dtype)

    def _preprocess_input(self, x: Tensor, x_hat: Tensor, m: Tensor,
                          u: Optional[Tensor] = None):
        if self.detach_input:
            x_p = torch.where(m, x, x_hat.detach())
        else:
            x_p = torch.where(m, x, x_hat)

        if u is not None:
            x_p = torch.cat([x_p, u], -1)
        if self.concat_mask:
            x_p = torch.cat([x_p, m], -1)
        return x_p

    def forward(self, x: Tensor, input_mask: Tensor,
                u: Optional[Tensor] = None,
                return_hidden: bool = False) -> Union[Tensor, list]:
        """"""
        # x: [batch, time, nodes, features]
        steps, nodes = x.size(1), x.size(2)

        x = rearrange(x, f'b t n f -> {self._to_pattern}')
        input_mask = rearrange(input_mask, f'b t n f -> {self._to_pattern}')
        if u is not None:
            u = rearrange(u, f'b t n f -> {self._to_pattern}')

        h = self.init_hidden_state(x)

        hs, preds = [], []
        for s in range(steps):
            # readout phase
            x_hat = self.readout(h)
            hs.append(h)
            preds.append(x_hat)
            # state update phase
            u_t = None if u is None else u[:, s]
            x_t = self._preprocess_input(x[:, s], x_hat, input_mask[:, s], u_t)
            h = self.rnn_cell(x_t, h)

        x_hat = torch.stack(preds, 1)  # [b t (n f)] or [(b n) t f]
        h = torch.stack(hs, 1)  # [b t h] or [(b n) t h]

        x_hat = rearrange(x_hat, f'{self._to_pattern} -> b t n f', n=nodes)

        if not return_hidden:
            return x_hat

        if not self.fully_connected:
            h = rearrange(h, f'{self._to_pattern} -> b t n f', n=nodes)
        return [x_hat, h]

    def predict(self, x: Tensor, input_mask: Tensor,
                u: Optional[Tensor] = None) -> Tensor:
        """"""
        return self.forward(x=x, input_mask=input_mask, u=u,
                            return_hidden=False)


class BiRNNImputerModel(BaseModel):
    r"""Fill the blanks with a bidirectional GRU 1-step-ahead predictor."""

    def __init__(self, input_size: int, hidden_size: int = 64,
                 exog_size: int = 0,
                 cell: str = 'gru',
                 dropout=0.,
                 concat_mask: bool = True,
                 fully_connected: bool = False,
                 n_nodes: Optional[int] = None,
                 detach_input: bool = False,
                 state_init: str = 'zero'):
        super(BiRNNImputerModel, self).__init__(return_type=list)
        self.fwd_rnn = RNNImputerModel(input_size, hidden_size,
                                       exog_size=exog_size,
                                       cell=cell,
                                       concat_mask=concat_mask,
                                       n_nodes=n_nodes,
                                       fully_connected=fully_connected,
                                       detach_input=detach_input,
                                       state_init=state_init)
        self.bwd_rnn = RNNImputerModel(input_size, hidden_size,
                                       exog_size=exog_size,
                                       cell=cell,
                                       concat_mask=concat_mask,
                                       n_nodes=n_nodes,
                                       fully_connected=fully_connected,
                                       detach_input=detach_input,
                                       state_init=state_init)
        self.dropout = nn.Dropout(dropout)

        if fully_connected:
            self.read_out = nn.Linear(2 * hidden_size, input_size)
        else:
            assert n_nodes is not None
            self.read_out = nn.Sequential(
                nn.Linear(2 * hidden_size, input_size * n_nodes),
                Rearrange('... t (n h) -> ... t n h', n=n_nodes)
            )

    def forward(self, x: Tensor, input_mask: Tensor,
                u: Optional[Tensor] = None,
                return_hidden: bool = False) -> list:
        """"""
        # x: [batches, steps, nodes, features]
        x_hat_fwd, h_fwd = self.fwd_rnn(x, input_mask, u=u, return_hidden=True)
        u_rev = reverse_tensor(u, 1) if u is not None else None
        x_hat_bwd, h_bwd = self.bwd_rnn(reverse_tensor(x, 1),
                                        reverse_tensor(input_mask, 1),
                                        u=u_rev,
                                        return_hidden=True)
        x_hat_bwd = reverse_tensor(x_hat_bwd, 1)
        h_bwd = reverse_tensor(h_bwd, 1)
        h = self.dropout(torch.cat([h_fwd, h_bwd], -1))
        x_hat = self.read_out(h)
        if return_hidden:
            return [x_hat, (x_hat_fwd, x_hat_bwd), h]
        return [x_hat, (x_hat_fwd, x_hat_bwd)]

    def predict(self, x: Tensor, input_mask: Tensor,
                u: Optional[Tensor] = None) -> Tensor:
        """"""
        return self.forward(x=x, input_mask=input_mask, u=u,
                            return_hidden=False)[0]
