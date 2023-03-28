from typing import List, Optional, Tuple

import torch
from einops import rearrange
from torch import Tensor, nn

from tsl.nn.layers.recurrent import GRUCell, LSTMCell, StateType
from tsl.nn.utils import maybe_cat_exog

from .base import RNNIBase


class RNN(nn.Module):
    """Simple RNN encoder with optional linear readout.

    Args:
        input_size (int): Input size.
        hidden_size (int): Units in the hidden layers.
        exog_size (int, optional): Size of the optional exogenous variables.
        output_size (int, optional): Size of the optional readout.
        n_layers (int, optional): Number of hidden layers.
            (default: ``1``)
        cell (str, optional): Type of cell that should be use (options:
            ``'gru'``, ``'lstm'``). (default: ``'gru'``)
        dropout (float, optional): Dropout probability.
            (default: ``0.``)
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 exog_size: int = None,
                 output_size: int = None,
                 n_layers: int = 1,
                 return_only_last_state: bool = False,
                 cell: str = 'gru',
                 bias: bool = True,
                 dropout: float = 0.,
                 **kwargs):
        super(RNN, self).__init__()

        self.return_only_last_state = return_only_last_state

        if cell == 'gru':
            cell = nn.GRU
        elif cell == 'lstm':
            cell = nn.LSTM
        else:
            raise NotImplementedError(f'"{cell}" cell not implemented.')

        if exog_size is not None:
            input_size += exog_size

        self.rnn = cell(input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=n_layers,
                        bias=bias,
                        dropout=dropout)

        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.register_parameter('readout', None)

    def forward(self, x: Tensor, u: Optional[Tensor] = None):
        """Process the input sequence :obj:`x` with optional exogenous variables
        :obj:`u`.

        Args:
            x (Tensor): Input data.
            u (Tensor): Exogenous data.

        Shapes:
            x: :math:`(B, T, N, F_x)` where :math:`B` is the batch dimension,
                :math:`T` is the number of time steps, :math:`N` is the number
                of nodes, and :math:`F_x` is the number of input features.
            u: :math:`(B, T, N, F_u)` or :math:`(B, T, F_u)` where :math:`B` is
                the batch dimension, :math:`T` is the number of time steps,
                :math:`N` is the number of nodes (optional), and :math:`F_u` is
                the number of exogenous features.
        """
        # x: [batches, steps, nodes, features]
        x = maybe_cat_exog(x, u)
        b, *_ = x.size()
        x = rearrange(x, 'b s n f -> s (b n) f')
        x, *_ = self.rnn(x)
        # [steps batches * nodes, features] -> [steps batches, nodes, features]
        x = rearrange(x, 's (b n) f -> b s n f', b=b)
        if self.return_only_last_state:
            x = x[:, -1]
        if self.readout is not None:
            return self.readout(x)
        return x


class RNNI(RNNIBase):
    """RNN encoder for sequences with missing data.

    Args:
        input_size (int): Input size.
        hidden_size (int): Units in the hidden layers.
        exog_size (int): Size of the optional exogenous variables.
            (default: ``0.``)
        cell (str): Type of recurrent cell to be used, one of [:obj:`gru`,
            :obj:`lstm`].
            (default: :obj:`gru`)
        concat_mask (bool): If :obj:`True`, then the input tensor is
            concatenated to the mask when fed to the RNN cell.
            (default: :obj:`True`)
        flip_time (bool): If :obj:`True`, then the time is folded in the
            backward direction.
            (default: :obj:`False`)
        n_layers (int, optional): Number of hidden layers.
            (default: :obj:`1`)
        detach_input (bool): If :obj:`True`, call :meth:`~torch.Tensor.detach`
            on predictions before they are used to fill the gaps, breaking the
            error backpropagation.
            (default: :obj:`False`)
        cat_states_layers (bool): If :obj:`True`, then the states of the RNN are
            concatenated together.
            (default: :obj:`False`)
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 exog_size: int = 0,
                 cell: str = 'gru',
                 concat_mask: bool = True,
                 flip_time: bool = False,
                 n_layers: int = 1,
                 detach_input: bool = False,
                 cat_states_layers: bool = False):

        if cell == 'gru':
            cell = GRUCell
        elif cell == 'lstm':
            cell = LSTMCell
        else:
            raise NotImplementedError(f'"{cell}" cell not implemented.')

        self.input_size = input_size
        self.hidden_size = hidden_size

        if concat_mask:
            input_size = 2 * input_size
        input_size = input_size + exog_size

        cells = [
            cell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(n_layers)
        ]

        super(RNNI, self).__init__(cells, detach_input, concat_mask, flip_time,
                                   cat_states_layers)
        self.readout = nn.Linear(hidden_size, self.input_size)

    def state_readout(self, h: List[StateType]):
        return self.readout(h[-1])

    def preprocess_input(self,
                         x: Tensor,
                         x_hat: Tensor,
                         input_mask: Tensor,
                         step: int,
                         u: Optional[Tensor] = None,
                         h: Optional[List[StateType]] = None):
        x_t = super().preprocess_input(x, x_hat, input_mask, step)
        if u is not None:
            x_t = torch.cat([x_t, u[:, step]], -1)
        return x_t

    def single_pass(self, x: Tensor, h: List[StateType], *args,
                    **kwargs) -> List[StateType]:
        return super().single_pass(x, h)

    def forward(self, x: Tensor, input_mask: Tensor, u: Optional[Tensor] = None,
                h: Optional[List[StateType]] = None) \
            -> Tuple[Tensor, Tensor, List[StateType]]:
        """Process the input sequence :obj:`x` with optional exogenous variables
        :obj:`u`.

        Args:
            x (Tensor): Input data.
            u (Tensor): Exogenous data.

        Shapes:
            x: :math:`(B, T, N, F_x)` where :math:`B` is the batch dimension,
                :math:`T` is the number of time steps, :math:`N` is the number
                of nodes, and :math:`F_x` is the number of input features.
            u: :math:`(B, T, N, F_u)` or :math:`(B, T, F_u)` where :math:`B` is
                the batch dimension, :math:`T` is the number of time steps,
                :math:`N` is the number of nodes (optional), and :math:`F_u` is
                the number of exogenous features.
        """
        return super().forward(x, input_mask, u=u, h=h)
