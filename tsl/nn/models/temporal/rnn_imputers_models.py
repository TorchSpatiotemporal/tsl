from typing import Optional, Union

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from tsl import logger
from tsl.nn.blocks.encoders.recurrent import RNNI
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
        n_layers (int, optional): Number of hidden layers.
            (default: :obj:`1`)
        cat_states_layers (bool): If :obj:`True`, then the states of the RNN are
            concatenated together.
            (default: :obj:`False`)
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 exog_size: int = 0,
                 cell: str = 'gru',
                 concat_mask: bool = True,
                 fully_connected: bool = False,
                 n_nodes: Optional[int] = None,
                 detach_input: bool = False,
                 n_layers: int = 1,
                 cat_states_layers: bool = False):
        super(RNNImputerModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.exog_size = exog_size
        self.concat_mask = concat_mask
        self.fully_connected = fully_connected
        self.detach_input = detach_input
        self.n_layers = n_layers
        self.cat_states_layers = cat_states_layers

        if fully_connected:
            assert n_nodes is not None
            input_size = input_size * n_nodes
            self._to_pattern = 'b t (n f)'
        else:
            self._to_pattern = '(b n) t f'

        if concat_mask and fully_connected:
            logger.warning("Parameter 'concat_mask' can be True only when "
                           "'fully_connected' is False.")
            concat_mask = False

        self.rnn = RNNI(input_size=input_size,
                        hidden_size=hidden_size,
                        exog_size=exog_size,
                        cell=cell,
                        concat_mask=concat_mask,
                        n_layers=n_layers,
                        detach_input=detach_input,
                        cat_states_layers=cat_states_layers)

    def forward(self,
                x: Tensor,
                mask: Tensor,
                u: Optional[Tensor] = None,
                return_hidden: bool = False) -> Union[Tensor, list]:
        """"""
        # x: [batch, time, nodes, features]
        nodes = x.size(2)

        x = rearrange(x, f'b t n f -> {self._to_pattern}')
        mask = rearrange(mask, f'b t n f -> {self._to_pattern}')

        if u is not None:
            if self.fully_connected:  # fc and 'b t f'
                assert u.ndim == 3, \
                    "Only graph-level exogenous are supported in " \
                    "fully connected mode."
            elif u.ndim == 3:  # no fc and 'b t f'
                u = repeat(u, f'b t f -> {self._to_pattern}', n=nodes)
            else:  # no fc and 'b t n f'
                u = rearrange(u, f'b t n f -> {self._to_pattern}')

        x_hat, h, _ = self.rnn(x, mask, u)

        x_hat = rearrange(x_hat, f'{self._to_pattern} -> b t n f', n=nodes)

        if not return_hidden:
            return x_hat

        if not self.fully_connected:
            h = rearrange(h, f'{self._to_pattern} -> b t n f', n=nodes)
        return [x_hat, h]

    def predict(self,
                x: Tensor,
                mask: Tensor,
                u: Optional[Tensor] = None) -> Tensor:
        """"""
        return self.forward(x=x, mask=mask, u=u, return_hidden=False)


class BiRNNImputerModel(BaseModel):
    r"""Fill the blanks with 1-step-ahead predictions of a bidirectional
    recurrent neural network.

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
        n_layers (int, optional): Number of hidden layers.
            (default: :obj:`1`)
        return_previous_state (bool): If :obj:`True`, then the returned states
            are shifted one-step behind the imputations.
            (default: :obj:`True`)
        cat_states_layers (bool): If :obj:`True`, then the states of the RNN are
            concatenated together.
            (default: :obj:`False`)
        dropout (float, optional): Dropout probability in the decoder.
            (default: ``0.``)
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 exog_size: int = 0,
                 cell: str = 'gru',
                 concat_mask: bool = True,
                 fully_connected: bool = False,
                 n_nodes: Optional[int] = None,
                 detach_input: bool = False,
                 n_layers: int = 1,
                 cat_states_layers: bool = False,
                 dropout: float = 0.):
        super(BiRNNImputerModel, self).__init__(return_type=list)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.exog_size = exog_size
        self.concat_mask = concat_mask
        self.fully_connected = fully_connected
        self.detach_input = detach_input
        self.n_layers = n_layers
        self.cat_states_layers = cat_states_layers

        if fully_connected:
            assert n_nodes is not None
            input_size = input_size * n_nodes
            self._to_pattern = 'b t (n f)'
        else:
            self._to_pattern = '(b n) t f'

        if concat_mask and fully_connected:
            logger.warning("Parameter 'concat_mask' can be True only when "
                           "'fully_connected' is False.")
            concat_mask = False

        self.fwd_rnn = RNNI(input_size=input_size,
                            hidden_size=hidden_size,
                            exog_size=exog_size,
                            cell=cell,
                            concat_mask=concat_mask,
                            n_layers=n_layers,
                            detach_input=detach_input,
                            cat_states_layers=cat_states_layers)
        self.bwd_rnn = RNNI(input_size=input_size,
                            hidden_size=hidden_size,
                            exog_size=exog_size,
                            cell=cell,
                            concat_mask=concat_mask,
                            flip_time=True,
                            n_layers=n_layers,
                            detach_input=detach_input,
                            cat_states_layers=cat_states_layers)

        self.dropout = nn.Dropout(dropout)

        out_size = hidden_size * (n_layers if cat_states_layers else 1)

        if fully_connected:
            assert n_nodes is not None
            self.readout = nn.Sequential(
                nn.Linear(2 * out_size, input_size),
                Rearrange('... t (n h) -> ... t n h', n=n_nodes))
        else:
            self.readout = nn.Linear(2 * out_size, self.input_size)

    def forward(self,
                x: Tensor,
                mask: Tensor,
                u: Optional[Tensor] = None,
                return_hidden: bool = False,
                return_predictions: bool = True) -> Union[Tensor, list]:
        """"""
        # x: [batch, time, nodes, features]
        nodes = x.size(2)

        x = rearrange(x, f'b t n f -> {self._to_pattern}')
        mask = rearrange(mask, f'b t n f -> {self._to_pattern}')

        if u is not None:
            if self.fully_connected:  # fc and 'b t f'
                assert u.ndim == 3, \
                    "Only graph-level exogenous are supported in " \
                    "fully connected mode."
            elif u.ndim == 3:  # no fc and 'b t f'
                u = repeat(u, f'b t f -> {self._to_pattern}', n=nodes)
            else:  # no fc and 'b t n f'
                u = rearrange(u, f'b t n f -> {self._to_pattern}')

        x_hat_fwd, h_fwd, _ = self.fwd_rnn(x, mask, u)
        x_hat_bwd, h_bwd, _ = self.bwd_rnn(x, mask, u)

        if not self.fully_connected:
            h_fwd = rearrange(h_fwd, f'{self._to_pattern} -> b t n f', n=nodes)
            h_bwd = rearrange(h_bwd, f'{self._to_pattern} -> b t n f', n=nodes)

        h = self.dropout(torch.cat([h_fwd, h_bwd], -1))
        x_hat = self.readout(h)

        if not (return_predictions or return_hidden):
            return x_hat

        res = [x_hat]

        if return_predictions:
            x_hat_fwd = rearrange(x_hat_fwd,
                                  f'{self._to_pattern} -> b t n f',
                                  n=nodes)
            x_hat_bwd = rearrange(x_hat_bwd,
                                  f'{self._to_pattern} -> b t n f',
                                  n=nodes)
            res.append((x_hat_fwd, x_hat_bwd))
        if return_hidden:
            res.append(h)
        return res

    def predict(self,
                x: Tensor,
                mask: Tensor,
                u: Optional[Tensor] = None) -> Tensor:
        """"""
        return self.forward(x=x,
                            mask=mask,
                            u=u,
                            return_hidden=False,
                            return_predictions=False)
