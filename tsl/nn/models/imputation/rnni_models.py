from typing import Optional

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from tsl.nn.functional import reverse_tensor
from tsl.utils.parser_utils import str_to_bool


class RNNImputerModel(nn.Module):
    r"""Fill the blanks with a GRU 1-step-ahead predictor."""

    def __init__(self, input_size, hidden_size, exog_size=0,
                 cell='gru',
                 concat_mask=True,
                 n_nodes: Optional[int] = None,
                 process_nodes_independently=False,
                 detach_input=False,
                 state_init='zero'):
        super(RNNImputerModel, self).__init__()

        if process_nodes_independently:
            self._to_pattern = '(b n) s c'
        else:
            assert n_nodes is not None
            input_size = input_size * n_nodes
            self._to_pattern = 'b s (n c)'

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.exog_size = exog_size

        self.concat_mask = concat_mask
        self.process_nodes_independently = process_nodes_independently
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

    def init_hidden_state(self, x):
        if self.state_init == 'zero':
            return torch.zeros((x.size(0), self.hidden_size), device=x.device,
                               dtype=x.dtype)
        if self.state_init == 'noise':
            return torch.randn(x.size(0), self.hidden_size, device=x.device,
                               dtype=x.dtype)

    def _preprocess_input(self, x, x_hat, m, u):
        if self.detach_input:
            x_p = torch.where(m, x, x_hat.detach())
        else:
            x_p = torch.where(m, x, x_hat)

        if u is not None:
            x_p = torch.cat([x_p, u], -1)
        if self.concat_mask:
            x_p = torch.cat([x_p, m], -1)
        return x_p

    def forward(self, x, mask, u=None, return_hidden=False):
        # x: [batches, steps, nodes, features]
        steps, nodes = x.size(1), x.size(2)

        x = rearrange(x, f'b s n c -> {self._to_pattern}')
        mask = rearrange(mask, f'b s n c -> {self._to_pattern}')
        if u is not None:
            u = rearrange(u, f'b s n c -> {self._to_pattern}')

        h = self.init_hidden_state(x)
        x_hat = self.readout(h)
        hs = [h]
        preds = [x_hat]
        for s in range(steps - 1):
            u_t = None if u is None else u[:, s]
            x_t = self._preprocess_input(x[:, s], x_hat, mask[:, s], u_t)
            h = self.rnn_cell(x_t, h)
            x_hat = self.readout(h)
            hs.append(h)
            preds.append(x_hat)

        x_hat = torch.stack(preds, 1)  # [b s (n c)] or [(b n) s c]
        h = torch.stack(hs, 1)  # [b s h] or [(b n) s h]

        x_hat = rearrange(x_hat, f'{self._to_pattern} -> b s n c', n=nodes)

        if not return_hidden:
            return x_hat

        if self.process_nodes_independently:
            h = rearrange(h, f'{self._to_pattern} -> b s n c', n=nodes)
        return x_hat, h

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--input-size', type=int)
        parser.add_argument('--hidden-size', type=int)
        parser.add_argument('--exog-size', type=int, default=0)
        parser.add_argument('--cell', type=str, default='gru')
        parser.add_argument('--concat-mask', type=str_to_bool, nargs='?',
                            const=True, default=True)
        parser.add_argument('--detach-input', type=str_to_bool, nargs='?',
                            const=True, default=False)
        parser.add_argument('--process-nodes-independently', type=str_to_bool,
                            nargs='?', const=True, default=False)
        parser.add_argument('--state-init', type=str, default='zero')
        return parser


class BiRNNImputerModel(nn.Module):
    r"""Fill the blanks with a bidirectional GRU 1-step-ahead predictor."""

    def __init__(self, input_size, hidden_size, exog_size=0,
                 cell='gru',
                 concat_mask=True,
                 n_nodes: Optional[int] = None,
                 process_nodes_independently=False,
                 detach_input=False,
                 state_init='zero',
                 dropout=0.):
        super(BiRNNImputerModel, self).__init__()
        self.fwd_rnn = RNNImputerModel(input_size, hidden_size,
                                       exog_size=exog_size,
                                       cell=cell,
                                       concat_mask=concat_mask,
                                       n_nodes=n_nodes,
                                       process_nodes_independently=process_nodes_independently,
                                       detach_input=detach_input,
                                       state_init=state_init)
        self.bwd_rnn = RNNImputerModel(input_size, hidden_size,
                                       exog_size=exog_size,
                                       cell=cell,
                                       concat_mask=concat_mask,
                                       n_nodes=n_nodes,
                                       process_nodes_independently=process_nodes_independently,
                                       detach_input=detach_input,
                                       state_init=state_init)
        self.dropout = nn.Dropout(dropout)

        if process_nodes_independently:
            self.read_out = nn.Linear(2 * hidden_size, input_size)
        else:
            assert n_nodes is not None
            self.read_out = nn.Sequential(
                nn.Linear(2 * hidden_size, input_size * n_nodes),
                Rearrange('b s (n h) -> b s n h', n=n_nodes)
            )

    def forward(self, x, mask, u=None, return_hidden=False):
        # x: [batches, steps, nodes, features]
        x_hat_fwd, h_fwd = self.fwd_rnn(x, mask, u=u, return_hidden=True)
        u_rev = reverse_tensor(u, 1) if u is not None else None
        x_hat_bwd, h_bwd = self.bwd_rnn(reverse_tensor(x, 1),
                                        reverse_tensor(mask, 1),
                                        u=u_rev,
                                        return_hidden=True)
        x_hat_bwd = reverse_tensor(x_hat_bwd, 1)
        h_bwd = reverse_tensor(h_bwd, 1)
        h = self.dropout(torch.cat([h_fwd, h_bwd], -1))
        x_hat = self.read_out(h)
        if return_hidden:
            return x_hat, (x_hat_fwd, x_hat_bwd), h
        return x_hat, (x_hat_fwd, x_hat_bwd)

    @staticmethod
    def add_model_specific_args(parser):
        RNNImputerModel.add_model_specific_args(parser)
        parser.add_argument('--dropout', type=float, default=0.)
        return parser
