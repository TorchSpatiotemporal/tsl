from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from tsl.nn.layers.recurrent.base import RNNCellBase, StateType
from tsl.utils import ensure_list


class RNNBase(nn.Module):
    r"""Base class for implementing recurrent neural networks (RNNs)."""

    def __init__(self,
                 cells: Union[RNNCellBase, List[RNNCellBase], nn.ModuleList],
                 cat_states_layers: bool = False,
                 return_only_last_state: bool = False):
        super().__init__()
        self.cat_states_layers = cat_states_layers
        self.return_only_last_state = return_only_last_state
        if not isinstance(cells, nn.ModuleList):
            cells = nn.ModuleList(ensure_list(cells))
        self.cells = cells
        self.n_layers = len(self.cells)

    def __repr__(self) -> str:
        args = [
            f'cell={self.cells[0].__class__.__name__}',
            f'return_only_last_state={self.return_only_last_state}'
        ]
        return f"{self.__class__.__name__}({', '.join(args)})"

    def reset_parameters(self):
        for cell in self.cells:
            cell.reset_parameters()

    def initialize_state(self, x: Tensor) -> List[StateType]:
        return [cell.initialize_state(x) for cell in self.cells]

    def single_pass(self, x: Tensor, h: List[StateType], *args,
                    **kwargs) -> List[StateType]:
        # x: [batch, *, channels]
        # h[i]: [batch, *, channels]
        out = []
        input = x
        for i, cell in enumerate(self.cells):
            input = h_new = cell(input, h[i], *args, **kwargs)
            if isinstance(h_new, (list, tuple)):
                input = input[0]
            out.append(h_new)
        return out

    def forward(self,
                x: Tensor,
                *args,
                h: Optional[List[StateType]] = None,
                **kwargs) -> Tuple[Tensor, List[StateType]]:
        """"""
        # x: [batch, time, *, features]
        if h is None:
            h = self.initialize_state(x)
        elif not isinstance(h, list):
            h = [h]
        # temporal conv
        out = []
        steps = x.size(1)
        for step in range(steps):
            h_out = h = self.single_pass(x[:, step], h, *args, **kwargs)
            # for multi-state rnns (e.g., LSTMs), use first state for readout
            if not isinstance(h_out[0], torch.Tensor):
                h_out = [_h[0] for _h in h_out]
            # append hidden state of the last layer
            if self.cat_states_layers:
                h_out = torch.cat(h_out, dim=-1)
            else:  # or take last layer's state
                h_out = h_out[-1]
            out.append(h_out)

        if self.return_only_last_state:
            # out: [batch, *, features]
            return out[-1]
        # out: [batch, time, *, features]
        out = torch.stack(out, dim=1)
        return out, h


class RNNIBase(RNNBase):
    r"""Base class for implementing recurrent neural networks (RNNs) for
    sequences with missing data."""

    def __init__(self,
                 cells: Union[RNNCellBase, List[RNNCellBase], nn.ModuleList],
                 detach_input: bool = False,
                 concat_mask: bool = False,
                 flip_time: bool = False,
                 cat_states_layers: bool = False):
        super().__init__(cells,
                         cat_states_layers,
                         return_only_last_state=False)
        self.detach_input = detach_input
        self.concat_mask = concat_mask
        self.flip_time = flip_time

    def state_readout(self, h: List[StateType]):
        raise NotImplementedError

    def preprocess_input(self,
                         x: Tensor,
                         x_hat: Tensor,
                         input_mask: Tensor,
                         step: int,
                         *args,
                         h: Optional[List[StateType]] = None,
                         **kwargs):
        if self.detach_input:
            x_hat = x_hat.detach()
        x_t = torch.where(input_mask, x, x_hat)
        if self.concat_mask:
            x_t = torch.cat([x_t, input_mask], -1)
        return x_t

    def postprocess_state(self, h: List[StateType]):
        # for multi-state rnns (e.g., LSTMs), use first state for readout
        if not isinstance(h[0], torch.Tensor):
            h = [_h[0] for _h in h]
        if self.cat_states_layers:  # concat rnn layers' states
            h = torch.cat(h, dim=-1)
        else:  # or take last layer's state
            h = h[-1]
        return h

    def forward(self,
                x: Tensor,
                input_mask: Tensor,
                *args,
                h: Optional[List[StateType]] = None,
                **kwargs) -> Tuple[Tensor, Tensor, List[StateType]]:
        """"""
        # x: [batch, time, *, features]
        if h is None:
            h = self.initialize_state(x)
        elif not isinstance(h, list):
            h = [h]
        # temporal conv
        h_out, x_out = [], []
        steps = x.size(1)
        steps = range(steps) if not self.flip_time else range(
            steps - 1, -1, -1)
        for step in steps:
            # readout phase
            x_hat = self.state_readout(h)
            x_out.append(x_hat)
            h_out.append(self.postprocess_state(h))
            # preprocess input to rnn's cell given readout
            x_t = self.preprocess_input(x[:, step], x_hat, input_mask[:, step],
                                        step, *args, **kwargs)
            h = self.single_pass(x_t, h, *args, **kwargs)

        if self.flip_time:
            x_out, h_out = x_out[::-1], h_out[::-1]

        x_out = torch.stack(x_out, dim=1)  # [b t * f]
        h_out = torch.stack(h_out, dim=1)  # [b t * f]

        return x_out, h_out, h
