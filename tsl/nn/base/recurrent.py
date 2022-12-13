from typing import Union, Sequence, Tuple, List, Any, Optional

import torch
from torch import Tensor, nn

from tsl.utils import ensure_list


class GRUCell(nn.Module):
    """Base class for implementing gated recurrent unit (GRU) cells."""

    def __init__(self, hidden_size: int,
                 forget_gate: nn.Module,
                 update_gate: nn.Module,
                 candidate_gate: nn.Module):
        super().__init__()
        self.hidden_size = hidden_size
        self.forget_gate = forget_gate
        self.update_gate = update_gate
        self.candidate_gate = candidate_gate

    def reset_parameters(self):
        self.forget_gate.reset_parameters()
        self.update_gate.reset_parameters()
        self.candidate_gate.reset_parameters()

    def initialize_state(self, x) -> Tensor:
        return torch.zeros(x.size(0), self.hidden_size,
                           dtype=x.dtype, device=x.device)

    def forward(self, x: Tensor, h: Tensor,
                *args, **kwargs) -> Tensor:
        """"""
        # x: [batch, *, channels]
        # h: [batch, *, channels]
        x_gates = torch.cat([x, h], dim=-1)
        r = torch.sigmoid(self.forget_gate(x_gates, *args, **kwargs))
        u = torch.sigmoid(self.update_gate(x_gates, *args, **kwargs))
        x_c = torch.cat([x, r * h], dim=-1)
        c = torch.tanh(self.candidate_gate(x_c, *args, **kwargs))
        h_new = u * h + (1. - u) * c
        return h_new


class GraphGRUCell(GRUCell):

    def initialize_state(self, x) -> Tensor:
        return torch.zeros(x.size(0), x.size(-2), self.hidden_size,
                           dtype=x.dtype, device=x.device)


class LSTMCell(nn.Module):
    """Base class for implementing long short-term memory (LSTM) cells."""

    def __init__(self, hidden_size: int,
                 input_gate: nn.Module,
                 forget_gate: nn.Module,
                 cell_gate: nn.Module,
                 output_gate: nn.Module):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_gate = input_gate
        self.forget_gate = forget_gate
        self.cell_gate = cell_gate
        self.output_gate = output_gate

    def reset_parameters(self):
        self.input_gate.reset_parameters()
        self.forget_gate.reset_parameters()
        self.cell_gate.reset_parameters()
        self.output_gate.reset_parameters()

    def initialize_state(self, x) -> Tuple[Tensor, Tensor]:
        return (torch.zeros(x.size(0), self.hidden_size,
                            dtype=x.dtype, device=x.device),
                torch.zeros(x.size(0), self.hidden_size,
                            dtype=x.dtype, device=x.device))

    def forward(self, x: Tensor, hc: Tuple[Tensor, Tensor],
                *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """"""
        # x: [batch, *, channels]
        # hc: (h=[batch, *, channels], c=[batch, *, channels])
        h, c = hc
        x_gates = torch.cat([x, h], dim=-1)
        i = torch.sigmoid(self.input_gate(x_gates, *args, **kwargs))
        f = torch.sigmoid(self.forget_gate(x_gates, *args, **kwargs))
        g = torch.tanh(self.cell_gate(x_gates, *args, **kwargs))
        o = torch.sigmoid(self.output_gate(x_gates, *args, **kwargs))
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class GraphLSTMCell(LSTMCell):

    def initialize_state(self, x) -> Tuple[Tensor, Tensor]:
        return (torch.zeros(x.size(0), x.size(-2), self.hidden_size,
                            dtype=x.dtype, device=x.device),
                torch.zeros(x.size(0), x.size(-2), self.hidden_size,
                            dtype=x.dtype, device=x.device))


class RNNBase(nn.Module):
    r"""Base class for implementing recurrent neural networks (RNNs)."""

    def __init__(self, cells: Union[nn.Module, nn.ModuleList, List],
                 cat_states_layers: bool = False,
                 return_only_last_state: bool = False):
        super().__init__()
        self.cat_states_layers = cat_states_layers
        self.return_only_last_state = return_only_last_state
        if not isinstance(cells, nn.ModuleList):
            cells = nn.ModuleList(ensure_list(cells))
        self.cells = cells
        self.n_layers = len(self.cells)

    def reset_parameters(self):
        for cell in self.cells:
            cell.reset_parameters()

    def initialize_state(self, x: Tensor) -> List[Any]:
        return [cell.initialize_state(x) for cell in self.cells]

    def single_pass(self, x: Tensor, h: Union[Tensor, Sequence[Tensor]],
                    *args, **kwargs) -> List[Any]:
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

    def forward(self, x: Tensor, *args, h: Optional[Any] = None,
                **kwargs) -> Union[Tensor, Tuple[Tensor, Any]]:
        # x: [batch, time, *, features]
        if h is None:
            h = self.initialize_state(x)
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
