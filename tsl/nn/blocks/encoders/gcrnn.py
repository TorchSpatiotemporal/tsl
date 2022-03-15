import torch

from einops import rearrange


class _GraphGRUCell(torch.nn.Module):
    r"""
    Base class for implementing `GraphGRU` cells.
    """
    def forward(self, x, h, *args, **kwargs):
        """"""
        # x: [batch, nodes, channels]
        # h: [batch, nodes, channels]
        x_gates = torch.cat([x, h], dim=-1)
        r = torch.sigmoid(self.forget_gate(x_gates, *args, **kwargs))
        u = torch.sigmoid(self.update_gate(x_gates, *args, **kwargs))
        x_c = torch.cat([x, r * h], dim=-1)
        c = torch.tanh(self.candidate_gate(x_c, *args, **kwargs))
        return u * h + (1. - u) * c


class _GraphLSTMCell(torch.nn.Module):
    r"""
    Base class for implementing `GraphLSTM` cells.
    """
    def forward(self, x, hs, *args, **kwargs):
        """"""
        # x: [batch, nodes, channels]
        # hs: (h, c)
        # h: [batch, nodes, channels]
        # c: [batch, nodes, channels]
        h, c = hs
        x_gates = torch.cat([x, h], dim=-1)
        i = torch.sigmoid(self.input_gate(x_gates, *args, **kwargs))
        f = torch.sigmoid(self.forget_gate(x_gates, *args, **kwargs))
        g = torch.tanh(self.cell_gate(x_gates, *args, **kwargs))
        o = torch.sigmoid(self.output_gate(x_gates, *args, **kwargs))
        c_new = f * c + i * g
        h_new = o * torch.tan(c)
        return (h_new, c_new)


class _GraphRNN(torch.nn.Module):
    r"""
    Base class for GraphRNNs
    """
    _n_states = None

    def _init_states(self, x):
        return torch.zeros(size=(self.n_layers, x.shape[0], x.shape[-2], x.shape[-1]), device=x.device)

    def single_pass(self, x, h, *args, **kwargs):
        # x: [batch, nodes, channels]
        # h: [layers, batch, nodes, channels]
        h_new = []
        out = x
        for i, cell in enumerate(self.rnn_cells):
            out = cell(out, h[i], *args, **kwargs)
            h_new.append(out)
        return torch.stack(h_new)

    def forward(self, x, *args, h=None, **kwargs):
        # x: [batch, steps, nodes, channels]
        steps = x.size(1)
        if h is None:
            *h, = self._init_states(x)
        if not len(h):
            h = h[0]
        # temporal conv
        out = []
        for step in range(steps):
            h = self.single_pass(x[:, step], h, *args, **kwargs)
            # append hidden state of the last layer
            h_out = h[-1]
            if not isinstance(h_out, torch.Tensor):
                h_out = h_out[0]
            out.append(h_out)
        out = torch.stack(out)
        # out: [steps, batch, nodes, channels]
        out = rearrange(out, 's b n c -> b s n c')
        # h: [l b n c]
        return out, h
