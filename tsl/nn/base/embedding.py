import math

import torch
from torch import nn, Tensor
from torch_geometric.nn import inits


class StaticGraphEmbedding(nn.Module):
    r"""Creates a table of embeddings with the specified size.

    Args:
        n_tokens (int): Number of elements for which to store an embedding.
        emb_size (int): Size of the embedding.
        dim (int): Node dimension.
        bind_to (nn.Module, optional): Bind the embedding to an nn.Module for lazy init.
        initializer (str): Initialization methods.
        requires_grad (bool): Whether to compute gradients w.r.t. the embeddings.
        infer_nodes_from_pos (int): Index of the element of input data from which to infer the number of embeddings for lazy initt.
    """
    def __init__(self, n_tokens, emb_size,
                 dim=-2,
                 bind_to=None,
                 initializer='uniform',
                 requires_grad=True,
                 infer_nodes_from_pos=0):
        super(StaticGraphEmbedding, self).__init__()
        assert emb_size > 0
        self.n_tokens = int(n_tokens)
        self.emb_size = int(emb_size)
        self.dim = int(dim)
        self.infer_tokens_from_pos = infer_nodes_from_pos

        if isinstance(initializer, Tensor):
            self.initializer = "from_values"
            self.register_buffer('_default_values', initializer.float())
        else:
            self.initializer = initializer
            self.register_buffer('_default_values', None)

        if self.n_tokens > 0:
            self.emb = nn.Parameter(Tensor(self.n_tokens, self.emb_size),
                                    requires_grad=requires_grad)
        else:
            assert isinstance(bind_to, nn.Module)
            self.emb = nn.parameter.UninitializedParameter(
                requires_grad=requires_grad)
            bind_to._hook = bind_to.register_forward_pre_hook(
                self.initialize_parameters)

        self.reset_parameters()

    def reset_parameters(self):
        if self.n_tokens > 0:
            if self.initializer == 'from_values':
                self.emb.data = self._default_values.data
            if self.initializer == 'glorot':
                inits.glorot(self.emb)
            elif self.initializer == 'uniform' or self.initializer is None:
                inits.uniform(self.emb_size, self.emb)
            elif self.initializer == 'kaiming_normal':
                nn.init.kaiming_normal_(self.emb, nonlinearity='relu')
            elif self.initializer == 'kaiming_uniform':
                inits.kaiming_uniform(self.emb, fan=self.emb_size,
                                      a=math.sqrt(5))
            else:
                raise RuntimeError(f"Embedding initializer '{self.initializer}'"
                                   " is not supported")

    def extra_repr(self) -> str:
        return f"n_tokens={self.n_tokens}, embedding_size={self.emb_size}"

    @torch.no_grad()
    def initialize_parameters(self, module, input):
        if isinstance(self.emb, torch.nn.parameter.UninitializedParameter):
            self.n_tokens = input[self.infer_tokens_from_pos].size(self.dim)
            self.emb.materialize((self.n_tokens, self.emb_size))
            self.reset_parameters()
        module._hook.remove()
        delattr(module, '_hook')

    def forward(self, expand=None, nodes_first=True):
        """"""
        if expand is None:
            return self.emb if nodes_first else self.emb.T
        shape = [self.n_tokens, self.emb_size]
        view = [1 if d > 0 else shape.pop(0 if nodes_first else -1)
                for d in expand]
        return self.emb.view(*view).expand(*expand)

