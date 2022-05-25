import math
from typing import Optional, Union, List

import torch
from torch import nn, Tensor
from torch_geometric.nn import inits
from torch_geometric.typing import OptTensor


class StaticGraphEmbedding(nn.Module):
    r"""Creates a table of embeddings with the specified size.

    Args:
        n_tokens (int): Number of elements for which to store an embedding.
        emb_size (int): Size of the embedding.
        initializer (str or Tensor): Initialization methods.
            (default :obj:`'uniform'`)
        requires_grad (bool): Whether to compute gradients for the embeddings.
            (default :obj:`True`)
        bind_to (nn.Module, optional): Bind the embedding to a nn.Module for
            lazy init. (default :obj:`None`)
        infer_tokens_from_pos (int): Index of the element of input data from
            which to infer the number of embeddings for lazy init.
            (default :obj:`0`)
        dim (int): Token dimension. (default :obj:`-2`)
    """

    def __init__(self, n_tokens: int, emb_size: int,
                 initializer: Union[str, Tensor] = 'uniform',
                 requires_grad: bool = True,
                 bind_to: Optional[nn.Module] = None,
                 infer_tokens_from_pos: int = 0,
                 dim: int = -2):
        super(StaticGraphEmbedding, self).__init__()
        assert emb_size > 0
        self.n_tokens = int(n_tokens)
        self.emb_size = int(emb_size)
        self.dim = int(dim)
        self.infer_tokens_from_pos = infer_tokens_from_pos

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

    def forward(self, expand: Optional[List] = None,
                token_index: OptTensor = None,
                tokens_first: bool = True):
        """"""
        emb = self.emb if token_index is None else self.emb[token_index]
        if not tokens_first:
            emb = emb.T
        if expand is None:
            return emb
        shape = [*emb.size()]
        view = [1 if d > 0 else shape.pop(0 if tokens_first else -1)
                for d in expand]
        return emb.view(*view).expand(*expand)
