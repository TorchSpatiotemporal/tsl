import math
from typing import List, Optional, Union

import torch
from torch import Tensor, nn
from torch_geometric.typing import OptTensor


class NodeEmbedding(nn.Module):
    r"""Creates a table of node embeddings with the specified size.

    Args:
        n_nodes (int): Number of elements for which to store an embedding.
        emb_size (int): Size of the embedding.
        initializer (str or Tensor): Initialization methods.
            (default :obj:`'uniform'`)
        requires_grad (bool): Whether to compute gradients for the embeddings.
            (default :obj:`True`)
    """

    def __init__(self,
                 n_nodes: int,
                 emb_size: int,
                 initializer: Union[str, Tensor] = 'uniform',
                 requires_grad: bool = True):
        super(NodeEmbedding, self).__init__()
        self.n_nodes = int(n_nodes)
        self.emb_size = int(emb_size)

        if isinstance(initializer, Tensor):
            self.initializer = "from_values"
            self.register_buffer('_default_values', initializer.float())
        else:
            self.initializer = initializer
            self.register_buffer('_default_values', None)

        self.emb = nn.Parameter(Tensor(self.n_nodes, self.emb_size),
                                requires_grad=requires_grad)

        self.reset_emb()

    def __repr__(self) -> str:
        return "{}(n_nodes={}, embedding_size={})".format(
            self.__class__.__name__, self.n_nodes, self.emb_size)

    def reset_emb(self):
        with torch.no_grad():
            if self.initializer == 'uniform' or self.initializer is None:
                bound = 1.0 / math.sqrt(self.emb.size(-1))
                self.emb.data.uniform_(-bound, bound)
            elif self.initializer == 'from_values':
                self.emb.data.copy_(self._default_values)
            else:
                raise RuntimeError(
                    f"Embedding initializer '{self.initializer}'"
                    " is not supported.")

    def reset_parameters(self):
        self.reset_emb()

    def get_emb(self):
        return self.emb

    def forward(self,
                expand: Optional[List] = None,
                node_index: OptTensor = None,
                nodes_first: bool = True):
        """"""
        emb = self.get_emb()
        if node_index is not None:
            emb = emb[node_index]
        if not nodes_first:
            emb = emb.T
        if expand is None:
            return emb
        shape = [*emb.size()]
        view = [
            1 if d > 0 else shape.pop(0 if nodes_first else -1) for d in expand
        ]
        return emb.view(*view).expand(*expand)
