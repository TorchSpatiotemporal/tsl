import numpy as np
from numpy.random import Generator


def build_knn_graph(num_nodes: int,
                    k: int,
                    include_self: bool = True,
                    rng: Generator = None):
    r"""Build a directed k-nearest neighbor graph.

    Args:
        num_nodes (int): Number of nodes.
        k (int): Number of nearest neighbors per node.
        include_self (bool): If :obj:`False`, then self-loops are not
            considered as candidates.
            (default: :obj:`True`)
        rng (Generator, optional): The :class:`~numpy.random.Generator` to be
            used for choosing the neighbors.
            (default: :obj:`None`)

    Returns:
        `node_idx`, `edge_index`, `edge_weight`
    """
    row = node_idx = np.arange(num_nodes)
    row = np.repeat(row, k)
    # row: [0, 0, 0, 1, 1, 1, 2, 2, 2, ...]
    if rng is None:
        rng = np.random.default_rng()
    node_set = np.arange(num_nodes)
    # if not include_self, remove central nodes from neighbor candidates
    col = np.concatenate([
        rng.choice(node_set if include_self else np.delete(node_set, i),
                   size=k,
                   replace=False) for i in range(num_nodes)
    ])
    # col: [3, 1, 2, 0, 2, 1, 1, 0, 3, ...]
    edge_index = np.stack((row, col))
    return node_idx, edge_index, None
