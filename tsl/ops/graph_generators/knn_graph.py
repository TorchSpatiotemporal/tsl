import numpy as np


def build_knn_graph(num_nodes, k):
    r"""
    Build a directed k-nearest neighbor graph.
    Args:
        num_nodes (int): number of nodes
        k (int): number of nearest neighbors

    Returns:
        `node_idx`, `edge_index`, `edge_weight`
    """
    row = node_idx = np.arange(num_nodes)
    row = np.repeat(row, k)
    # row: [0, 0, 0, 1, 1, 1, 2, 2, 2, ...]
    col = np.concatenate([
        np.random.choice(num_nodes, k, replace=False) for _ in range(num_nodes)
    ])
    # col: [3, 1, 2, 0, 2, 1, 1, 0, 3, ...]
    edge_index = np.stack((row, col))
    return node_idx, edge_index, None
