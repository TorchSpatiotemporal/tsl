import numpy as np


def build_circle_graph(num_nodes: int, undirected: bool = False):
    """Build a directed circle graph.

    Args:
        num_nodes (int): The number of nodes.
        undirected (bool): If :obj:`True`, then each node will be connected to
            both previous and following nodes in the circle.
            (default: :obj:`False`)

    Returns:
        `node_idx`, `edge_index`, `edge_weight`
    """
    row = np.arange(num_nodes)
    col = np.roll(row, -1)
    edge_index = np.stack((row, col))
    if undirected:
        edge_index_t = np.stack((col, row))
        edge_index = np.concatenate([edge_index, edge_index_t], axis=1)
    return row, edge_index, None
