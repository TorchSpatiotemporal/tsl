import numpy as np


def build_line_graph(num_nodes):
    """
    Build a directed line graph.

    Args:
        num_nodes: number of nodes

    Returns:
        `node_idx`, `edge_index`, `edge_weight`
    """
    row = np.arange(num_nodes - 1)
    col = np.arange(1, num_nodes)
    edge_index = np.stack((row, col))
    return row, edge_index, None
