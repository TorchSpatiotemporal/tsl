import numpy as np


def build_circle_graph(num_nodes):
    """
    Build a directed circle graph.

    Args:
        num_nodes: number of nodes

    Returns:
        `node_idx`, `edge_index`, `edge_weight`
    """
    row = np.arange(num_nodes)
    col = np.roll(row, -1)
    edge_index = np.stack((row, col))
    return row, edge_index, None
