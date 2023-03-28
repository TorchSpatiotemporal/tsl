import numpy as np


def _create_community():
    r"""

    .. code::

                  2
                 / \
                1 - 4
               / \ / \
        ... - 0 - 3 - 5 - ...
    """
    nodes = np.arange(6)
    # yapf: disable
    edges = np.asarray([[0, 1], [1, 2], [3, 4],  # slashes
                        [1, 3], [2, 4], [4, 5],  # backslashes
                        [0, 3], [1, 4], [3, 5]])  # horizontal
    # yapf: enable
    return nodes, edges


def build_tri_community_graph(num_communities):
    r"""
    A family of planar graphs composed of a number of communities.
    Each community takes the form of a 6-node triangle:

    .. code::

            2
           / \
          1 - 4
         / \ / \
        0 - 3 - 5

    All communities are arranged as a line

    .. code::

        c0 - c1 - c2 -  ....

    Args:
        num_communities (int): number of communities in the created graph.

    Returns:
        tuple: Returns a tuple containing the list of nodes, list of edges and
            list of edge weights (which is :obj:`None`).
    """
    nodes = []
    edges = []
    for c in range(num_communities):
        n, e = _create_community()
        n += c * 6
        e += c * 6
        nodes += list(n)
        edges += list(e)
        if c > 0:
            edges.append([n[0] - 1, n[0]])  # connect the two communities
    node_idx = np.stack(nodes, 0)
    edge_idx = np.stack(edges, 1)
    edge_idx = np.concatenate([edge_idx, edge_idx[::-1]], 1)
    edge_idx = np.unique(edge_idx, axis=1)
    return node_idx, edge_idx, None
