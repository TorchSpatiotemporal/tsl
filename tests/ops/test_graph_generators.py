import numpy as np

from tsl.ops.graph_generators import (build_circle_graph, build_knn_graph,
                                      build_line_graph,
                                      build_tri_community_graph)


def test_build_knn_graph():
    num_nodes = 10
    k = 5
    node_idx, edge_index, edge_weight = build_knn_graph(num_nodes, k)
    assert node_idx.shape == (num_nodes, )
    assert edge_index.shape == (2, num_nodes * k)
    assert edge_weight is None
    # Check that there are 5 outgoing edges for each node
    assert np.all(np.bincount(edge_index[0]) == k)


def test_build_circle_graph():
    num_nodes = 10
    node_idx, edge_index, edge_weight = build_circle_graph(num_nodes)
    assert node_idx.shape == (num_nodes, )
    assert edge_index.shape == (2, num_nodes)
    assert edge_weight is None
    # Check 1 outgoing and 1 ingoing edge for each node
    assert np.all(np.bincount(edge_index[0]) == 1)
    assert np.all(np.bincount(edge_index[1]) == 1)
    # Check no isolated nodes
    assert np.unique(edge_index[1]).size == num_nodes


def test_build_line_graph():
    num_nodes = 10
    node_idx, edge_index, edge_weight = build_line_graph(num_nodes)
    assert node_idx.shape == (num_nodes - 1, )
    assert edge_index.shape == (2, num_nodes - 1)
    assert edge_weight is None
    # Check at most 1 outgoing edge for each node
    assert np.all(np.bincount(edge_index[0]) == 1)
    # Check 1 ingoing edge for each node (except the first one)
    assert np.all(np.bincount(edge_index[1])[1:] == 1)
    # Check only 1 isolated node
    assert np.unique(edge_index[1]).size == num_nodes - 1


def test_build_tri_community_graph_single():
    node_idx, edge_index, edge_weight = build_tri_community_graph(1)
    # each community is a 6-node triangle with 9 (undirected) edges
    assert node_idx.shape == (6, )
    assert np.array_equal(node_idx, np.arange(6))
    assert edge_index.shape == (2, 18)  # 9 undirected -> 18 directed
    assert edge_weight is None
    # symmetric (undirected) edge set
    assert np.array_equal(np.unique(edge_index, axis=1),
                          np.unique(edge_index[::-1], axis=1))


def test_build_tri_community_graph_multiple():
    num_communities = 3
    node_idx, edge_index, edge_weight = build_tri_community_graph(
        num_communities)
    assert node_idx.shape == (6 * num_communities, )
    assert np.array_equal(node_idx, np.arange(6 * num_communities))
    # 9 edges per community + 1 connector between consecutive communities
    n_undirected = 9 * num_communities + (num_communities - 1)
    assert edge_index.shape == (2, 2 * n_undirected)
    # graph is connected: every node appears as an edge endpoint
    assert np.unique(edge_index).size == 6 * num_communities
