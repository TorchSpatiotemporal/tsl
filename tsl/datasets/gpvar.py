import os

import numpy as np
import torch

from tsl.nn.layers.graph_convs.gpvar import GraphPolyVAR

from tsl.datasets import GaussianNoiseSyntheticDataset
from tsl.ops.graph_generators import build_tri_community_graph

import tsl


def _gcn_gso(edge_index, num_nodes):
    """
    Graph shift operator based on the GCN normalized adjacency matrix (see Kipf et al., "Semi-supervised classification
    with graph convolutional networks."  ICLR 2017)..
    """
    from torch_geometric.utils import add_self_loops, degree
    edge_index, _ = add_self_loops(edge_index=edge_index, num_nodes=num_nodes)
    row, col = edge_index
    deg = degree(col, num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return edge_index, edge_weight


class _GPVAR(GraphPolyVAR):
    def forward(self, x, edge_index, edge_weight, h=None):
        out = super(_GPVAR, self).forward(x, edge_index, edge_weight)
        return torch.tanh(out), None


class GPVARDataset(GaussianNoiseSyntheticDataset):
    """
    A synthetic dataset generated from a GraphPolyVarFilter filter on a
    TriCommunityGraph graph.
    """

    def __init__(self,
                 num_communities,
                 num_steps,
                 filter_params,
                 sigma_noise=.2,
                 name=None):
        if name is None:
            self.name = f"GP-VAR"
        else:
            self.name = name
        node_idx, edge_index, _ = build_tri_community_graph(num_communities=num_communities)
        num_nodes = len(node_idx)
        connectivity = _gcn_gso(torch.tensor(edge_index), num_nodes)

        filter = _GPVAR.from_params(filter_params=torch.tensor(filter_params),
                                    activation='tanh')
        super(GPVARDataset, self).__init__(num_features=1,
                                           num_nodes=num_nodes,
                                           num_steps=num_steps,
                                           connectivity=connectivity,
                                           min_window=filter.temporal_order,
                                           model=filter,
                                           sigma_noise=sigma_noise,
                                           name=name)


class GPVARDatasetAZ(GPVARDataset):
    seed = 1234
    NUM_COMMUNITIES = 5
    NUM_STEPS = 30000
    SIGMA_NOISE = 0.2

    def __init__(self, root=None):
        self.root = root
        filter_params = torch.tensor([[5, 2],
                                      [-4, 6],
                                      [-1, 0]], dtype=torch.float32)
        super(GPVARDatasetAZ, self).__init__(num_communities=self.NUM_COMMUNITIES,
                                             num_steps=self.NUM_STEPS,
                                             filter_params=filter_params,
                                             sigma_noise=self.SIGMA_NOISE,
                                             name='GPVAR-AZ-Dataset')

    @property
    def required_file_names(self):
        return [f'GPVAR_AZ.npy']

    def build(self) -> None:
        x, _ = self.generate_data()
        np.save(self.required_files_paths[0], x)

    def load_raw(self, *args, **kwargs):
        self.maybe_build()
        x = np.load(self.required_files_paths[0])
        return x, np.ones_like(x)


if __name__ == '__main__':
    dataset = GPVARDatasetAZ()
    print('done')