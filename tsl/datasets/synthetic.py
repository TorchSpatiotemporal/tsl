import math

import os

import numpy as np
import torch
from torch import nn

from tsl.datasets import TabularDataset

import tsl
from tsl.typing import SparseTensArray
from tsl.ops.connectivity import parse_connectivity
from tsl.utils.casting import torch_to_numpy


class GaussianNoiseSyntheticDataset(TabularDataset):
    r"""
    A generator of synthetic datasets from an input model and input graph.

    The input model must be implemented as a torch model and must return the observation at the next step and the hidden
    state for the next step (can be None). Gaussian noise will be added to the output of the model at each step.

     Args:
        num_features (int): Number of features in the generated dataset.
        num_nodes (int): Number of nodes in the graph.
        num_steps (int): Number of steps to generate.
        connectivity (SparseTensArray): Connectivity of the underlying graph.
        model (optional, nn.Module): Model used to generate data. If `None`, it will attempt to create model from
                    `model_class` and `model_kwargs`.
        model_class (optional, nn.Module): Class of the model used to generate the data.
        model_kwargs (optional, nn.Module): Keyword arguments to initialize the model.
        sigma_noise (float): Standard deviation of the noise.
        name (optional, str): Name for the generated dataset.
        seed (optional, int): Seed for the random number generator.
    """

    seed: int = None

    def __init__(self,
                 num_features: int,
                 num_nodes: int,
                 num_steps: int,
                 connectivity: SparseTensArray,
                 min_window: int = 1,
                 model: nn.Module = None,
                 model_class=None,
                 model_kwargs=None,
                 sigma_noise=.2,
                 name=None,
                 seed=None,
                 **kwargs):
        self.name = name
        self._num_nodes = num_nodes
        self._num_features = num_features
        self._num_steps = num_steps
        self._min_window = min_window
        if seed is not None:
            self.seed = seed

        if model is not None:
            self.model = model
        else:
            self.model = model_class(**model_kwargs)
        self.sigma_noise = sigma_noise
        if connectivity is not None:
            self.connectivity = parse_connectivity(
                connectivity,
                target_layout='edge_index',
                num_nodes=num_nodes
            )
        else:
            self.connectivity = None

        target, optimal_pred, mask = self.load()
        super().__init__(target=target,
                         mask=mask,
                         name=name,
                         **kwargs)

        self.add_covariate('optimal_pred', optimal_pred, 't n f')

    def load_raw(self, *args, **kwargs):
        return self.generate_data(self.seed)

    @property
    def mae_optimal_model(self):
        """ E[|X|] of a Gaussian X"""
        return math.sqrt(2.0 / math.pi) * self.sigma_noise

    def generate_data(self, seed=None):
        r"""
        """
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        # initialize with noise
        x = torch.empty((self._num_steps + self._min_window,
                         self._num_nodes,
                         self._num_features)).normal_(generator=rng) * self.sigma_noise

        y_opt = torch.empty((self._num_steps,
                             self._num_nodes,
                             self._num_features))

        if self.connectivity is None:
            edge_index = edge_weight = None
        else:
            edge_index, edge_weight = self.connectivity

        with torch.no_grad():
            h_t = None
            for t in range(self._min_window, self._min_window + self._num_steps):
                x_t, h_t = self.model(x[None, t - self._min_window: t],
                                      h=h_t,
                                      t=t,
                                      edge_index=edge_index,
                                      edge_weight=edge_weight)
                y_opt[t - self._min_window:t + 1 - self._min_window] = x_t[0]
                # add noise
                x_t = x_t + torch.zeros_like(x_t).normal_(generator=rng) * self.sigma_noise
                x[t:t + 1] = x_t[0]

        x = torch_to_numpy(x[self._min_window:])
        y_opt = torch_to_numpy(y_opt)
        return x, y_opt, np.ones_like(x)

    def get_connectivity(self, layout: str = 'edge_index'):
        if self.connectivity is None:
            return self.connectivity
        return parse_connectivity(connectivity=self.connectivity,
                                  target_layout=layout,
                                  num_nodes=self.n_nodes)
