import math
from typing import Mapping, Type

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from tsl.datasets import TabularDataset
from tsl.ops.connectivity import parse_connectivity
from tsl.typing import SparseTensArray
from tsl.utils.casting import torch_to_numpy
from tsl.utils.python_utils import foo_signature


class GaussianNoiseSyntheticDataset(TabularDataset):
    r"""A generator of synthetic datasets from an input model and input graph.

    The input model must be implemented as a :class:`torch.nn.Module` and must
    return the observation at the next step and (optionally) the hidden state
    for the next step. Gaussian noise will be added to the output of the model
    at each step.

    Args:
        num_features (int): Number of features in the generated dataset.
        num_nodes (int): Number of nodes in the graph.
        num_steps (int): Number of steps to generate.
        connectivity (SparseTensArray): Connectivity of the underlying graph.
        model (torch.nn.Module): Model used to generate data. If :obj:`None`,
            it will attempt to create model from ``model_class`` and
            ``model_kwargs``.
        model_class (type, optional): Class of the model used to generate the
            data.
            (default: :obj:`None`)
        model_kwargs (dict, optional): Keyword arguments needed to initialize
            the model.
            (default: :obj:`None`)
        sigma_noise (float): Standard deviation of the noise.
            (default: :obj:`0.2`)
        name (str, optional): Name for the generated dataset.
            (default: :obj:`None`)
        seed (int, optional): Seed for the random number generator.
            (default: :obj:`None`)
    """

    seed: int = None

    def __init__(self,
                 num_features: int,
                 num_nodes: int,
                 num_steps: int,
                 connectivity: SparseTensArray,
                 min_window: int = 1,
                 model: nn.Module = None,
                 model_class: Type = None,
                 model_kwargs: Mapping = None,
                 sigma_noise: float = .2,
                 name: str = None,
                 seed: int = None,
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

        self._model_forward_signature = foo_signature(model.forward)

        self.sigma_noise = sigma_noise
        if connectivity is not None:
            self.connectivity = parse_connectivity(connectivity,
                                                   target_layout='edge_index',
                                                   num_nodes=num_nodes)
        else:
            self.connectivity = None

        target, optimal_pred, mask = self.load()
        super().__init__(target=target, mask=mask, name=name, **kwargs)

        self.add_covariate('optimal_pred', optimal_pred, 't n f')

    def load_raw(self, *args, **kwargs):
        return self.generate_data(self.seed)

    @property
    def mae_optimal_model(self):
        r""":math:`\mathbb{E}[|\mathbf{X}|]` of a Gaussian
        :math:`\mathbf{X} \sim \mathcal{N}(0, \sigma^2)`, computed as
        :math:`\varepsilon = \sqrt{\frac{2}{\pi}}\sigma`.
        """
        return math.sqrt(2.0 / math.pi) * self.sigma_noise

    def _filter_forward_kwargs(self, kwargs):
        if not self._model_forward_signature['has_kwargs']:
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in self._model_forward_signature['signature']
            }
        return kwargs

    def _model_forward(self, *args, **kwargs):
        kwargs = self._filter_forward_kwargs(kwargs)
        out = self.model(*args, **kwargs)
        if len(out) != 2:
            return out, None
        # Assumes that if the output has length 2,
        # then it will contain [output, hidden_state].
        return out

    def generate_data(self, seed=None):
        """"""
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        # initialize with noise
        x = torch.empty(
            (self._num_steps + self._min_window, self._num_nodes,
             self._num_features)).normal_(generator=rng) * self.sigma_noise

        y_opt = torch.empty(
            (self._num_steps, self._num_nodes, self._num_features))

        if self.connectivity is None:
            edge_index = edge_weight = None
        else:
            edge_index, edge_weight = self.connectivity

        with torch.no_grad():
            h_t = None
            for t in tqdm(range(self._min_window,
                                self._min_window + self._num_steps),
                          desc=f"Generating {self.__class__.__name__} data"):
                x_t, h_t = self._model_forward(x[None, t - self._min_window:t],
                                               h=h_t,
                                               t=t,
                                               edge_index=edge_index,
                                               edge_weight=edge_weight)
                y_opt[t - self._min_window:t + 1 - self._min_window] = x_t[0]
                # add noise
                x_t = x_t + torch.zeros_like(x_t).normal_(
                    generator=rng) * self.sigma_noise
                x[t:t + 1] = x_t[0]

        x = torch_to_numpy(x[self._min_window:])
        y_opt = torch_to_numpy(y_opt)
        return x, y_opt, np.ones_like(x)

    def get_connectivity(self, layout: str = 'edge_index'):
        """"""
        if self.connectivity is None:
            return self.connectivity
        return parse_connectivity(connectivity=self.connectivity,
                                  target_layout=layout,
                                  num_nodes=self.n_nodes)
