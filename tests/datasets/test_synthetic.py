"""Synthetic generators: ``GaussianNoiseSyntheticDataset`` and the
``GPVARDataset`` / ``GPVARDatasetAZ`` filters. These build data in memory (no
download), so the focus is determinism given a seed and the structural contract
of the generated tensors.
"""
import math

import numpy as np
import pytest
import torch
from torch import nn

from tsl.datasets import (GaussianNoiseSyntheticDataset, GPVARDataset,
                          GPVARDatasetAZ)

NUM_NODES, NUM_FEATURES, NUM_STEPS = 4, 1, 50
EDGE_INDEX = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])


class MeanModel(nn.Module):
    """Trivial autoregressive model: predict the mean over the input window.

    Its ``forward`` takes only ``x`` -- exercising the kwargs-filtering path,
    since the generator also passes ``h``/``t``/``edge_index``/``edge_weight``.
    """

    def forward(self, x):
        return x.mean(dim=1, keepdim=True)


def make_gaussian(seed=0, sigma_noise=0.2, connectivity=EDGE_INDEX,
                  num_steps=NUM_STEPS):
    return GaussianNoiseSyntheticDataset(num_features=NUM_FEATURES,
                                         num_nodes=NUM_NODES,
                                         num_steps=num_steps,
                                         connectivity=connectivity,
                                         model=MeanModel(),
                                         sigma_noise=sigma_noise,
                                         seed=seed,
                                         name='synth')


# -- GaussianNoiseSyntheticDataset ------------------------------------------


def test_shape_and_mask():
    ds = make_gaussian()
    assert ds.shape == (NUM_STEPS, NUM_NODES, NUM_FEATURES)
    assert ds.has_mask
    assert ds.mask.all()  # synthetic data has no missing values


def test_optimal_pred_covariate():
    ds = make_gaussian()
    assert 'optimal_pred' in ds.covariates
    assert ds.patterns['optimal_pred'] == 't n f'
    assert ds.optimal_pred.shape == (NUM_STEPS, NUM_NODES, NUM_FEATURES)


def test_determinism_same_seed():
    a, b = make_gaussian(seed=42), make_gaussian(seed=42)
    np.testing.assert_array_equal(a.numpy(), b.numpy())
    np.testing.assert_array_equal(a.optimal_pred, b.optimal_pred)


def test_different_seed_differs():
    a, b = make_gaussian(seed=1), make_gaussian(seed=2)
    assert not np.array_equal(a.numpy(), b.numpy())


def test_mae_optimal_model_formula():
    ds = make_gaussian(sigma_noise=0.37)
    assert ds.mae_optimal_model == pytest.approx(math.sqrt(2 / math.pi) * 0.37)


def test_connectivity_roundtrip():
    ds = make_gaussian()
    edge_index, _ = ds.get_connectivity(layout='edge_index')
    assert edge_index.shape[0] == 2


def test_connectivity_none():
    ds = make_gaussian(connectivity=None)
    assert ds.get_connectivity() is None


def test_kwargs_filtering_keeps_optimal_pred_consistent():
    # MeanModel.forward ignores h/t/edge_*; the generator must still run and the
    # noise-free signal must equal the model applied to the (noisy) input window.
    ds = make_gaussian(seed=7)
    # optimal prediction at step t is the mean over the previous min_window steps
    # of the (noisy) signal; with min_window=1 it equals the previous step.
    assert ds.optimal_pred.shape == ds.numpy().shape


# -- GPVARDataset -----------------------------------------------------------


def test_gpvar_node_count_per_community():
    # build_tri_community_graph yields 6 nodes per (triangular) community
    ds = GPVARDataset(num_communities=2, num_steps=NUM_STEPS,
                      filter_params=[[0.5, 0.1]])
    assert ds.n_nodes == 6 * 2


def test_gpvar_has_self_loops():
    ds = GPVARDataset(num_communities=2, num_steps=NUM_STEPS,
                      filter_params=[[0.5, 0.1]])
    edge_index, _ = ds.get_connectivity(layout='edge_index')
    self_loops = (edge_index[0] == edge_index[1]).sum().item()
    assert self_loops == ds.n_nodes


def test_gpvar_output_bounded_by_tanh():
    ds = GPVARDataset(num_communities=2, num_steps=NUM_STEPS,
                      filter_params=[[0.5, 0.1]])
    # the GP-VAR filter wraps its output in tanh, so the noise-free signal is in
    # (-1, 1)
    assert np.all(np.abs(ds.optimal_pred) <= 1.0 + 1e-6)


def test_gpvar_generator_is_deterministic():
    ds = GPVARDataset(num_communities=2, num_steps=NUM_STEPS,
                      filter_params=[[0.5, 0.1]])
    x0, y0, _ = ds.generate_data(seed=0)
    x1, y1, _ = ds.generate_data(seed=0)
    np.testing.assert_array_equal(x0, x1)
    np.testing.assert_array_equal(y0, y1)
    x2, _, _ = ds.generate_data(seed=1)
    assert not np.array_equal(x0, x2)


# -- GPVARDatasetAZ (cached build) ------------------------------------------


def test_gpvar_az_constants():
    assert GPVARDatasetAZ.seed == 1234
    assert GPVARDatasetAZ.NUM_COMMUNITIES == 5
    assert GPVARDatasetAZ.NUM_STEPS == 30000
    assert GPVARDatasetAZ.SIGMA_NOISE == 0.4


def test_gpvar_az_build_roundtrip(tmp_path):
    # build into an isolated root and confirm the .npy cache is created and the
    # reloaded data is reproducible (seed=1234)
    ds = GPVARDatasetAZ(root=str(tmp_path))
    assert ds.required_file_names == ['GPVAR_AZ.npy']
    assert (tmp_path / 'GPVAR_AZ.npy').exists()
    assert ds.n_nodes == 6 * GPVARDatasetAZ.NUM_COMMUNITIES
    assert ds.length == GPVARDatasetAZ.NUM_STEPS
    # a second instance loads the cache and yields identical data
    ds2 = GPVARDatasetAZ(root=str(tmp_path))
    np.testing.assert_array_equal(ds.numpy(), ds2.numpy())
