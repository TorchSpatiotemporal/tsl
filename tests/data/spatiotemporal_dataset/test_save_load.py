""" save / load roundtrip."""
import numpy as np
import pytest
import torch

from tsl.data import SpatioTemporalDataset

from .helpers import _grid_target


def test_save_load_roundtrip(tmp_path):
    ds = SpatioTemporalDataset(target=_grid_target(40, 3, 2), window=6,
                               horizon=3, delay=1)
    ds.add_covariate('u', _grid_target(40, 3, 1), 't n f')
    path = tmp_path / 'dataset.pt'
    ds.save(str(path))

    loaded = SpatioTemporalDataset.load(str(path))
    assert isinstance(loaded, SpatioTemporalDataset)
    assert loaded.shape == ds.shape
    assert (loaded.window, loaded.horizon, loaded.delay) == (6, 3, 1)
    assert loaded.n_samples == ds.n_samples
    np.testing.assert_array_equal(loaded.numpy(), ds.numpy())
    np.testing.assert_array_equal(loaded.indices.numpy(), ds.indices.numpy())
    assert set(loaded.covariates) == {'u'}


def test_load_non_instance_raises(tmp_path):
    path = tmp_path / 'not_a_dataset.pt'
    torch.save(torch.zeros(3), str(path))
    with pytest.raises(TypeError):
        SpatioTemporalDataset.load(str(path))
