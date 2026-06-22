import numpy as np

from tsl.data import SpatioTemporalDataset
from tsl.datasets.prototypes import TabularDataset

from .helpers import _grid_target


def test_from_dataset_copies_fields_and_filters_covariates():
    target = _grid_target(40, 3, 1)
    src = TabularDataset(target=target, name='src',
                         covariates={'u': _grid_target(40, 3, 1),
                                     'v': _grid_target(40, 3, 1)})
    ds = SpatioTemporalDataset.from_dataset(src, covariate_keys=['u'],
                                            window=6, horizon=3)
    # core fields are carried over
    assert ds.name == 'src'
    np.testing.assert_array_equal(ds.numpy(), target)
    # windowing kwargs are applied
    assert ds.window == 6 and ds.horizon == 3
    # only the requested covariate is kept
    assert set(ds.covariates) == {'u'}
