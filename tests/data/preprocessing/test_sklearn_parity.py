"""Optional cross-checks of the tsl scalers against the reference
implementations in :mod:`sklearn.preprocessing`.

sklearn fits its scalers per-feature over the sample (row) axis, so the data here is 2D ``(samples,
features)`` and the tsl scalers use ``axis=0`` to match that convention. tsl
divides by ``scale + tsl.epsilon`` while sklearn divides by ``scale``, hence the
small absolute tolerance on transformed values.
"""
import numpy as np
import pytest

from tsl.data.preprocessing.scalers import (MinMaxScaler, RobustScaler,
                                            StandardScaler)

# skip the whole module if sklearn is unavailable -> keeps it optional
sk_pre = pytest.importorskip("sklearn.preprocessing")

ATOL = 1e-4


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((60, 4)).astype(np.float32)


def test_standard_scaler_matches_sklearn(data):
    tsl_sc = StandardScaler(axis=0).fit(data)
    sk_sc = sk_pre.StandardScaler().fit(data)
    np.testing.assert_allclose(tsl_sc.bias.ravel(), sk_sc.mean_, atol=ATOL)
    np.testing.assert_allclose(tsl_sc.scale.ravel(), sk_sc.scale_, atol=ATOL)
    np.testing.assert_allclose(tsl_sc.transform(data), sk_sc.transform(data),
                               atol=ATOL)


@pytest.mark.parametrize('out_range', [(0., 1.), (-1., 1.), (0., 5.)])
def test_minmax_scaler_matches_sklearn(data, out_range):
    tsl_sc = MinMaxScaler(axis=0, out_range=out_range).fit(data)
    sk_sc = sk_pre.MinMaxScaler(feature_range=out_range).fit(data)
    np.testing.assert_allclose(tsl_sc.transform(data), sk_sc.transform(data),
                               atol=ATOL)


def test_robust_scaler_matches_sklearn(data):
    tsl_sc = RobustScaler(axis=0).fit(data)
    sk_sc = sk_pre.RobustScaler().fit(data)
    np.testing.assert_allclose(tsl_sc.bias.ravel(), sk_sc.center_, atol=ATOL)
    np.testing.assert_allclose(tsl_sc.scale.ravel(), sk_sc.scale_, atol=ATOL)
    np.testing.assert_allclose(tsl_sc.transform(data), sk_sc.transform(data),
                               atol=ATOL)


def test_robust_scaler_custom_quantiles_matches_sklearn(data):
    q = (10., 90.)
    tsl_sc = RobustScaler(axis=0, quantile_range=q).fit(data)
    sk_sc = sk_pre.RobustScaler(quantile_range=q).fit(data)
    np.testing.assert_allclose(tsl_sc.scale.ravel(), sk_sc.scale_, atol=ATOL)
    np.testing.assert_allclose(tsl_sc.transform(data), sk_sc.transform(data),
                               atol=ATOL)


def test_robust_scaler_unit_variance_matches_sklearn(data):
    tsl_sc = RobustScaler(axis=0, unit_variance=True).fit(data)
    sk_sc = sk_pre.RobustScaler(unit_variance=True).fit(data)
    np.testing.assert_allclose(tsl_sc.scale.ravel(), sk_sc.scale_, atol=ATOL)
    np.testing.assert_allclose(tsl_sc.transform(data), sk_sc.transform(data),
                               atol=ATOL)
