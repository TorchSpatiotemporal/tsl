import numpy as np

from tsl.data.datamodule.splitters import CustomSplitter

from .helpers import _make_dt_dataset


def _tail_split(dataset, length, mask=None):
    """Minimal sequential tail split fn for :class:`CustomSplitter`: the last
    ``length`` fraction of the (un-masked) samples becomes the held-out set."""
    idx = np.arange(len(dataset))
    if mask is not None:
        idx = np.setdiff1d(idx, mask)
    cut = len(idx) - max(int(length * len(idx)), 1)
    return idx[:cut], idx[cut:]


def test_custom_splitter_covers_and_disjoint():
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)
    splitter = CustomSplitter(val_split_fn=_tail_split,
                              test_split_fn=_tail_split,
                              val_kwargs=dict(length=0.1),
                              test_kwargs=dict(length=0.2))
    out = splitter.split(dataset)
    tr, va, te = out['train'], out['val'], out['test']
    # the three splits partition all samples...
    assert len(tr) + len(va) + len(te) == len(dataset)
    assert set(tr) | set(va) | set(te) == set(range(len(dataset)))
    # ...and are pairwise disjoint
    assert np.intersect1d(tr, va).size == 0
    assert np.intersect1d(va, te).size == 0
    assert np.intersect1d(tr, te).size == 0


def test_custom_splitter_policies_report_fn_names():
    splitter = CustomSplitter(val_split_fn=_tail_split,
                              test_split_fn=_tail_split,
                              val_kwargs=dict(length=0.1),
                              test_kwargs=dict(length=0.2))
    assert splitter.val_policy == '_tail_split'
    assert splitter.test_policy == '_tail_split'


def test_custom_splitter_policies_none_when_not_callable():
    splitter = CustomSplitter(val_split_fn=None, test_split_fn=None)
    assert splitter.val_policy is None
    assert splitter.test_policy is None


def test_mask_test_indices_in_val_keeps_val_test_disjoint():
    # The test tail is masked out before the val fn runs, so val draws from the
    # remaining samples and stays disjoint from test.
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)
    splitter = CustomSplitter(val_split_fn=_tail_split,
                              test_split_fn=_tail_split,
                              val_kwargs=dict(length=0.2),
                              test_kwargs=dict(length=0.2),
                              mask_test_indices_in_val=True)
    out = splitter.split(dataset)
    assert np.intersect1d(out['val'], out['test']).size == 0


def test_custom_splitter_empty_test_set():
    # When the test fn yields no indices, the ``len(test_idxs)`` guard skips
    # masking and the val fn runs over all samples.
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)

    def empty_test(ds, **kwargs):
        return np.arange(len(ds)), np.array([], dtype=int)

    splitter = CustomSplitter(val_split_fn=_tail_split,
                              test_split_fn=empty_test,
                              val_kwargs=dict(length=0.1),
                              mask_test_indices_in_val=True)
    out = splitter.split(dataset)
    assert len(out['test']) == 0
    assert len(out['val']) and len(out['train'])
    assert len(out['train']) + len(out['val']) == len(dataset)


def test_no_mask_passes_no_mask_kwarg_to_val_fn():
    # With masking disabled the val fn is called without a ``mask`` kwarg; using
    # a fn that rejects ``mask`` proves the kwarg is not forwarded.
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)

    def val_fn(ds, length):  # no ``mask`` parameter
        idx = np.arange(len(ds))
        cut = len(idx) - int(length * len(idx))
        return idx[:cut], idx[cut:]

    splitter = CustomSplitter(val_split_fn=val_fn,
                              test_split_fn=_tail_split,
                              val_kwargs=dict(length=0.1),
                              test_kwargs=dict(length=0.2),
                              mask_test_indices_in_val=False)
    out = splitter.split(dataset)
    assert len(out['train']) and len(out['val']) and len(out['test'])
