import numpy as np

from tsl.data import SpatioTemporalDataset, TemporalSplitter


def test_temporal_splitter():
    splitter = TemporalSplitter(val_len=0.1, test_len=0.2)
    # create a dummy sequence
    seq = np.arange(100)
    # create dummy SpatiotemporalDataset
    window = 3
    horizon = 3

    dataset = SpatioTemporalDataset(
        target=seq,
        window=window,
        horizon=horizon,
    )
    n_samples = 100 - window - horizon + 1
    idxs = splitter.split(dataset)
    # check that the split is correct
    assert len(dataset) == n_samples
    assert len(idxs['train']) == 69 - 3
    assert len(idxs['val']) == 7 - 3
    assert len(idxs['test']) == 19
    # repeat using integer val/test len
    splitter = TemporalSplitter(val_len=8, test_len=20)
    idxs = splitter.split(dataset)
    assert len(idxs['train']) == 67 - 3
    assert len(idxs['val']) == 8 - 3
    assert len(idxs['test']) == 20
