import numpy as np

from tsl.data import SpatioTemporalDataset, TemporalSplitter


def test_temporal_splitter():
    splitter = TemporalSplitter(val_len=0.1, test_len=0.2, offset='window')
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
    # With the default offset='window', the closest train/val samples are
    # samples_offset = ceil(window / stride) = 3 positions apart, so the slice
    # drops samples_offset - 1 = 2 samples between splits.
    gap = window - 1
    # check that the split is correct
    assert len(dataset) == n_samples
    # val_len = int(0.1 * (95 - 19)) = 7, test_len = int(0.2 * 95) = 19
    # test_start = 76, val_start = 69
    assert len(idxs['train']) == 69 - gap
    assert len(idxs['val']) == 7 - gap
    assert len(idxs['test']) == 19
    # repeat using integer val/test len
    splitter = TemporalSplitter(val_len=8, test_len=20)
    idxs = splitter.split(dataset)
    # test_start = 75, val_start = 67
    assert len(idxs['train']) == 67 - gap
    assert len(idxs['val']) == 8 - gap
    assert len(idxs['test']) == 20
