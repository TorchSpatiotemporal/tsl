"""Group 13 -- transform callable applied on __getitem__."""
from tsl.data import SpatioTemporalDataset

from .helpers import _grid_target


def test_transform_applied_on_getitem():
    seen = []

    def transform(item):
        seen.append(1)
        item.tagged = True
        return item

    ds = SpatioTemporalDataset(target=_grid_target(40, 2, 1), window=4,
                               horizon=2, transform=transform)
    item = ds[0]
    # the callable ran on access and its mutation is visible on the item
    assert getattr(item, 'tagged', False)
    assert len(seen) == 1
