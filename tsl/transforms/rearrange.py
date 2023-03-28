from torch_geometric.transforms import BaseTransform

from tsl.data import Data, StaticBatch


class Rearrange(BaseTransform):
    """Rearrange all keys in :class:`~tsl.data.Data` according to the provided
    pattern using `einops.rearrange <https://einops.rocks/api/rearrange/>`_.

    If the objects is of type :class:`~tsl.data.StaticBatch`, then the batch
    dimension in the output pattern is automatically considered."""

    def __init__(self, patterns: dict):
        self.item_patterns = patterns
        self.batch_patterns = dict()
        for key, pattern in patterns.items():
            if pattern.startswith("t"):
                self.batch_patterns[key] = "b " + pattern
            else:
                self.batch_patterns[key] = pattern

    def __call__(self, data: Data) -> Data:
        if isinstance(data, StaticBatch):
            data.rearrange(self.batch_patterns)
        else:
            data.rearrange(self.item_patterns)
        return data


class NodeThenTime(Rearrange):
    """Rearrange all keys in :class:`~tsl.data.Data` such that the node
    dimension precedes the temporal one.

    For time-variant but node-invariant signals, a new dummy dimension is
    added."""

    def __init__(self, original_patterns: dict):
        patterns = dict()
        for key, pattern in original_patterns.items():
            if pattern.startswith("t"):
                if "n" in pattern:
                    patterns[key] = "n t" + pattern[3:]
                else:
                    patterns[key] = "1 " + pattern
        super(NodeThenTime, self).__init__(patterns)
