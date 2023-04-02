from torch_geometric.transforms import BaseTransform

from tsl.data import Data


class MaskInput(BaseTransform):
    """Whiten masked values in :attr:`input_key` according to mask in
    :attr:`mask_key`.

    Args:
        input_key (str): The key in ``Data`` to be masked.
            (default: :obj:`'input_key'`)
        mask_key (str): The key in ``Data`` to serve as mask.
            (default: :obj:`'mask_key'`)
    """

    def __init__(self, input_key: str = 'x', mask_key: str = 'mask'):
        self.input_key = input_key
        self.mask_key = mask_key

    def __call__(self, data: Data) -> Data:
        data[self.input_key] *= data[self.mask_key]
        return data
