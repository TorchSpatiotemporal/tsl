import numpy as np
import torch
from typing import Any, Union, List, Mapping


def numpy(tensors: Union[List[torch.Tensor], Mapping[Any, torch.Tensor], torch.Tensor]) \
        -> Union[List[np.ndarray], Mapping[Any, np.ndarray], np.ndarray]:
    """
    Cast tensors to numpy arrays.

    Args:'
        tensors: A tensor or a list or dictinary containing tensors.

    Returns:
        Tensors casted to numpy arrays.
    """
    if isinstance(tensors, list):
        return [t.detach().cpu().numpy() for t in tensors]
    if isinstance(tensors, dict):
        for k, v in tensors.items():
            tensors[k] = v.detach().cpu().numpy()
        return tensors
    if isinstance(tensors, torch.Tensor):
        return tensors.detach().cpu().numpy()
    raise ValueError
