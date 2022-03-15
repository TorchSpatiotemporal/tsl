from typing import Optional, Union, Tuple, Mapping, List

import numpy as np
import torch
from torch import Tensor
from torch_geometric.utils import dense_to_sparse

from tsl.typing import TensArray
from . import utils


class DataParsingMixin:

    def _parse_data(self, obj: TensArray) -> Tensor:
        assert obj is not None
        obj = utils.to_steps_nodes_channels(obj)
        obj = utils.copy_to_tensor(obj)
        obj = utils.cast_tensor(obj, self.precision)
        return obj

    def _parse_mask(self, mask: Optional[TensArray]) -> Optional[Tensor]:
        if mask is None:
            return None
        mask = utils.to_steps_nodes_channels(mask)
        mask = utils.copy_to_tensor(mask)
        mask = utils.cast_tensor(mask)
        return mask

    def _parse_node_level_exogenous(self, obj: TensArray, name: str) -> Tensor:
        obj = utils.to_steps_nodes_channels(obj)
        obj = utils.copy_to_tensor(obj)
        obj = utils.cast_tensor(obj, self.precision)
        self._check_same_dim(obj.shape[0], 'n_steps', name)
        self._check_same_dim(obj.shape[1], 'n_nodes', name)
        return obj

    def _parse_graph_level_exogenous(self, obj: TensArray, name: str) -> Tensor:
        obj = utils.to_steps_channels(obj)
        obj = utils.copy_to_tensor(obj)
        obj = utils.cast_tensor(obj, self.precision)
        self._check_same_dim(obj.shape[0], 'n_steps', name)
        return obj

    def _parse_node_level_attribute(self, obj: TensArray, name: str) -> Tensor:
        obj = utils.to_nodes_channels(obj)
        obj = utils.copy_to_tensor(obj)
        obj = utils.cast_tensor(obj, self.precision)
        self._check_same_dim(obj.shape[0], 'n_nodes', name)
        return obj

    def _parse_graph_level_attribute(self, obj: TensArray) -> Tensor:
        obj = utils.copy_to_tensor(obj)
        obj = utils.cast_tensor(obj, self.precision)
        return obj

    def _parse_adj(self, connectivity: Union[TensArray, Tuple[TensArray]]
                   ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if connectivity is None:
            return None, None
        # if connectivity is (edge_index, edge_weight)
        if isinstance(connectivity, (list, tuple)):
            edge_index, edge_weight = connectivity
            edge_index = utils.copy_to_tensor(edge_index)
            if edge_weight is not None:
                edge_weight = utils.copy_to_tensor(edge_weight)
        elif isinstance(connectivity, (np.ndarray, Tensor)):
            connectivity = utils.copy_to_tensor(connectivity)
            assert connectivity.ndim == 2
            # if connectivity is edge_index
            if connectivity.size(0) == 2:
                edge_index, edge_weight = connectivity, None
            # if connectivity is dense_adj
            elif connectivity.size(0) == connectivity.size(1):
                self._check_same_dim(connectivity.size(0), 'n_nodes',
                                     'connectivity')
                edge_index, edge_weight = dense_to_sparse(connectivity)
            else:
                raise ValueError("`connectivity` must be a dense matrix or in "
                                 "COO format (i.e., an `edge_index`).")
        else:
            raise TypeError("`connectivity` must be a dense matrix or in "
                            "COO format (i.e., an `edge_index`).")
        if edge_weight is not None:
            edge_weight = utils.cast_tensor(edge_weight, self.precision)
        return edge_index, edge_weight

    def _check_same_dim(self, dim: int, attr: str, name: str):
        dim_data = getattr(self, attr)
        if dim != dim_data:
            raise ValueError("Cannot assign {0} with {1}={2}: data has {1}={3}"
                             .format(name, attr, dim, dim_data))

    def _check_name(self, name: str):
        if name.startswith('edge_'):
            raise ValueError(f"Cannot set attribute with name '{name}' in this "
                             f"way, consider adding edge attributes as "
                             f"{self.name}.{name} = value.")
        # name cannot be an attribute of self, nor a key in get
        invalid_names = set(dir(self)).union(self.keys)
        if name in invalid_names:
            raise ValueError(f"Cannot set attribute with name '{name}', there "
                             f"is already an attribute named '{name}' in the "
                             "dataset.")

    def _value_to_kwargs(self, value: Union[TensArray, List, Tuple, Mapping],
                         keys: Optional[Union[List, Tuple]] = None):
        if isinstance(value, TensArray.__args__):
            return dict(value=value)
        if isinstance(value, (list, tuple)):
            return dict(zip(keys, value))
        elif isinstance(value, Mapping):
            return value
        else:
            raise TypeError('Invalid type for value "{}"'.format(type(value)))

    def _exog_value_to_kwargs(self,
                              value: Union[TensArray, List, Tuple, Mapping]):
        keys = ['value', 'node_level', 'add_to_input_map', 'synch_mode',
                'preprocess']
        return self._value_to_kwargs(value, keys)

    def _attr_value_to_kwargs(self,
                              value: Union[TensArray, List, Tuple, Mapping]):
        keys = ['value', 'node_level', 'add_to_batch']
        return self._value_to_kwargs(value, keys)
