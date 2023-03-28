from typing import List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from torch import Tensor
from torch_geometric.typing import Adj

from tsl.ops.connectivity import parse_connectivity
from tsl.typing import DataArray, SparseTensArray

from ..ops.pattern import check_pattern, infer_pattern
from ..utils import casting


class DataParsingMixin:

    def _parse_target(self, obj: DataArray) -> Tensor:
        obj = casting.copy_to_tensor(obj)
        obj = casting.to_time_nodes_channels(obj)
        obj = casting.convert_precision_tensor(obj, precision=self.precision)
        return obj

    def _parse_covariate(self,
                         obj: DataArray,
                         pattern: Optional[str] = None,
                         allow_broadcasting: bool = False,
                         convert_precision: bool = True,
                         name: str = 'covariate') -> Tuple[Tensor, str]:
        # convert to tensor
        obj = casting.copy_to_tensor(obj)

        # infer pattern if it is None, otherwise sanity check
        if pattern is None:
            pattern = infer_pattern(obj.shape,
                                    t=self.n_steps,
                                    n=self.n_nodes,
                                    e=self.n_edges)
        else:
            pattern = check_pattern(pattern, ndim=obj.ndim)

        # check that pattern and shape match
        self._check_pattern(obj,
                            pattern,
                            name,
                            allow_broadcasting=allow_broadcasting)

        if convert_precision:
            obj = casting.convert_precision_tensor(obj, self.precision)

        return obj, pattern

    def _parse_connectivity(
        self,
        connectivity: Union[SparseTensArray, Tuple[DataArray]],
        target_layout: Optional[str] = None
    ) -> Tuple[Optional[Adj], Optional[Tensor]]:
        # target_layout in [dense, sparse, edge_index, None]
        # where None means keep as input
        if connectivity is None:
            return None, None

        connectivity = parse_connectivity(connectivity=connectivity,
                                          target_layout=target_layout,
                                          num_nodes=self.n_nodes)

        if isinstance(connectivity, (list, tuple)):
            edge_index, edge_weight = connectivity
            if edge_weight is not None:
                edge_weight = casting.convert_precision_tensor(
                    edge_weight, self.precision)
        else:
            edge_index, edge_weight = connectivity, None
            self._check_same_dim(edge_index.size(0), 'n_nodes', 'connectivity')

        return edge_index, edge_weight

    def _check_pattern(self,
                       obj: Tensor,
                       pattern: str,
                       name: str,
                       allow_broadcasting: bool = False):
        dims = pattern.strip().split(' ')
        for token, size in zip(dims, obj.size()):
            if token == 't':
                self._check_same_dim(size, 'n_steps', name, allow_broadcasting)
            elif token == 'n':
                self._check_same_dim(size, 'n_nodes', name, allow_broadcasting)
            elif token == 'e':
                assert self.edge_index is not None
                self._check_same_dim(size, 'n_edges', name, allow_broadcasting)

    def _check_same_dim(self,
                        dim: int,
                        attr: str,
                        name: str,
                        allow_broadcasting: bool = False):
        dim_data = getattr(self, attr)
        if not (dim == dim_data or (dim == 1 and allow_broadcasting)):
            raise ValueError(
                "Cannot assign {0} with {1}={2}: data has {1}={3}".format(
                    name, attr, dim, dim_data))

    def _check_name(self, name: str):
        # name cannot be an attribute of self, nor a key in get
        invalid_names = set(dir(self))
        if name in invalid_names:
            raise ValueError(f"Cannot set attribute with name '{name}', there "
                             f"is already an attribute named '{name}' in the "
                             "dataset.")

    def _value_to_kwargs(self, value: Union[DataArray, List, Tuple, Mapping]):
        keys = [
            'value', 'pattern', 'add_to_input_map', 'synch_mode', 'preprocess',
            'convert_precision'
        ]
        if isinstance(value, (pd.DataFrame, np.ndarray, Tensor)):
            return dict(value=value)
        if isinstance(value, (list, tuple)):
            return dict(zip(keys, value))
        elif isinstance(value, Mapping):
            assert set(value.keys()).issubset(keys)
            return value
        else:
            raise TypeError('Invalid type for value "{}"'.format(type(value)))
