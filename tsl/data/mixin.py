from typing import Optional, Union, Tuple, Mapping, List

from torch import Tensor
from torch_geometric.data.storage import recursive_apply
from torch_geometric.typing import Adj
from torch_sparse import SparseTensor

from tsl.ops.connectivity import convert_torch_connectivity
from tsl.typing import DataArray, SparseTensArray, ScipySparseMatrix
from . import utils


class DataParsingMixin:

    def _parse_data(self, obj: DataArray) -> Tensor:
        assert obj is not None
        obj = utils.copy_to_tensor(obj)
        obj = utils.to_steps_nodes_channels(obj)
        obj = utils.cast_tensor(obj, self.precision)
        return obj

    def _parse_mask(self, mask: Optional[DataArray]) -> Optional[Tensor]:
        if mask is None:
            return None
        mask = utils.copy_to_tensor(mask)
        mask = utils.to_steps_nodes_channels(mask)
        self._check_same_dim(mask.size(0), 'n_steps', 'mask')
        self._check_same_dim(mask.size(1), 'n_nodes', 'mask')
        if mask.size(-1) > 1:
            self._check_same_dim(mask.size(-1), 'n_channels', 'mask')
        mask = utils.cast_tensor(mask)
        return mask

    def _parse_exogenous(self, obj: DataArray, name: str,
                         node_level: bool) -> Tensor:
        obj = utils.copy_to_tensor(obj)
        if node_level:
            obj = utils.to_steps_nodes_channels(obj)
            self._check_same_dim(obj.shape[1], 'n_nodes', name)
        else:
            obj = utils.to_steps_channels(obj)
        self._check_same_dim(obj.shape[0], 'n_steps', name)
        obj = utils.cast_tensor(obj, self.precision)
        return obj

    def _parse_attribute(self, obj: DataArray, name: str,
                         node_level: bool) -> Tensor:
        obj = utils.copy_to_tensor(obj)
        if node_level:
            obj = utils.to_nodes_channels(obj)
            self._check_same_dim(obj.shape[0], 'n_nodes', name)
        obj = utils.cast_tensor(obj, self.precision)
        return obj

    def _parse_adj(self, connectivity: Union[SparseTensArray, Tuple[DataArray]],
                   target_layout: Optional[str] = None
                   ) -> Tuple[Optional[Adj], Optional[Tensor]]:
        # format in [sparse, edge_index, None], where None means keep as input
        if connectivity is None:
            return None, None

        # Convert to torch
        # from np.ndarray, pd.DataFrame or torch.Tensor
        if isinstance(connectivity, DataArray.__args__):
            connectivity = utils.copy_to_tensor(connectivity)
        elif isinstance(connectivity, (list, tuple)):
            connectivity = recursive_apply(connectivity, utils.copy_to_tensor)
        # from scipy sparse matrix
        elif isinstance(connectivity, ScipySparseMatrix):
            connectivity = SparseTensor.from_scipy(connectivity)
        elif not isinstance(connectivity, SparseTensor):
            raise TypeError("`connectivity` must be a dense matrix or in "
                            "COO format (i.e., an `edge_index`).")

        if target_layout is not None:
            connectivity = convert_torch_connectivity(connectivity,
                                                      target_layout,
                                                      num_nodes=self.n_nodes)

        if isinstance(connectivity, (list, tuple)):
            edge_index, edge_weight = connectivity
            if edge_weight is not None:
                edge_weight = utils.cast_tensor(edge_weight, self.precision)
        else:
            edge_index, edge_weight = connectivity, None
            self._check_same_dim(edge_index.size(0), 'n_nodes', 'connectivity')

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

    def _value_to_kwargs(self, value: Union[DataArray, List, Tuple, Mapping],
                         keys: Optional[Union[List, Tuple]] = None):
        if isinstance(value, DataArray.__args__):
            return dict(value=value)
        if isinstance(value, (list, tuple)):
            return dict(zip(keys, value))
        elif isinstance(value, Mapping):
            return value
        else:
            raise TypeError('Invalid type for value "{}"'.format(type(value)))

    def _exog_value_to_kwargs(self,
                              value: Union[DataArray, List, Tuple, Mapping]):
        keys = ['value', 'node_level', 'add_to_input_map', 'synch_mode',
                'preprocess']
        return self._value_to_kwargs(value, keys)

    def _attr_value_to_kwargs(self,
                              value: Union[DataArray, List, Tuple, Mapping]):
        keys = ['value', 'node_level', 'add_to_batch']
        return self._value_to_kwargs(value, keys)
