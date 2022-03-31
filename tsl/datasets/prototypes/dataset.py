import functools
import os
from typing import (
    Union,
    Optional,
    Iterable,
    List,
    Tuple,
    Set,
    Sequence
)

import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from torch import TensorType

import tsl
from tsl import logger, config
from ...data.datamodule import splitters, Splitter
from ...ops.similarities import top_k
from tsl.ops.connectivity import adj_to_edge_index
from ...typing import ScipySparseMatrix
from ...utils import preprocessing
from ...utils.io import save_pickle, load_pickle
from ...utils.python_utils import ensure_list, files_exist, hash_dict


class Dataset(object):
    """Base class for Datasets in tsl.


    Args:
        name (str, optional): Name of the dataset. If :obj:`None`, use name of
            the class. (default: :obj:`None`)
        spatial_aggregation (str): Function (as string) used for aggregation
            along temporal dimension. (default: :obj:`'sum'`)
        spatial_aggregation (str): Permutation invariant function (as string)
            used for aggregation along nodes' dimension. (default: :obj:`'sum'`)
    """
    root: Optional[str] = None

    similarity_options: Optional[Set] = None
    temporal_aggregation_options: Optional[Set] = None
    spatial_aggregation_options: Optional[Set] = None

    def __init__(self, name: Optional[str] = None,
                 similarity_score: Optional[str] = None,
                 temporal_aggregation: str = 'sum',
                 spatial_aggregation: str = 'sum',
                 default_splitting_method: str = 'temporal'):
        # Set name
        self.name = name if name is not None else self.__class__.__name__
        # Set similarity method
        if self.similarity_options is not None:
            if similarity_score not in self.similarity_options:
                raise ValueError("{} is not a valid similarity method."
                                 .format(similarity_score))
        self.similarity_score = similarity_score
        # Set temporal aggregation method
        if self.temporal_aggregation_options is not None:
            if temporal_aggregation not in self.temporal_aggregation_options:
                raise ValueError("{} is not a valid temporal aggregation "
                                 "method.".format(temporal_aggregation))
        self.temporal_aggregation = temporal_aggregation
        # Set spatial aggregation method
        if self.spatial_aggregation_options is not None:
            if spatial_aggregation not in self.spatial_aggregation_options:
                raise ValueError("{} is not a valid spatial aggregation "
                                 "method.".format(spatial_aggregation))
        self.spatial_aggregation = spatial_aggregation
        # Set splitting method
        self.default_splitting_method = default_splitting_method

    def __new__(cls, *args, **kwargs) -> "Dataset":
        obj = super().__new__(cls)

        # decorate `get_splitter`
        obj.get_splitter = cls._wrap_method(obj, obj.get_splitter)

        return obj

    @staticmethod
    def _wrap_method(obj: "Dataset", fn: callable) -> callable:
        """A decorator that extends functionalities of some methods.

        - When ``ds.get_splitter(...)`` is called, if no method is specified or
        if the method is not dataset-specific (specified by overriding the
        method), the method is looked-up among the ones provided by the library.
        Notice that this happens whether or not this method is overridden.

        Args:
            obj: Object whose function will be tracked.
            fn: Function that will be wrapped.

        Returns:
            Decorated method to extend functionalities.
        """

        @functools.wraps(fn)
        def get_splitter(method: Optional[str] = None, *args, **kwargs) \
                -> Splitter:
            if method is None:
                method = obj.default_splitting_method
            splitter = fn(method, *args, **kwargs)
            if splitter is None:
                try:
                    splitter = getattr(splitters, method)(*args, **kwargs)
                except AttributeError:
                    raise NotImplementedError(f'Splitter option "{method}" '
                                              f'does not exists.')
            return splitter

        if fn.__name__ == 'get_splitter':
            return get_splitter

    def __getstate__(self) -> dict:
        # avoids _pickle.PicklingError: Can't pickle <...>: it's not the same
        # object as <...>
        d = self.__dict__.copy()
        del d['get_splitter']
        return d

    # Data dimensions

    @property
    def length(self) -> int:
        """Returns the length -- in terms of time steps -- of the dataset.

        Returns:
            int: Temporal length of the dataset.
        """
        raise NotImplementedError

    @property
    def n_nodes(self) -> int:
        """Returns the number of nodes in the dataset. In case of dynamic graph,
        :obj:`n_nodes` is the total number of nodes present in at least one
        time step.

        Returns:
            int: Total number of nodes in the dataset.
        """
        raise NotImplementedError

    @property
    def n_channels(self) -> int:
        """Returns the number of node-level channels of the main signal in the
        dataset.

        Returns:
            int: Number of channels of the main signal.
        """
        raise NotImplementedError

    #

    def __repr__(self):
        return "{}(length={}, n_nodes={}, n_channels={})" \
            .format(self.name, self.length, self.n_nodes, self.n_channels)

    def __len__(self):
        """Returns the length -- in terms of time steps -- of the dataset.

        Returns:
            int: Temporal length of the dataset.
        """
        return self.length

    # Directory information

    @property
    def root_dir(self) -> str:
        if isinstance(self.root, str):
            root = os.path.expanduser(os.path.normpath(self.root))
        elif self.root is None:
            root = os.path.join(config.data_dir, self.__class__.__name__)
        else:
            raise ValueError
        return root

    @property
    def raw_file_names(self) -> Union[str, Sequence[str]]:
        """The name of the files in the :obj:`self.root_dir` folder that must be
        present in order to skip downloading."""
        return []

    @property
    def required_file_names(self) -> Union[str, Sequence[str]]:
        """The name of the files in the :obj:`self.root_dir` folder that must be
        present in order to skip building."""
        return []

    @property
    def raw_files_paths(self) -> List[str]:
        """The absolute filepaths that must be present in order to skip
        downloading."""
        files = ensure_list(self.raw_file_names)
        return [os.path.join(self.root_dir, f) for f in files]

    @property
    def required_files_paths(self) -> List[str]:
        """The absolute filepaths that must be present in order to skip
        building."""
        files = ensure_list(self.required_file_names)
        return [os.path.join(self.root_dir, f) for f in files]

    # Loading pipeline: load() → load_raw() → build() → download()

    def maybe_download(self):
        if not files_exist(self.raw_files_paths):
            os.makedirs(self.root_dir, exist_ok=True)
            self.download()

    def maybe_build(self):
        if not files_exist(self.required_files_paths):
            os.makedirs(self.root_dir, exist_ok=True)
            self.build()

    def download(self) -> None:
        """Downloads dataset's files to the :obj:`self.root_dir` folder."""
        raise NotImplementedError

    def build(self) -> None:
        """Eventually build the dataset from raw data to :obj:`self.root_dir`
        folder."""
        pass

    def load_raw(self, *args, **kwargs):
        """Loads raw dataset without any data preprocessing."""
        raise NotImplementedError

    def load(self, *args, **kwargs):
        """Loads raw dataset and preprocess data."""
        raise NotImplementedError

    def clean_downloads(self):
        for file in self.raw_files_paths:
            if file not in self.required_files_paths:
                os.unlink(file)

    def clean_root_dir(self):
        import shutil
        for filename in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, filename)
            if file_path in self.required_files_paths + self.raw_files_paths:
                continue
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Representations

    def dataframe(self) -> Union[DataFrame, List[DataFrame]]:
        """Returns a pandas representation of the dataset in the form of a
        :class:`~pandas.DataFrame`. May be a list of DataFrames if the dataset
        has a dynamic structure."""
        raise NotImplementedError

    def numpy(self, return_idx: bool = False) -> \
            Union[ndarray, List[ndarray],
                  Tuple[ndarray, Series], Tuple[List[ndarray], Series]]:
        """Returns a numpy representation of the dataset in the form of a
        :class:`~numpy.ndarray`. If :obj:`return_index` is :obj:`True`, it
        returns also a :class:`~pandas.Series` that can be used as index. May
        be a list of ndarrays (and Series) if the dataset has a dynamic
        structure."""
        raise NotImplementedError

    def pytorch(self) -> Union[TensorType, List[TensorType]]:
        """Returns a pytorch representation of the dataset in the form of a
        :class:`~torch.Tensor`. May be a list of Tensors if the dataset has a
        dynamic structure."""
        raise NotImplementedError

    # IO

    def save_pickle(self, filename: str) -> None:
        """Save :obj:`Dataset` to disk.

        Args:
            filename (str): path to filename for storage.
        """
        save_pickle(self, filename)

    @classmethod
    def load_pickle(cls, filename: str) -> "Dataset":
        """Load instance of :obj:`Dataset` from disk.

        Args:
            filename (str): path of :obj:`Dataset`.
        """
        obj = load_pickle(filename)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded file is not of class {cls}.")
        return obj

    # Similarity pipeline: get_adj() → get_similarity() → compute_similarity()

    def compute_similarity(self, method: str, **kwargs) -> Optional[np.ndarray]:
        """Implements the options for the similarity matrix :math:`\mathbf{S}
        \in \mathbb{R}^{N \\times N}` computation, according to :obj:`method`.

        Args:
            method (str): Method for the similarity computation.
            **kwargs (optional): Additional optional keyword arguments.

        Returns:
            ndarray: The similarity dense matrix.
        """
        raise NotImplementedError

    def get_similarity(self, method: Optional[str] = None,
                       save: bool = False,
                       **kwargs) -> ndarray:
        """Returns the matrix :math:`\mathbf{S} \in \mathbb{R}^{N \\times N}`,
        where :math:`N=`:obj:`self.n_nodes`, with the pairwise similarity
        scores between nodes.

        Args:
            method (str, optional): Method for the similarity computation. If
                :obj:`None`, defaults to dataset-specific default method.
                (default: :obj:`None`)
            save (bool): Whether to save similarity matrix in dataset's
                directory after computation.
                (default: :obj:`True`)
            **kwargs (optional): Additional optional keyword arguments.

        Returns:
            ndarray: The similarity dense matrix.

        Raises:
            ValueError: If the similarity method is not valid.
        """
        if method is None:
            method = self.similarity_score
        if method not in self.similarity_options:
            raise ValueError("Similarity method '{}' not valid".format(method))
        if save:
            enc = hash_dict(dict(method=method,
                                 class_name=self.__class__.__name__,
                                 name=self.name,
                                 **kwargs))
            name = "sim_{}.npy".format(enc)
            path = os.path.join(self.root_dir, name)
            if os.path.exists(path):
                logger.warning("Loading cached similarity matrix.")
                return np.load(path)
        # get similarity method
        sim = self.compute_similarity(method, **kwargs)
        if save:
            np.save(path, sim)
            logger.info(f"Similarity matrix saved at {path}.")
        return sim

    def get_connectivity(self, method: Optional[str] = None,
                         threshold: Optional[float] = None,
                         knn: Optional[int] = None, include_self: bool = True,
                         force_symmetric: bool = False,
                         normalize_axis: Optional[int] = None,
                         layout: str = 'edge_index',
                         **kwargs) -> Union[ndarray, Tuple, ScipySparseMatrix]:
        r"""Returns the weighted adjacency matrix :math:`\mathbf{W} \in
        \mathbb{R}^{N \\times N}`, where :math:`N=`:obj:`self.n_nodes`. The
        element :math:`w_{i,j} \in \mathbf{W}` is 0 if there not exists an edge
        connecting node :math:`i` to node :math:`j`. If `sparse`, returns edge
        index :math:`\mathcal{E}` and edge weights :math:`\mathbf{w} \in
        \mathbb{R}^{|\mathcal{E}|}` (default: :obj:`True`).

        Args:
            method (str, optional): Method for the similarity computation. If
                :obj:`None`, defaults to dataset-specific default method.
                (default: :obj:`None`)
            threshold (float, optional): If not :obj:`None`, set to 0 the values
                below the threshold. (default: :obj:`None`)
            knn (int, optional): If not :obj:`None`, keep only :math:`k=`
                :obj:`knn` nearest incoming neighbors.
                (default: :obj:`None`)
            include_self (bool): If :obj:`False`, self-loops are never taken
                into account. (default: :obj:`False`)
            force_symmetric (bool): Force adjacency matrix to be symmetric by
                taking the maximum value between the two directions for each
                edge. (default: :obj:`False`)
            normalize_axis (int, optional): Divide edge weight :math:`w_{i, j}`
                by :math:`\sum_k w_{i, k}`, if :obj:`normalize_axis=0` or
                :math:`\sum_k w_{k, j}`, if :obj:`normalize_axis=1`. :obj:`None`
                for no normalization.
                (default: :obj:`None`)
            layout (str): Convert matrix to a dense/sparse format. Available
                options are:
                  - dense: keep matrix dense
                  - edge_index: convert to (edge_index, edge_weight) tuple
                  - coo, csr, csc: convert to specified scipy sparse matrix
                (default: 'dense')
            **kwargs (optional): Additional optional keyword arguments for
                similarity computation.

        Returns:
            The similarity dense matrix.
        """
        if 'sparse' in kwargs:
            import warnings
            warnings.warn("The argument 'sparse' is deprecated and will be "
                          "removed in future version of tsl. Please use "
                          "the argument `layout` instead.")
            layout = 'edge_index' if kwargs['sparse'] else 'dense'
        if method == 'full':
            adj = np.ones((self.n_nodes, self.n_nodes))
        elif method == 'identity':
            adj = np.eye(self.n_nodes)
        else:
            adj = self.get_similarity(method, **kwargs)
        if threshold is not None:
            adj[adj < threshold] = 0
        if knn is not None:
            adj = top_k(adj, knn, include_self=include_self)
        if not include_self:
            np.fill_diagonal(adj, 0)
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if normalize_axis:
            adj = adj / (adj.sum(normalize_axis, keepdims=True) + tsl.epsilon)
        if layout == 'dense':
            return adj
        elif layout == 'edge_index':
            return adj_to_edge_index(adj)
        elif layout == 'coo':
            return coo_matrix(adj)
        elif layout == 'csr':
            return csr_matrix(adj)
        elif layout == 'csc':
            return csc_matrix(adj)
        else:
            raise ValueError(f"Invalid format for connectivity: {layout}. Valid"
                             " options are [dense, edge_index, coo, csr, csc].")

    # Cross-validation splitting options

    def get_splitter(self, method: Optional[str] = None,
                     *args, **kwargs) -> Splitter:
        """Returns the splitter for a :class:`~tsl.data.SpatioTemporalDataset`.
        A :class:`~tsl.data.preprocessing.Splitter` provides the splits of the
        dataset -- in terms of indices -- for cross validation."""

    # Data methods

    def aggregate(self, node_index: Optional[Iterable[Iterable]] = None):
        """Aggregates nodes given an index of cluster assignments (spatial
        aggregation).

        Args:
            node_index: Sequence of grouped node ids.
        """
        return preprocessing.aggregate(self.dataframe(),
                                       node_index,
                                       self.spatial_aggregation)

    # Getters for SpatioTemporalDataset

    def get_config(self) -> dict:
        """Returns the keywords arguments (as dict) for instantiating a
         :class:`~tsl.data.SpatioTemporalDataset`."""
        raise NotImplementedError
