Data Structures
===============

    This page is still under development

TSL is meant to deal with discrete-time spatiotemporal data, i.e., signals that
evolve over discrete points in time and space. Common input examples
are data coming from sensor networks. In principle, data of this kind can be
represented by 3-dimensional tensors, with:

#. The **Time** (:obj:`t`) dimension, accounting for the temporal evolution of the signal within a node (i.e., a sensor).
#. The **Node** (:obj:`n`) dimension, accounting for simultaneous observations measured at the different nodes in the network in a given time step.
#. The **Features** (:obj:`f`) or **Channels** dimension, allowing for multiple (heterogeneous) measurements at the same spatio-temporal point.

We call a **spatiotemporal graph** a tensor with finite :obj:`t`, :obj:`n`, and
:obj:`f` dimensions, paired with the underlying topology. In TSL, we use the
class :class:`tsl.data.Data` to represent and store the attributes of
a single spatiotemporal graph.

The ``Data`` object
-------------------

The :class:`tsl.data.Data` object contains attributes related to a single spatiotemporal graph.
This object extends :class:`torch_geometric.data.Data`, preserving all its functionalities and
adding utilities for spatiotemporal data processing. The main APIs of
:class:`~tsl.data.Data` include:

* :obj:`Data.input`: view on the tensors stored in :obj:`Data` that are meant to serve as input to the model.
  In the simplest case of a single node-attribute matrix, we could just have :obj:`Data.input.x`.
* :obj:`Data.target`: view on the tensors stored in :obj:`Data` used as labels to train the model.
  In the common case of a single label, we could just have :obj:`Data.input.y`.
* :obj:`Data.edge_index`: graph connectivity. Can be in COO format (i.e., a :class:`~torch.Tensor` of shape :obj:`[2, E]`)
  or in form of a :class:`torch_sparse.SparseTensor` with shape :obj:`[N, N]`. For dynamic graphs -- with time-varying topology --
  :obj:`edge_index` is a Python list of :class:`~torch.Tensor`.
* :obj:`Data.edge_weight`: weights of the graph connectivity, if :obj:`Data.edge_index` is not a :class:`torch_sparse.SparseTensor`.
  For dynamic graphs, :obj:`edge_weight` is a Python list of :class:`~torch.Tensor`.
* :obj:`Data.mask`: binary mask indicating the data in :obj:`Data.target.y` to be used
  as ground-truth for the loss (default is :obj:`None`).
* :obj:`Data.transform`: mapping of :class:`~tsl.data.preprocessing.scalers.ScalerModule`, whose keys must be
  transformable (or transformed) tensors in :obj:`Data`.
* :obj:`Data.pattern`: mapping containing the pattern for each tensor in :obj:`Data`.

None of these attributes are required and custom attributes can be seamlessly added.
:obj:`Data.input` and :obj:`Data.target` -- of type :class:`~tsl.data.StorageView` --
provide a view on the unique (shared) storage in :class:`~tsl.data.Data`, such that
the same key in :obj:`Data.input` and :obj:`Data.target` cannot reference different
objects.


The ``StaticBatch`` object
--------------------------

The :class:`tsl.data.StaticBatch` object models a temporal graph signal over a
static graph: while data change over time, the topology does not. This object
extends :class:`tsl.data.Data`, and has two additional methods for collating
(and separating) ``Data`` objects into ``StaticBatch`` objects.

The class method :meth:`~tsl.data.StaticBatch.from_data_list` creates a new
:class:`tsl.data.StaticBatch` object from a list of :class:`~tsl.data.Data`
objects. The implicit assumption is that all objects in the list **share the
same topology**, and only the graph in the first object is kept. Accordingly,
all the tensors in the ``Data`` objects having a static signal (i.e., without
temporal dimension) are not collated -- only one copy of them is kept. Instead,
all time-varying data are stacked along the first dimension, as usually done
in mini-batch collations. Also, :class:`~tsl.data.preprocessing.ScalerModule`
objects are collated or copied in a similar fashion. Consider also that the
changes made in the tensors are then reflected in the ``StaticBatch``
object's patterns.

The ``DisjointBatch`` object
----------------------------

    This page is still under development