Data Structures
===============

:tsl:`tsl` is meant to deal with discrete-time spatiotemporal data, i.e., signals that
evolve over discrete points in time and space. Common input examples
are data coming from sensor networks. In principle, data of this kind can be
represented by 3-dimensional tensors, with:

#. The **Time** (:obj:`t`) dimension, accounting for the temporal evolution of the signal within a node (i.e., a sensor).
#. The **Node** (:obj:`n`) dimension, accounting for simultaneous observations measured at the different nodes in the network in a given time step.
#. The **Features** (:obj:`f`) or **Channels** dimension, allowing for multiple (heterogeneous) measurements at the same spatio-temporal point.

We call a **spatiotemporal graph** a tensor with finite :obj:`t`, :obj:`n`, and
:obj:`f` dimensions, paired with the underlying topology. In :tsl:`tsl`, we use the
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

.. admonition:: Dynamic graphs
    :class: tip

    If the graph connectivity changes over time, you can pass Python lists as
    :obj:`Data.edge_index` and :obj:`Data.edge_weight`.

We now consider a simple fully-connected, undirected graph with 3 nodes as the
underlying topology. We assume to have a univariate signal -- uniformly sampled
and synchronized across nodes -- on each node, plus a graph-wise exogenous
variable (may be, for instance, an encoding of time, equal for all nodes). If we
now want to forecast the next step given a sequence of 12 observations, our
``Data`` object would look like this:

.. code-block:: python

    import torch
    from tsl.data import Data

    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2],
                               [1, 2, 0, 2, 0, 1]], dtype=torch.long)

    input = dict(
        x=torch.randn(12, 3, 1),  # t=12  n=3  f=1
        u=torch.randn(12, 4)      # t=12       f=4
    )
    target = dict(
        y=torch.randn(1, 3, 1)    # t=1   n=3  f=1
    )

    data = Data(input=input, target=target, edge_index=edge_index)
    >>> Data(
          input=(x=[12, 3, 1], u=[12, 4]),
          target=(y=[1, 3, 1]),
          has_mask=False
        )

Since we know also to which dimension each axis refers to in the tensors, it is
a best practice to explicit them in the ``Data`` object through **patterns**.

.. code-block:: python

    pattern = dict(x='t n f', u='t f', y='t n f')

    data = Data(input=input, target=target, edge_index=edge_index,
                pattern=pattern)
    >>> Data(
          input=(x=[t=12, n=3, f=1], u=[t=12, f=4]),
          target=(y=[t=1, n=3, f=1]),
          has_mask=False
        )

.. admonition:: Patterns
    :class: hint

    The usage of patterns is not mandatory, although they clarify the dimensions
    of each tensor in a spatiotemporal graph object and are used internally by
    :tsl:`tsl` for operations on graphs (e.g., reduction to subgraph, temporal
    resampling, tensors collation).

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

Conversely, the method :meth:`~tsl.data.StaticBatch.get_example` allows
accessing the ``idx``-th sample in the batch. This can be equally
achieved through the ``__get_item__`` function as ``StaticBatch[idx]``, which
supports also slices. Note that you can use this function also on
``StaticBatch`` that have been directly instantiated, without the use of the
method :meth:`~tsl.data.StaticBatch.from_data_list`.

The ``DisjointBatch`` object
----------------------------

More generally, data at hand come from a possibly **dynamic** setting, in which also the
underlying topology changes over time. We supports two different types of
discrete-time dynamic graph signals:

* **Disjoint Graph Signals**, where the topology is static within the temporal
  window of a sample, but may change from a sample to another. This is a common
  scenario when we put together multiple temporal graph signals, each on a
  different (static) graph.

* **Dynamic Graph Signals**, where the topology may change not only from sample
  to sample, but also from a time step to another in the same temporal window.

The aggregation of samples into mini-batches is handled in both these cases by
the :class:`tsl.data.DisjointBatch` object, a subclass of
:class:`torch_geometric.data.Batch` for dynamic spatiotemporal graphs.


.. grid:: 1 1 2 2
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`repo;1em;sd-text-primary` tsl.data API
        :link: ../modules/data
        :link-type: doc
        :shadow: sm

        Read the docs of this module.


    .. grid-item-card::  :octicon:`file-code;1em;sd-text-primary` Notebook
        :link: ../notebooks/a_gentle_introduction_to_tsl
        :link-type: doc
        :shadow: sm

        Check the introductory notebook.
