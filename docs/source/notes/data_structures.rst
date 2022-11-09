Data Structures
===============

    This page is still under development

The ``SpatioTemporalData`` object
---------------------------------

    This page is still under development

The :class:`tsl.data.Data` object contains attributes related to a single spatiotemporal graph:

* :obj:`Data.input`: view on the tensors stored in :obj:`Data` that are meant to serve as input to the model.
  In the simplest case of a single node-attribute matrix, we could just have :obj:`Data.input.x`.
* :obj:`Data.target`: view on the tensors stored in :obj:`Data` used as labels to train the model.
  In the common case of a single label, we could just have :obj:`Data.input.y`.
* :obj:`Data.edge_index`: graph connectivity.
* :obj:`Data.edge_weight`: weights of the graph connectivity, if :obj:`Data.edge_index` is not a :class:`torch_sparse.SparseTensor`.
* :obj:`Data.mask`: binary mask indicating the data in :obj:`Data.target.y` to be used
  as ground-truth for the loss (default is :obj:`None`).
* :obj:`Data.transform`: mapping of :class:`~tsl.data.preprocessing.scalers.ScalerModule`, whose keys must be
  transformable (or transformed) tensors in :obj:`Data`.
* :obj:`Data.pattern`: mapping containing the pattern for each tensor in :obj:`Data`.

None of these attributes are required and custom attributes can be seamlessly added.


The ``SpatioTemporalBatch`` object
----------------------------------

    This page is still under development

The ``SpatioTemporalDataset`` object
------------------------------------

    This page is still under development