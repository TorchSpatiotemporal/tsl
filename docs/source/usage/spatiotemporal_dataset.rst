Spatiotemporal Dataset
======================

.. warning::

    This page is still under development.

An elegant and effective solution to handle datasets in PyTorch is by means of
the :class:`torch.utils.data.Dataset` object. This object allows us to access
the samples as in any Python mapping object, by implementing the
``__getitem__()`` and ``__len__()`` protocols. As such, a the ``idx``-th sample
in such a map-style dataset is retrieved by ``dataset[idx]`` (see the `PyTorch
tutorial <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_
on datasets and dataloaders for more information).

The main class in :tsl:`tsl` for handling spatiotemporal datasets is the
:class:`tsl.data.SpatioTemporalDataset` object, that inherits directly from the
PyTorch ``Dataset``. The core functionality of :obj:`SpatioTemporalDataset` is
to map (long) sequence of spatiotemporal data into :class:`tsl.data.Data`
samples. In this section, we explain in details how to create
properly ``SpatioTemporalDataset`` objects.


Sliding window
--------------

A common approach in time-series analysis when dealing with long sequences
of data consists in splitting the data along the temporal dimension in
**sliding windows** of fixed length. Then, for supervised learning methods,
the created windows are associated with a label that can be a new sequence of
data (as in regression problems) or a class (in the case of classification
problems).

The :class:`~tsl.data.SpatioTemporalDataset` object eases the creation of such a
dataset from tabular data. In particular, the parameters used to define how to
create the windows from the entire sequence are the following:

* ``window``:  length of the temporal window.
* ``horizon``:  length of the target sequence (e.g., forecasting horizon).
* ``delay``: number of steps between the window's end and the target sequence's
  beginning.
* ``stride``: number of steps between a sample and the next one.
* ``window_lag``: window's sampling frequency (in number of time steps).
* ``horizon_lag``:  horizon's sampling frequency (in number of time steps).

:numref:`fig_sliding_window` shows graphically how these parameters affect the
dataset partitioning into windows.

.. _fig_sliding_window:

.. figure:: /_static/img/sliding_window.svg
    :align: center
    :width: 80%

    Sliding window parameters.

In the case illustrated in the figure, we have ``window=6``,
``horizon=4``, ``delay=3``, and ``stride=3``, with unitary ``window_lag``
and ``horizon_lag``. Note that the number of samples
:attr:`~tsl.data.SpatioTemporalDataset.n_samples` will always be lower than
the number of time steps :attr:`~tsl.data.SpatioTemporalDataset.n_steps`.

.. note::

    The :class:`~tsl.data.SpatioTemporalDataset` object is automatically
    partitioned into samples every time that any of these parameter is updated.
    You can override the computed windows by assigning to the dataset specific
    sample indices (see :meth:`~tsl.data.SpatioTemporalDataset.set_indices`).

We report in :numref:`tab_prediction_examples` some example configuration for
prediction/forecasting problems.

.. _tab_prediction_examples:

.. list-table:: Examples of windowing parameters settings (prediction).
    :align: center
    :widths: 28 18 18 18 18
    :header-rows: 1
    :stub-columns: 1

    * -
      - Window
      - Horizon
      - Delay
      - Stride
    * - :math:`H`-step-ahead prediction
      - Any
      - :math:`H`
      - 0
      - Any
    * - :math:`L`-lagged :math:`H`-step-ahead prediction
      - Any
      - :math:`H`
      - :math:`L`
      - Any
    * - :math:`H`-step-ahead predictions (disjoint windows)
      - Any
      - :math:`H`
      - 0
      - :math:`H`

Nonetheless, we can play around with these parameters to enable more complex
configuration, as for instance window reconstruction.
:numref:`tab_imputation_examples` shows some examples on how to set the
windowing parameters for imputation.

.. _tab_imputation_examples:

.. list-table:: Examples of windowing parameters settings (imputation).
    :align: center
    :widths: 28 18 18 18 18
    :header-rows: 1
    :stub-columns: 1

    * -
      - Window
      - Horizon
      - Delay
      - Stride
    * - In-window imputation
      - :math:`W`
      - :math:`W`
      - :math:`-W`
      - Any
    * - In-window imputation with :math:`K`-th steps of warmup
      - :math:`W`
      - :math:`W - K`
      - :math:`-W`
      - Any
    * - :math:`t`-th step imputation
      - :math:`2t - 1`
      - :math:`1`
      - :math:`-t`
      - :math:`1`


.. admonition:: ImputationDataset
    :class: tip

    The :class:`tsl.data.ImputationDataset` object provides shortcut APIs for
    the creation of :class:`~tsl.data.SpatioTemporalDataset` objects tailored
    for the imputation task.


Adding spatiotemporal data
--------------------------

A spatiotemporal dataset need spatiotemporal data. In standard autoregressive
problems (e.g., forecasting), the objective is to model future values of a time
series conditioned to a (finite) set of past observations. We call the
3-dimensional tensor representing this time series -- spanning over temporal,
spatial and features dimensions -- the **target** of the dataset.

The ``target`` argument is the only mandatory argument for creating a
:class:`~tsl.data.SpatioTemporalDataset`. Unless otherwise specified, the tensor
set as :attr:`~tsl.data.SpatioTemporalDataset.target` is mapped in dataset sample
``dataset[idx]`` as:

.. (see `Mapping tensors to graph attributes`_)

* ``dataset[idx].x``, the sequence of past observations, lasting for
  ``dataset.window`` time steps.
* ``dataset[idx].y``, the sequence of future values with length
  ``dataset.horizon``.

.. note::

    The :attr:`~tsl.data.SpatioTemporalDataset.target` tensor is assumed to have
    always three dimensions: time, nodes (i.e., spatial points) and features. If
    the input data is bi-dimensional, then a dummy uni-dimensional feature is
    inferred.

Any other data coming into play is handled as a **covariate** to the target
sequence. Covariates are not restricted to a specific shape or number of
dimensions. It is a good practice to specify to which dimension each axis in the
data refers to by means of **patterns**.


.. grid:: 1 1 2 2
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`repo;1em;sd-text-primary` SpatioTemporalDataset API
        :link: ../modules/data_pytorch_datasets
        :link-type: doc
        :shadow: sm

        See more about the class APIs.


    .. grid-item-card::  :octicon:`file-code;1em;sd-text-primary` Notebook
        :link: ../notebooks/a_gentle_introduction_to_tsl
        :link-type: doc
        :shadow: sm

        Check the introductory notebook.


..  Understanding patterns
    ++++++++++++++++++++++


    Spatial relationships
    ---------------------


    Mapping tensors to graph attributes
    -----------------------------------


    Understanding patterns
    ----------------------

    The `t > n > f` Convention
    ++++++++++++++++++++++++++
    In :tsl:`tsl`, tabular data of this form are represented by following the [Time, Node, Features]
    (T N F) convention. Considering the previous example, we represent measurements
    acquired by 400 air quality monitoring stations in a day (with a sampling interval
    of one hour) as a tensor :math:`\mathbf{X}` with dimensions :math:`\left(24, 400, 3 \right)`.

    .. Note::
        Unless otherwise stated, all layers and models in :mod:`tsl.nn` expect
        as input a 4-dim tensor shaped as :obj:`[batch_size, steps, nodes, channels]`.
