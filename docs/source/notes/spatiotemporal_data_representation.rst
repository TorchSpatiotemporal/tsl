Spatiotemporal Data Representation
==================================

    This page is still under development

TSL is meant to deal with discrete-time, spatiotemporal data. Common input examples
are data streams coming from sensor networks. In principle, data of this kind can be
represented by 3-dimensional tensors, with:

#. The **Time** (:obj:`t`) dimension, accounting for the temporal evolution of the signal within a node (i.e., a sensor).
#. The **Node** (:obj:`n`) dimension, accounting for simultaneous observations measured at the different nodes in the network in a discrete time step.
#. The **Features** (:obj:`f`) or **Channels** dimension, allowing for multiple (heterogeneous) measurements at the same spatio-temporal point.

As a real-life example, you may think of an air quality monitoring system with
:math:`N` air quality monitoring stations, each measuring :math:`F` different pollution
parameters (e.g., PM2.5, PM10, CO2).

.. note::

    We are planning to improve support for irregular sampling and continuous time dynamic graphs in future versions of tsl.

We denote by :math:`X_{t} \in \mathbb{R}^{N \times F}` the :math:`F`-dimensional measurements acquired by the :math:`N`
sensors at time :math:`t`, with :math:`X_{t:t+T}` being the sequence of measurements in the interval :math:`[t, t+T)`.


Sliding window
--------------

.. figure:: /_static/img/sliding_window.svg
    :align: center
    :width: 80%

    Sliding window parameters.

.. list-table:: Examples of windowing parameters settings with hourly data.
    :align: center
    :widths: 28 18 18 18 18
    :header-rows: 1
    :stub-columns: 1

    * -
      - Window
      - Horizon
      - Delay
      - Stride
    * - :math:`K`-step-ahead prediction
      - Any
      - :math:`K`
      - 0
      - Any
    * - :math:`L`-lagged :math:`K`-step-ahead prediction
      - Any
      - :math:`K`
      - :math:`L`
      - Any
    * - :math:`K`-step-ahead predictions (no overlap)
      - Any
      - :math:`K`
      - 0
      - :math:`K`
    * - Watch today, predict tomorrow
      - 24
      - 24
      - 0
      - 24
    * - Impute central hour
      - 24
      - 1
      - -13
      - 1

Understanding patterns
----------------------

The `t > n > f` Convention
++++++++++++++++++++++++++
In TSL, tabular data of this form are represented by following the [Time, Node, Features]
(T N F) convention. Considering the previous example, we represent measurements
acquired by 400 air quality monitoring stations in a day (with a sampling interval
of one hour) as a tensor :math:`\mathbf{X}` with dimensions :math:`\left(24, 400, 3 \right)`.

.. Note::
    Unless otherwise stated, all layers and models in :mod:`tsl.nn` expect
    as input a 4-dim tensor shaped as :obj:`[batch_size, steps, nodes, channels]`.