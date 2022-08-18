Data format
===========

TSL is meant to deal with discrete-time, spatio-temporal data. Therefore, common input examples
are data streams coming from sensor networks. In principle, data of this kind can be
represented by 3-dimensional tensors, with:

#. The **Time** (:math:`T`) dimension, accounting for the temporal evolution of the signal within a node (i.e., a sensor).
#. The **Node** (:math:`N`) dimension, accounting for simultaneous observations measured at the different nodes in the network in a discrete time step.
#. The **Features** (:math:`F`) or **Channels** dimension, allowing for multiple (heterogeneous) measurements at the same spatio-temporal point.

As a real-life example, you may think of an air quality monitoring system, in
which :math:`N` air quality monitoring stations are displaced in a geographic area, with every
station measuring :math:`F` different pollution parameters (like :math:`\text{PM}2.5, \text{PM}10, \text{CO}_2`).
In this example, the sizes of the node and features dimensions are :math:`N` and :math:`F`, respectively.

The "T N F" Convention
------------------

In TSL, tabular data of this form are represented by following the [Time, Node, Features]
(T N F) convention. Considering the previous example, we represent measurements
acquired by 400 air quality monitoring stations in a day (with a sampling interval
of one hour) as a tensor :math:`\mathbf{X}` with dimensions :math:`\left(24, 400, 3 \right)`.

.. Note::
    Unless otherwise stated, all layers and models in :mod:`tsl.nn` expect
    as input a 4-dim tensor shaped as :obj:`[batch_size, steps, nodes, channels]`.