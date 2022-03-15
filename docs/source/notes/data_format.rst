Data format
===========

TSL is meant to deal with spatio-temporal data. Therefore, common input examples
are data streams coming from sensor networks. In principle, such data can be
naturally casted into 3 dimensions:

#. The **Step** dimension, accounting for the temporal evolution of the signal within a node (i.e., a sensor).
#. The **Node** dimension, accounting for the observations measured at the different nodes in the network.
#. The **Channel** dimension, allowing for multiple, heterogeneous measurements.

As a real-life example, you may think of an air quality monitoring system, in
which hundreds stations are displaced in a wide geographic area, with every
station measuring common pollution parameters, like [:obj:`PM2.5`, :obj:`PM10`, :obj:`CO2`].

"S N C" Convention
------------------

In TSL, we order multi-dimensional data by following the "Step, Node, Channel"
("S N C") convention. Considering the previous example, we represent measurements
acquired by 400 air quality monitoring stations in a day (with a sampling interval
of one hour) as a tensor :math:`\mathbf{X}` with dimensions :math:`\left(24, 400, 3 \right)`.

.. Note::
    Unless otherwise stated, all layers and models in :mod:`tsl.nn` expect
    as input a 4-dim tensor shaped as :obj:`[batch_size, steps, nodes, channels]`.