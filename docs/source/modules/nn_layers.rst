Layers
======

This module contains all the neural layers available in :tsl:`tsl`.


Graph Convolutional Layers
--------------------------

The subpackage :mod:`tsl.nn.layers.graph_convs` contains the graph convolutional
layers.

.. currentmodule:: tsl.nn.layers.graph_convs
.. autosummary::
   :nosignatures:
   {% for cls in tsl.nn.layers.graph_convs.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.nn.layers.graph_convs
    :members:


Recurrent Layers
----------------

The subpackage :mod:`tsl.nn.layers.recurrent` contains the cells used in
encoders that process the input sequence in a recurrent fashion.

.. currentmodule:: tsl.nn.layers.recurrent

Base classes
++++++++++++

.. autosummary::
   :nosignatures:
   {% for cls in tsl.nn.layers.recurrent.base_classes %}
     {{ cls }}
   {% endfor %}

Implemented cells
+++++++++++++++++

.. autosummary::
   :nosignatures:
   {% for cls in tsl.nn.layers.recurrent.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.nn.layers.recurrent
    :members:


Multi Layers
------------

The subpackage :mod:`tsl.nn.layers.multi` contains the layers that perform an
operation using a different set of parameters for the different instances
stacked in a dimension of the input data (e.g., the node dimension). They can be
used to process with independent parameters each node (or time step), breaking
the permutation equivariant property of the original operation.

.. currentmodule:: tsl.nn.layers.multi
.. autosummary::
   :nosignatures:
   {% for cls in tsl.nn.layers.multi.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.nn.layers.multi
    :members:


Normalization Layers
--------------------

The subpackage :mod:`tsl.nn.layers.norm` contains the normalization layers.

.. currentmodule:: tsl.nn.layers.norm
.. autosummary::
   :nosignatures:
   {% for cls in tsl.nn.layers.norm.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.nn.layers.norm
    :members:


Base Layers
-----------

The subpackage :mod:`tsl.nn.layers.base` contains basic layers used at the core
of other layers.

.. currentmodule:: tsl.nn.layers.base
.. autosummary::
   :nosignatures:
   {% for cls in tsl.nn.layers.base.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.nn.layers.base
    :members:


Operational Layers
------------------

The subpackage :mod:`tsl.nn.layers.ops` contains operational layers that do not
involve learnable parameters.

.. currentmodule:: tsl.nn.layers.ops
.. autosummary::
   :nosignatures:
   {% for cls in tsl.nn.layers.ops.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.nn.layers.ops
    :members:
