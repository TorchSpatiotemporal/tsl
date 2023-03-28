Transforms
==========

In the spirit of `torchvision <https://pytorch.org/vision/stable/>`_ and
:pyg:`null` `PyG <https://www.pyg.org/>`_, this module contains transform operations
called on every :class:`~tsl.data.SpatioTemporalDataset` item get.
A ``transform`` object expects a :class:`~tsl.data.Data` object as input and
returns a transformed object of the same type.

.. currentmodule:: tsl.transforms
.. autosummary::
   :nosignatures:
   {% for cls in tsl.transforms.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.transforms
   :members:
