Datasets
========

.. currentmodule:: tsl.datasets


Prototypes
----------

TSL provides interfaces that can help in defining datasets. All the datasets in
tsl are implemented following these interfaces. They are also simple to extend
for adding new custom datasets.

The most general interface is :obj:`Dataset` and is the parent class for every
dataset in TSL.

.. autosummary::
   :nosignatures:
   {% for cls in tsl.datasets.prototype_classes %}
     {{ cls }}
   {% endfor %}

.. autoclass:: Dataset
    :inherited-members:

.. autoclass:: PandasDataset


Datasets in TSL
---------------

.. autosummary::
   :nosignatures:
   {% for cls in tsl.datasets.dataset_classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.datasets
    :members: {% for cls in tsl.datasets.dataset_classes %}{{ cls }}, {% endfor %}
