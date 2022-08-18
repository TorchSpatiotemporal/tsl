Datasets
========

.. currentmodule:: tsl.datasets

The module :mod:`tsl.datasets` contains all the publicly available
spatiotemporal datasets provided by the library, as well as prototype classes
for creating new datasets.


Datasets in TSL
---------------

.. autosummary::
   :nosignatures:
   {% for cls in tsl.datasets.dataset_classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.datasets
    :members: {% for cls in tsl.datasets.dataset_classes %}{{ cls }}, {% endfor %}
    :noindex:

Prototypes
----------

The submodule :mod:`tsl.datasets.prototypes` provides interfaces that can help in creating new
datasets. All datasets provided by the library are implemented following these
interfaces.

The most general interface is :class:`tsl.datasets.Dataset`, which is the
parent class for every dataset in tsl. The more complete class
:class:`tsl.datasets.TabularDataset` provides useful functionalities for
multivariate time series datasets that can be expressed by
:class:`pandas.DataFrame`.

.. autosummary::
   :nosignatures:
   {% for cls in tsl.datasets.prototypes.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.datasets
    :members: {% for cls in tsl.datasets.prototype_classes %}{{ cls }}, {% endfor %}
    :inherited-members:
