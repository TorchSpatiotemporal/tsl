.. py:module:: tsl.datasets.prototypes
.. currentmodule:: tsl.datasets.prototypes

Prototypes
==========

The submodule :mod:`tsl.datasets.prototypes` provides interfaces that can help
in creating new datasets. All datasets provided by the library are implemented
extending these interfaces.

The most general interface is :class:`~tsl.datasets.prototypes.Dataset`, which is the
parent class for every dataset in :tsl:`tsl`. The more complete class
:class:`~tsl.datasets.prototypes.TabularDataset` provides useful functionalities for
multivariate time series datasets with data in a tabular format, i.e., with
time, node and feature dimensions. Data passed to this dataset should be
:class:`pandas.DataFrame` and/or :class:`numpy.ndarray`. Missing values are
supported either by setting as :obj:`nan` the missing entry or by explicitly
setting the attribute :attr:`~tsl.datasets.prototypes.TabularDataset.mask`.

If your data are timestamped, meaning that each observation is associated with
a specific date and time, then you can consider using
:class:`~tsl.datasets.prototypes.DatetimeDataset`, which extends
:class:`~tsl.datasets.prototypes.TabularDataset` and provides additional functionalities
for temporal data (e.g., :meth:`~tsl.datasets.prototypes.DatetimeDataset.datetime_encoded`,
:meth:`~tsl.datasets.prototypes.DatetimeDataset.resample`). This class accepts
:class:`~pandas.DataFrame` with index of type :class:`~pandas.DatetimeIndex` and
columns of type :class:`~pandas.MultiIndex` (with :obj:`nodes` as the first level
and :obj:`channels` as the second) for the :attr:`~tsl.datasets.prototypes.TabularDataset.target`.

.. autosummary::
   :nosignatures:
   {% for cls in tsl.datasets.prototypes.classes %}
     {{ cls }}
   {% endfor %}

{% for cls in tsl.datasets.prototype_classes %}
.. autoclass:: {{ cls }}
    :members:
    :inherited-members:
{% endfor %}
