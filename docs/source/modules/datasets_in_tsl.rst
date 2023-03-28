.. py:module:: tsl.datasets
.. currentmodule:: tsl.datasets

Datasets in :tsl:`tsl`
======================

.. autosummary::
   :nosignatures:
   {% for cls in tsl.datasets.dataset_classes %}
     {{ cls }}
   {% endfor %}

{% for cls in tsl.datasets.dataset_classes %}
.. autoclass:: {{ cls }}
    :members:
{% endfor %}
