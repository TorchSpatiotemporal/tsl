.. currentmodule:: tsl.data

PyTorch Datasets
================

.. autosummary::
   :nosignatures:
   {% for cls in tsl.data.dataset_classes %}
     {{ cls }}
   {% endfor %}

{% for cls in tsl.data.dataset_classes %}
.. autoclass:: {{ cls }}
    :members:
{% endfor %}
