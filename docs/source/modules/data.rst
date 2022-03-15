Data handling
=============

.. currentmodule:: tsl.data


Data
----

.. autosummary::
   :nosignatures:
   {% for cls in tsl.data.data_classes %}
     {{ cls }}
   {% endfor %}

{% for cls in tsl.data.data_classes %}
.. autoclass:: {{ cls }}
    :inherited-members:
{% endfor %}

Dataset
-------

.. autosummary::
   :nosignatures:
   {% for cls in tsl.data.dataset_classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.data
    :members: {% for cls in tsl.data.dataset_classes %}{{ cls }}, {% endfor %}
