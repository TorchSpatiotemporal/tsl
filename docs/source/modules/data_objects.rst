Data objects
============

.. currentmodule:: tsl.data

.. autosummary::
   :nosignatures:
   {% for cls in tsl.data.data_classes %}
     {{ cls }}
   {% endfor %}

{% for cls in tsl.data.data_classes %}
.. autoclass:: {{ cls }}
    :members:
{% endfor %}
