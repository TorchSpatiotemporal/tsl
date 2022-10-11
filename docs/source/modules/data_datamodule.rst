.. py:module:: tsl.data.datamodule
.. currentmodule:: tsl.data.datamodule

Organizing data
===============


DataModule
----------

.. autosummary::
   :nosignatures:
   {% for cls in tsl.data.datamodule.datamodule_classes %}
     {{ cls }}
   {% endfor %}

{% for cls in tsl.data.datamodule.datamodule_classes %}
.. autoclass:: {{ cls }}
    :members:
{% endfor %}


Splitters
---------

.. currentmodule:: tsl.data.datamodule.splitters
.. autosummary::
   :nosignatures:
   {% for cls in tsl.data.datamodule.splitter_classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.data.datamodule.splitters
    :members:
