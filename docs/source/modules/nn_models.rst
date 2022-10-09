Models
======

.. currentmodule:: tsl.nn.models

.. autoclass:: BaseModel
    :members:

Spatiotemporal Models
---------------------

.. currentmodule:: tsl.nn.models.stgn
.. autosummary::
   :nosignatures:
   {% for cls in tsl.nn.models.stgn.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.nn.models.stgn
    :members:
    :undoc-members:
    :exclude-members: training, add_model_specific_args


Temporal Models
---------------

.. currentmodule:: tsl.nn.models.temporal
.. autosummary::
   :nosignatures:
   {% for cls in tsl.nn.models.temporal.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.nn.models.temporal
    :members:
    :undoc-members:
    :exclude-members: training, add_model_specific_args
