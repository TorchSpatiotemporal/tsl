Models
======


General Models
--------------

.. currentmodule:: tsl.nn.models
.. autosummary::
   :nosignatures:
   {% for cls in tsl.nn.models.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.nn.models
    :members:


Spatio-Temporal Prediction Models
---------------------------------

.. currentmodule:: tsl.nn.models.stgn
.. autosummary::
   :nosignatures:
   {% for cls in tsl.nn.models.stgn.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.nn.models.stgn
    :members:


Spatio-Temporal Imputation Models
---------------------------------

.. currentmodule:: tsl.nn.models.imputation
.. autosummary::
   :nosignatures:
   {% for cls in tsl.nn.models.imputation.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.nn.models.imputation
    :members:
