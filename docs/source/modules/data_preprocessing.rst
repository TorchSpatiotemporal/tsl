.. py:module:: tsl.data.preprocessing
.. currentmodule:: tsl.data.preprocessing

Preprocessing
=============

The module :mod:`tsl.data.preprocessing` exposes API to preprocess
spatiotemporal data.


Scalers
-------

.. currentmodule:: tsl.data.preprocessing.scalers
.. autosummary::
   :nosignatures:
   {% for cls in tsl.data.preprocessing.scaler_classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.data.preprocessing.scalers
    :members:
