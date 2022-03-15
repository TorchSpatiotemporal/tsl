Inference modules
=================

In TSL, inference engines are packed into :class:`~pytorch_lightning.core.LightningModule`.
These modules are meant to wrap deep models in order to ease training and inference phases.

Currently there are two categories of inference modules:

#. `Predictors`_ (in :mod:`tsl.predictors`).
#. `Imputers`_ (in :mod:`tsl.imputers`).


Predictors
----------

.. currentmodule:: tsl.predictors
.. autosummary::
   :nosignatures:
   {% for cls in tsl.predictors.predictor_classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.predictors
    :members:


Imputers
--------

.. currentmodule:: tsl.imputers
.. autosummary::
   :nosignatures:
   {% for cls in tsl.imputers.imputer_classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.imputers
    :members:
