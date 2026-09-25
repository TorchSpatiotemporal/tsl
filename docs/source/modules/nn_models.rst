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


Foundation Models
-----------------

Foundation models are pretrained forecasting models available through optional
dependencies. Install them with ``pip install torch-spatiotemporal[foundation]``.
The adapters currently cover the Amazon Chronos family and Google Research
TimesFM 3. Chronos-2 and TimesFM can use nodes within a sample as covariates,
while batch samples always remain independent. Check the upstream checkpoint
license before deploying pretrained weights.

For an input shaped ``b t n f``, ``nodes_as_covariates=False`` forecasts the
``b n`` multivariate ``f t`` series independently. When it is :obj:`True`,
nodes belonging to the same sample can cross-learn, but no model call or
attention group ever combines entries from the ``b`` axis. Inputs shaped
``b t f`` are treated as ``b`` independent multivariate time series.

.. currentmodule:: tsl.nn.models.foundation
.. autosummary::
   :nosignatures:
   {% for cls in tsl.nn.models.foundation.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.nn.models.foundation
    :members:
    :undoc-members:
    :exclude-members: training, add_model_specific_args
