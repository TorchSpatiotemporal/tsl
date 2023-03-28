Inference engines
=================

The module :mod:`tsl.engines` contains inference engines, i.e., modules meant
to wrap deep models in order to ease training and inference phases.
Every engine extends a :class:`~pytorch_lightning.core.LightningModule` from
:lightning:`Lightning`.

Currently, there are two basic engines:

* :class:`~tsl.engines.Predictor`

  Use this module when you are addressing the prediction task, i.e., inference
  of future observations.

* :class:`~tsl.engines.Imputer`

  Use this module when you are addressing the imputation task, i.e.,
  reconstruction of missing observations.

We suggest to extends this module and override some of the methods to meet your
project-specific needs.

.. automodule:: tsl.engines
    :members:
