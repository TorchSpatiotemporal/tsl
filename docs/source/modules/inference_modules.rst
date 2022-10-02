Inference modules
=================

The module :mod:`tsl.inference` contains inference modules, i.e., modules meant
to wrap deep models in order to ease training and inference phases.
Every inference module extends a :class:`~pytorch_lightning.core.LightningModule`.

Currently, there are two basic inference modules:

* :class:`~tsl.inference.Predictor`

  Use this module when you are addressing the prediction task, i.e., inference
  of future observations.

* :class:`~tsl.inference.Imputer`

  Use this module when you are addressing the imputation task, i.e.,
  reconstruction of missing observations.

We suggest to extends this module and override some of the methods to meet your
project-specific needs.

.. automodule:: tsl.inference
    :members:
