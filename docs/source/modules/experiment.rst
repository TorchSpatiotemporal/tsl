.. py:module:: tsl.experiment
.. currentmodule:: tsl.experiment

Experiment
==========

The module :mod:`tsl.experiment` contains classes and utilities for experiment
pipelining, scalability and reproducibility. The main class in the package is
:class:`tsl.experiment.Experiment` and relies on :hydra:`null` `Hydra <https://hydra.cc/>`_
for managing configurations.


Experiment
----------

.. autoclass:: tsl.experiment.Experiment
   :members:

Loggers
-------

.. currentmodule:: tsl.experiment.loggers

.. autosummary::
   :nosignatures:
   {% for cls in tsl.experiment.loggers.logger_classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.experiment.loggers
   :members:
