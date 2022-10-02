:github_url: https://github.com/TorchSpatiotemporal/tsl

Torch Spatiotemporal
====================

.. raw:: html

    <div id="particles-js"></div>
    <script src="_static/js/particles.js"></script>
    <script src="_static/js/particles_config.js"></script>

Torch Spatiotemporal (ts) is a python library for neural spatiotemporal data processing, with a focus on Graph Neural Networks.

It is built upon the most used libraries of the python scientific computing ecosystem, with the final objective of providing a straightforward process that goes from data preprocessing to model prototyping. In particular, tsl offers a wide range of utilities to develop neural networks in `PyTorch <https://pytorch.org/>`_ and `PyG <https://github.com/pyg-team/pytorch_geometric/>`_ for processing spatiotemporal data signals.

.. highlights::

    On the shoulders of giants

.. grid:: 3 6 6 6

    .. grid-item-card::
        :class-card: carousel-logo
        :img-background: _static/img/logos/python.svg
        :link: https://www.python.org/

    .. grid-item-card::
        :class-card: carousel-logo
        :img-background: _static/img/logos/numpy.svg
        :link: https://numpy.org/

    .. grid-item-card::
        :class-card: carousel-logo
        :img-background: _static/img/logos/pandas.svg
        :link: https://pandas.pydata.org/

    .. grid-item-card::
        :class-card: carousel-logo
        :img-background: _static/img/logos/pytorch.svg
        :link: https://pytorch.org/

    .. grid-item-card::
        :class-card: carousel-logo
        :img-background: _static/img/logos/pyg.svg
        :link: https://www.pyg.org/

    .. grid-item-card::
        :class-card: carousel-logo
        :img-background: _static/img/logos/lightning.svg
        :link: https://www.pytorchlightning.ai/

----

Get started
+++++++++++

.. grid:: 1 1 2 2
    :margin: 3 0 0 0
    :padding: 0

    .. grid-item-card::  :octicon:`rocket;1em;sd-text-primary` Installation
        :link: notes/quickstart
        :link-type: doc
        :shadow: md

        Read the guide on how to to install tsl on your system.

    .. grid-item-card::  :octicon:`gear;1em;sd-text-primary` Usage
        :link: notes/spatiotemporal_data_representation
        :link-type: doc
        :shadow: md

        Look at the basic functionalities of tsl for spatiotemporal data processing.


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Usage
   :hidden:

   notes/quickstart
   notes/spatiotemporal_data_representation

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package API
   :hidden:

   modules/data
   modules/data_datamodule
   modules/data_preprocessing
   modules/datasets
   modules/nn
   modules/inference_modules
   modules/ops

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Indices
   :hidden:

   genindex
   py-modindex
