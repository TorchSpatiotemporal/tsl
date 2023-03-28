:github_url: https://github.com/TorchSpatiotemporal/tsl

Torch Spatiotemporal
====================

.. raw:: html

    <div id="particles-js">
        <div class="hero-shade"></div>
        <div class="hero-content">
            <img class="hero-logo" src="_static/img/tsl_logo_text.svg"/>
            <p class="hero-lead">Neural spatiotemporal forecasting with PyTorch</p>
        </div>
    </div>
    <script src="_static/js/particles.js"></script>
    <script src="_static/js/particles_config.js"></script>

:tsl:`null` **Torch Spatiotemporal** (tsl) is a python library for **neural spatiotemporal data processing**, with a focus on Graph Neural Networks.

It is built upon the most used libraries of the python scientific computing ecosystem, with the final objective of providing a straightforward process that goes from data preprocessing to model prototyping. In particular, :tsl:`tsl` offers a wide range of utilities to develop neural networks in :pytorch:`null` `PyTorch <https://pytorch.org/>`_ and :pyg:`null` `PyG <https://www.pyg.org/>`_ for processing spatiotemporal data signals.

In detail, the package provide:

* High-level and easy-to-use APIs to build you own datasets and models for sensor networks.
* Tools to deal with irregularities in the data stream: missing data, variations in the underlying network, etc.
* Automatization of the preprocessing phase, with methods to scale and detrend the time series (see :doc:`modules/data_preprocessing` section).
* A set of most used datasets in spatiotemporal data processing literature (see :doc:`modules/datasets` section).
* A straightforward way of building spatiotemporal datasets that work with :pytorch:`null` `PyTorch <https://pytorch.org/>`_ and :pyg:`null` `PyG <https://www.pyg.org/>`_ (see :doc:`modules/data` section).
* Out-of-the-box scalability -- from a single CPU to clusters of GPUs -- with :lightning:`null` `PyTorch Lightning <https://www.pytorchlightning.ai/>`_  (see :doc:`modules/engines` section).
* Plug-and-play state-of-the-art models from neural spatiotemporal literature (see :doc:`modules/nn_models` section).
* A collection of neural layers for creating neural spatiotemporal models in a fast and modular way (see :doc:`modules/nn_layers` section).
* A standard for experiment reproducibility based on the :hydra:`null` `Hydra <https://hydra.cc/>`_ framework, to promote and support research on spatiotemporal data mining  (see :doc:`modules/experiment` section).

----

.. pull-quote::

    "If I have seen further it is by standing on the shoulders of Giants."

    -- Isaac Newton

:tsl:`tsl` relies heavily on these libraries for its functionalities:

.. grid:: 3 6 6 6
    :gutter: 2

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

    .. grid-item-card::
        :class-card: carousel-logo
        :img-background: _static/img/logos/hydra.svg
        :link: https://hydra.cc/

----

Get started
-----------

.. grid:: 1 1 2 2
    :margin: 3 0 0 0
    :gutter: 2
    :padding: 0

    .. grid-item-card::  :octicon:`rocket;1em;sd-text-primary` Installation
        :link: usage/quickstart
        :link-type: doc
        :shadow: sm

        Read the guide on how to to install :tsl:`tsl` on your system.

    .. grid-item-card::  :octicon:`gear;1em;sd-text-primary` Usage
        :link: usage/spatiotemporal_dataset
        :link-type: doc
        :shadow: sm

        Look at the basic functionalities of :tsl:`tsl` for spatiotemporal data processing.

    .. grid-item-card::  :octicon:`file-code;1em;sd-text-primary` Notebooks
        :link: usage/notebooks
        :link-type: doc
        :shadow: sm

        Check the notebooks for tutorial to use :tsl:`tsl` at the best.

    .. grid-item-card::  :octicon:`repo;1em;sd-text-primary` Package API
        :link: py-modindex
        :link-type: doc
        :shadow: sm

        In the index, you can find the main API for each submodule.


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Usage
   :hidden:

   usage/quickstart
   usage/data_structures
   usage/spatiotemporal_dataset
   usage/notebooks

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package API
   :hidden:

   modules/data
   modules/datasets
   modules/nn
   modules/engines
   modules/metrics
   modules/experiment
   modules/transforms
   modules/ops

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Indices
   :hidden:

   genindex
   py-modindex
