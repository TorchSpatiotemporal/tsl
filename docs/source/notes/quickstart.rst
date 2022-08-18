Quickstart
==========

Installation
------------

tsl is compatible with Python>=3.7. We recommend installation from source to be up-to-date with the latest version.

Installing from Source
++++++++++++++++++++++

To install tsl from source, clone the repository, navigate to the library root
directory and install using :code:`pip`.

.. code-block:: bash

    git clone https://github.com/TorchSpatiotemporal/tsl.git
    cd tsl
    python setup.py install  # Or 'pip install .'

To solve all dependencies, we recommend using `Anaconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>`_ and the provided environment configuration by running the command:

.. code-block:: bash

    conda env create -f tsl_env.yml

Installing using pip
++++++++++++++++++++

Alternatively, you can install the library directly from :code:`pip`.

.. code-block:: bash

    pip install torch-spatiotemporal

Please refer to `PyG installation guidelines <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_ for installation of PyG ecosystem without conda.

Example scripts
---------------

The github repository hosts `example scripts <https://github.com/TorchSpatiotemporal/tsl/tree/main/examples>`_ and `notebooks <https://github.com/TorchSpatiotemporal/tsl/tree/main/examples/notebooks>`_ on how to use the library for different use cases, such as spatiotemporal predictions and imputations.
You can refer to the notebook `A Gentle Introduction to TSL <https://colab.research.google.com/github/TorchSpatiotemporal/tsl/blob/main/examples/notebooks/a_gentle_introduction_to_tsl.ipynb>`_ for an introductory overview of the library main functionalities.

.. raw:: html

    <a target="_blank" href="https://colab.research.google.com/github/TorchSpatiotemporal/tsl/blob/main/examples/notebooks/a_gentle_introduction_to_tsl.ipynb">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>

Citing
------

If you use Torch Spatiotemporal for your research, please consider citing the library

{% raw %}
.. code-block:: latex

    @software{Cini_Torch_Spatiotemporal_2022,
        author = {Cini, Andrea and Marisca, Ivan},
        license = {MIT},
        month = {3},
        title = {{Torch Spatiotemporal}},
        url = {https://github.com/TorchSpatiotemporal/tsl},
        year = {2022}
    }
{% endraw %}