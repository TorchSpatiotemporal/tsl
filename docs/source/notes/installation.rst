Installation
============

tsl is compatible with Python>=3.7. We recommend installation from source to be up-to-date with the latest version.

From Source
-----------

To install tsl from source, clone the repository, navigate to the library root
directory and install using :code:`pip`.

.. code-block:: bash

    git clone https://github.com/TorchSpatiotemporal/tsl.git
    cd tsl
    python setup.py install  # Or 'pip install .'

To solve all dependencies, we recommend using `Anaconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>`_ and the provided environment configuration by running the command:

.. code-block:: bash

    conda env create -f tsl_env.yml

Using pip
---------

Alternatively, you can install the library directly from :code:`pip`.

.. code-block:: bash

    pip install torch-spatiotemporal

Please refer to `PyG installation guidelines <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_ for installation of PyG ecosystem without conda.
