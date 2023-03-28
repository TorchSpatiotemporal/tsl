Quickstart
==========

Installation
------------

:tsl:`tsl` is compatible with Python>=3.8. We recommend installation
on a `Anaconda or Miniconda <https://conda.io/projects/conda/en/latest/user-guide/install>`_
environment or a `virtual env <https://docs.python.org/3/library/venv.html>`_.

.. admonition:: Before installation
   :class: caution

   :tsl:`tsl` is built upon `PyTorch>=1.9 <https://pytorch.org/>`_ and
   `PyG>=2.0.3 <https://github.com/pyg-team/pytorch_geometric/>`_. Make sure you have
   both installed in your environment before installing :tsl:`tsl`. In the following,
   we provide instructions on how to install them for the chosen installation
   procedure.


Installing using conda
++++++++++++++++++++++

.. tip::

    Using conda allows to automatically solve PyTorch and PyG dependencies,
    choosing the latest CUDA version available supported by the system.

To install :tsl:`tsl` using conda, clone the repository, navigate to the library root
directory and create a new conda environment using the provided conda configuration:

.. code:: bash

    git clone https://github.com/TorchSpatiotemporal/tsl.git
    cd tsl
    conda env create -f conda_env.yml

Then, activate the environment and install :tsl:`tsl` using :code:`pip`.

.. code:: bash

    conda activate tsl
    python setup.py install  # Or 'pip install .'

.. note::

   Installation of :tsl:`tsl` directly from conda is on the roadmap!


Installing using pip
++++++++++++++++++++

Alternatively, you can install the library directly from :code:`pip`. Please
refer to `PyTorch <https://pytorch.org/>`_ and `PyG installation guidelines <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_
for installation without conda. After having installed the libraries, install
:code:`torch-spatiotemporal` using pip. For the latest version:

.. code-block:: bash

    pip install git+https://github.com/TorchSpatiotemporal/tsl.git

For the stable version:

.. code-block:: bash

    pip install torch-spatiotemporal


Example scripts
---------------

The github repository hosts `example scripts <https://github.com/TorchSpatiotemporal/tsl/tree/main/examples>`_ and `notebooks <https://github.com/TorchSpatiotemporal/tsl/tree/main/examples/notebooks>`_ on how to use the library for different use cases, such as spatiotemporal predictions and imputations.
You can refer to the notebook :doc:`../notebooks/a_gentle_introduction_to_tsl` for an introductory overview of the library main functionalities.

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
