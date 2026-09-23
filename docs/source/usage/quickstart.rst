Quickstart
==========

Installation
------------

The easiest way to install the package and its required dependencies is:

.. code-block:: bash

    pip install torch-spatiotemporal

This command installs the latest stable release from PyPI. To install the latest
development version instead of the latest stable release, run the following command:

.. code-block:: bash

    pip install git+https://github.com/TorchSpatiotemporal/tsl.git

The following requirements apply to :tsl:`tsl` versions before 1.0 (``tsl<1.0``):

* Python >= 3.8 and < 3.12;
* NumPy < 2;
* PyTorch >= 1.13 and < 2.4; and
* PyG >= 2.4.

The ``torch-scatter`` and ``torch-sparse`` packages are optional. The former
is only needed for sparse spatiotemporal attention, while the latter is only
needed when using :class:`torch_sparse.SparseTensor` adjacency matrices. If you
need either package, install its wheel from the `PyG installation guide
<https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_
after installing PyTorch, since the wheel must match your PyTorch and
accelerator builds.

Quickstart with uv
++++++++++++++++++

`uv <https://docs.astral.sh/uv/>`_ is a fast Python package and environment
manager. The following creates a fresh environment and installs the package
in one command; ``--torch-backend=auto`` asks uv to select a compatible PyTorch
backend for the machine.

.. tab-set::

   .. tab-item:: Unix and macOS

      .. code-block:: bash

         uv venv --python 3.10
         source .venv/bin/activate
         uv pip install torch-spatiotemporal --torch-backend=auto
         # Optional libraries, when needed; use the matching PyG wheel:
         # uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

   .. tab-item:: Windows PowerShell

      .. code-block:: powershell

         uv venv --python 3.10
         .venv\Scripts\Activate.ps1
         uv pip install torch-spatiotemporal --torch-backend=auto
         # Optional libraries, when needed; use the matching PyG wheel:
         # uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html


.. admonition:: PyG optional libraries
   :class: caution

   Select the matching optional-library command from the `PyG installation guide
   <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_
   after PyTorch has been installed.


Installing using conda
++++++++++++++++++++++

The repository includes ``conda_env.yml`` for users who prefer conda. Clone
the repository, navigate to its root, and create the environment:

.. code:: bash

    git clone https://github.com/TorchSpatiotemporal/tsl.git
    cd tsl
    conda env create -f conda_env.yml

Then, activate the environment and install :tsl:`tsl` using :code:`pip`.

.. code:: bash

    conda activate tsl
    pip install -e .

The provided environment is configured for GPU use: it includes the ``nvidia``
channel and the ``pytorch-cuda`` dependency. For a CPU-only installation,
remove the ``nvidia`` channel and ``pytorch-cuda`` from ``conda_env.yml``
before running ``conda env create``. The optional ``pytorch-scatter`` and
``pytorch-sparse`` packages in the conda file can likewise be removed unless
your workload needs sparse attention or ``torch_sparse.SparseTensor``
connectivity.

Example scripts
---------------

The github repository hosts `example scripts <https://github.com/TorchSpatiotemporal/tsl/tree/main/examples>`_ and `notebooks <https://github.com/TorchSpatiotemporal/tsl/tree/main/examples/notebooks>`_ on how to use the library for different use cases, such as spatiotemporal predictions and imputations.
You can refer to the notebook :doc:`../notebooks/a_gentle_introduction_to_tsl` for an
introductory overview of the library main functionalities.

.. raw:: html

    <a target="_blank" href="https://colab.research.google.com/github/TorchSpatiotemporal/tsl/blob/main/examples/notebooks/a_gentle_introduction_to_tsl.ipynb">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
