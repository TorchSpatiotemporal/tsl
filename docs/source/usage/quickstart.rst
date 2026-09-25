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

The following requirements apply to the current :tsl:`tsl` release:

* Python >= 3.10;
* NumPy >= 1.26 and < 3;
* PyTorch >= 2.2 and < 2.13; and
* PyG >= 2.4 and < 3.

The ``torch-sparse`` package is optional and is only needed when using
:class:`torch_sparse.SparseTensor` adjacency matrices. It requires
``torch-scatter`` at runtime, so if you need ``torch-sparse``, install both compiled
packages from
the `PyG installation guide
<https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_
after installing PyTorch, since their wheels must match your PyTorch and
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
         uv pip install torch-spatiotemporal --torch-backend=auto
         # Optional SparseTensor support; use matching PyG wheels:
         # uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

   .. tab-item:: Windows PowerShell

      .. code-block:: powershell

         uv venv --python 3.10
         uv pip install torch-spatiotemporal --torch-backend=auto
         # Optional SparseTensor support; use matching PyG wheels:
         # uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html


.. admonition:: PyG optional libraries
   :class: caution

   Install matching ``torch-scatter`` and ``torch-sparse`` wheels using the
   optional-library command from the `PyG installation guide
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
before running ``conda env create``. The optional ``pytorch-sparse`` and ``pytorch-scatter`` packages in
the conda file can likewise be removed unless your workload needs
``torch_sparse.SparseTensor`` connectivity.

Compiling models
----------------

Models with :attr:`~tsl.nn.models.BaseModel.can_be_compiled` set to
:obj:`True` are covered by full-graph compilation tests. PyTorch compiles a
model in place through its standard module method:

.. code-block:: python

    model.compile(mode="reduce-overhead")

The forecasting and imputation example configurations expose the same option
as ``compile.enabled=true``. Optional ``compile.mode``, ``compile.fullgraph``,
and ``compile.dynamic`` values are forwarded to
:meth:`torch.nn.Module.compile`. Compilation incurs a warm-up cost and is most
useful when the model is called repeatedly with stable input shapes.

Example scripts
---------------

The github repository hosts `example scripts <https://github.com/TorchSpatiotemporal/tsl/tree/main/examples>`_ and `notebooks <https://github.com/TorchSpatiotemporal/tsl/tree/main/examples/notebooks>`_ on how to use the library for different use cases, such as spatiotemporal predictions and imputations.
You can refer to the notebook :doc:`../notebooks/a_gentle_introduction_to_tsl` for an
introductory overview of the library main functionalities.

.. raw:: html

    <a target="_blank" href="https://colab.research.google.com/github/TorchSpatiotemporal/tsl/blob/main/examples/notebooks/a_gentle_introduction_to_tsl.ipynb">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
