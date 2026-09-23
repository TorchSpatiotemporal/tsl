<div align="center">
    <br><br>
    <img alt="Torch Spatiotemporal" src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo_text.svg" width="85%"/>
    <h3>Neural spatiotemporal forecasting with PyTorch</h3>
    <hr>
    <p>
    <a href='https://pypi.org/project/torch-spatiotemporal/'><img alt="PyPI" src="https://img.shields.io/pypi/v/torch-spatiotemporal"></a>
    <img alt="PyPI - Python Version" src="https://img.shields.io/badge/python-3.8--3.11-blue">
    <!-- img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/torch-spatiotemporal" -->
    <img alt="Total downloads" src="https://static.pepy.tech/badge/torch-spatiotemporal">
    <a href='https://torch-spatiotemporal.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/torch-spatiotemporal/badge/?version=latest' alt='Documentation Status' /></a>
    </p>
    <p>
    🚀 <a href="https://torch-spatiotemporal.readthedocs.io/en/latest/usage/quickstart.html">Getting Started</a> - 📚 <a href="https://torch-spatiotemporal.readthedocs.io/en/latest/">Documentation</a> - 💻 <a href="https://torch-spatiotemporal.readthedocs.io/en/latest/notebooks/a_gentle_introduction_to_tsl.html">Introductory notebook</a>
    </p>
</div>

<p><img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo.svg" width="25px" align="center"/> <b>tsl</b> <em>(Torch Spatiotemporal)</em> is a library built to accelerate research on neural spatiotemporal data processing
methods, with a focus on Graph Neural Networks.</p>

<p>Built upon popular libraries such as <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/logos/pytorch.svg" width="20px" align="center"/> <a href="https://pytorch.org"><b>PyTorch</b></a>, <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/logos/pyg.svg" width="20px" align="center"/> <a href="https://pyg.org">PyG</a> (PyTorch Geometric), and <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/logos/lightning.svg" width="20px" align="center"/> <a href="https://www.pytorchlightning.ai/">PyTorch Lightning</a>, <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo.svg" width="25px" align="center"/> tsl provides a unified and user-friendly framework for efficient neural spatiotemporal data processing, that goes from data preprocessing to model prototyping.</p>

## Features

* **Create Custom Models and Datasets**&nbsp;&nbsp; Easily build your own custom models and datasets for spatiotemporal data analysis. Whether you're working with sensor networks, environmental data, or any other spatiotemporal domain, <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo.svg" width="25px" align="center"/> tsl's high-level APIs empower you to develop tailored solutions.

* **Access a Wealth of Existing Datasets and Models**&nbsp;&nbsp; Leverage a vast collection of datasets and models from the spatiotemporal data processing literature. Explore and benchmark against state-of-the-art baselines, and test your brand new model on widely used public datasets.

* **Handle Irregularities and Missing Data**&nbsp;&nbsp; Seamlessly manage irregularities in your spatiotemporal data streams, including missing data and variations in network structures. Ensure the robustness and reliability of your data processing pipelines.

* **Streamlined Preprocessing**&nbsp;&nbsp; Automate the preprocessing phase with <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo.svg" width="25px" align="center"/> tsl's methods for scaling, resampling and clustering time series. Spend less time on data preparation and focus on extracting meaningful patterns and insights.

* **Efficient Data Structures**&nbsp;&nbsp; Utilize <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo.svg" width="25px" align="center"/> tsl's straightforward data structures, seamlessly integrated with <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/logos/pytorch.svg" width="20px" align="center"/> PyTorch and <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/logos/pyg.svg" width="20px" align="center"/> PyG, to accelerate your workflows. Benefit from the flexibility and compatibility of these widely adopted libraries.

* **Scalability with PyTorch Lightning**&nbsp;&nbsp; Scale your computations effortlessly, from a single CPU to clusters of GPUs, with <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo.svg" width="25px" align="center"/> tsl's integration with <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/logos/lightning.svg" width="20px" align="center"/> PyTorch Lightning. Accelerate training and inference across various hardware configurations.

* **Modular Neural Layers**&nbsp;&nbsp; Build powerful and modular neural spatiotemporal models using <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo.svg" width="25px" align="center"/> tsl's collection of specialized layers. Create architectures with ease, leveraging the flexibility and extensibility of the library.

* **Reproducible Experiments**&nbsp;&nbsp; Ensure experiment reproducibility using the <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/logos/hydra.svg" width="25px" align="center"/> <a href="https://hydra.cc/">Hydra</a> framework, a standard in the field. Validate and compare results confidently, promoting rigorous research in spatiotemporal data mining.

## Getting Started

Before you start using <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo.svg" width="25px" align="center"/> tsl, please review the <a href="https://torch-spatiotemporal.readthedocs.io/en/latest/">documentation</a> to get an understanding of the library and its capabilities.

You can also explore the examples provided in the `examples` directory to see how train deep learning models working with spatiotemporal data.

## Installation

<img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo.svg" width="25px" align="center"/> tsl supports NumPy < 2, <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/logos/pytorch.svg" width="20px" align="center"/> PyTorch >= 1.13 and < 2.4, and <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/logos/pyg.svg" width="20px" align="center"/> PyG >= 2.4. `torch-scatter` is optional for sparse spatiotemporal attention; `torch-sparse` is optional for `torch_sparse.SparseTensor` adjacency matrices.

The recommended setup for local development is [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install torch-spatiotemporal --torch-backend=auto
# Optional extensions; select the matching wheel from the PyG guide:
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
# uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

On Windows PowerShell:

```powershell
uv venv --python 3.10
.venv\Scripts\Activate.ps1
uv pip install torch-spatiotemporal --torch-backend=auto
```

Install PyTorch first so uv can select the wheel that matches the machine. `UV_TORCH_BACKEND=auto` lets uv choose the most compatible PyTorch backend when using `uv pip`: CUDA on NVIDIA systems with a compatible driver, and CPU wheels otherwise. macOS uses PyTorch's macOS wheels; GPU acceleration there is handled by PyTorch/MPS rather than CUDA.

Install the PyG stack after PyTorch, using the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) when compiled extensions require a platform-specific wheel. Then install tsl.

For a regular user install from PyPI:

```bash
uv pip install torch-spatiotemporal --torch-backend=auto
```

For development tools and tests, install the dev tools into the same environment:

```bash
UV_TORCH_BACKEND=auto uv pip install torch
# Follow the PyG installation guide linked above.
uv pip install -e ".[dev]"
uv run pytest
```

If PyTorch is already installed and you do not need uv's automatic PyTorch backend detection, the project dependency groups are also available. Use `--inexact` so uv does not remove the externally installed PyTorch package:

```bash
uv sync --group dev --inexact
uv run pytest
```

The default test command uses the coverage settings in `pyproject.toml`. To skip slow tests explicitly:

```bash
uv run pytest -m "not slow"
```

### PyG compiled extensions

`torch-scatter` is optional and used by sparse attention; `torch-sparse` is optional and only needed for `torch_sparse.SparseTensor` connectivity. Their wheels depend on the exact PyTorch and CUDA build, so install either extension from the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) after installing PyTorch.

### Conda

The repository still includes `conda_env.yml` for users who prefer conda:

```bash
conda env create -f conda_env.yml
conda activate tsl
```

The conda environment remains CUDA-oriented by default. Remove the `nvidia` channel and `pytorch-cuda` dependency from `conda_env.yml` for a CPU-only conda environment.

## Tutorial

The best way to start using <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo.svg" width="25px" align="center"/> tsl is by following the tutorial notebook in `examples/notebooks/a_gentle_introduction_to_tsl.ipynb`.

## Documentation

Visit the [documentation](https://torch-spatiotemporal.readthedocs.io/en/latest/) to learn more about the library, including detailed API references, examples, and tutorials.

The documentation is hosted on [readthedocs](https://torch-spatiotemporal.readthedocs.io/en/latest/). For local access, you can build it from the `docs` directory.

## Contributing

Contributions are welcome! For major changes or new features, please open an issue first to discuss your ideas. See the [Contributing guidelines](https://github.com/TorchSpatiotemporal/tsl/blob/dev/.github/CONTRIBUTING.md) for more details on how to get involved. Help us build a better <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo.svg" width="25px" align="center"/> tsl!

Thanks to all contributors! 🧡

<a href="https://github.com/TorchSpatiotemporal/tsl/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TorchSpatiotemporal/tsl" />
</a>

## Citing

If you use Torch Spatiotemporal for your research, please consider citing the library

```latex
@software{Cini_Torch_Spatiotemporal_2022,
    author = {Cini, Andrea and Marisca, Ivan},
    license = {MIT},
    month = {3},
    title = {{Torch Spatiotemporal}},
    url = {https://github.com/TorchSpatiotemporal/tsl},
    year = {2022}
}
```

By [Andrea Cini](https://andreacini.github.io/) and [Ivan Marisca](https://marshka.github.io/).

## License

This project is licensed under the terms of the MIT license. See the [LICENSE](https://github.com/TorchSpatiotemporal/tsl/blob/main/LICENSE) file for details.
