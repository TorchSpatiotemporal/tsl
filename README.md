<div align="center">
    <br><br>
    <img alt="Torch Spatiotemporal" src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo_text.svg" width="85%"/>
    <h3>Neural spatiotemporal forecasting with PyTorch</h3>
    <hr>
    <p>
    <img alt="PyPI" src="https://img.shields.io/pypi/v/torch-spatiotemporal">
    <img alt="PyPI - Python Version" src="https://img.shields.io/badge/python-%3E%3D3.8-blue">
    <!-- img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/torch-spatiotemporal" -->
    <img alt="Total downloads" src="https://static.pepy.tech/badge/torch-spatiotemporal">
    <a href='https://torch-spatiotemporal.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/torch-spatiotemporal/badge/?version=latest' alt='Documentation Status' />
    </a>
    </p>
    <p>
    🚀 <a href="https://torch-spatiotemporal.readthedocs.io/en/latest/usage/quickstart.html">Getting Started</a> - 📚 <a href="https://torch-spatiotemporal.readthedocs.io/en/latest/">Documentation</a> - 💻 <a href="https://torch-spatiotemporal.readthedocs.io/en/latest/notebooks/a_gentle_introduction_to_tsl.html">Introductory notebook</a>
    </p>
</div>

<p><img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo.svg" width="25px" align="center"/> <b>tsl</b> <em>(Torch Spatiotemporal)</em> is a library built to accelerate research on neural spatiotemporal data processing
methods, with a focus on Graph Neural Networks.</p>

<p><img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo.svg" width="25px" align="center"/> tsl is built on several libraries of the <b>Python</b> scientific computing ecosystem, with the final objective of providing a straightforward process that goes from data preprocessing to model prototyping.
In particular, <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo.svg" width="25px" align="center"/> tsl offers a wide range of utilities to develop neural networks in <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/logos/pytorch.svg" width="20px" align="center"/> <a href="https://pytorch.org"><b>PyTorch</b></a> for processing spatiotemporal data signals.</p>

## Getting Started

Before you start using <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo.svg" width="25px" align="center"/> tsl, please review the <a href="https://torch-spatiotemporal.readthedocs.io/en/latest/">documentation</a> to get an understanding of the library and its capabilities.

You can also explore the examples provided in the `examples` directory to see how train deep learning models working with spatiotemporal data.

## Installation

Before installing <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo.svg" width="25px" align="center"/> tsl, make sure you have installed <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/logos/pytorch.svg" width="20px" align="center"/> <a href="https://pytorch.org">PyTorch</a> (>=1.9.0) and <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/logos/pyg.svg" width="20px" align="center"/> <a href="https://pyg.org">PyG</a> (>=2.0.3) in your virtual environment (see [PyG installation guidelines](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)). <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo.svg" width="25px" align="center"/> tsl is available for Python>=3.8. We recommend installation from github to be up-to-date with the latest version:

```bash
pip install git+https://github.com/TorchSpatiotemporal/tsl.git
```

Alternatively, you can install the library from the pypi repository:

```bash
pip install torch-spatiotemporal
```

To avoid dependencies issues, we recommend using [Anaconda](https://www.anaconda.com/) and the provided environment configuration by running the command:

```bash
conda env create -f conda_env.yml
```

## Tutorial

The best way to start using <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo.svg" width="25px" align="center"/> tsl is by following the tutorial notebook in `examples/notebooks/a_gentle_introduction_to_tsl.ipynb`.

## Documentation

The documentation is hosted on [readthedocs](https://torch-spatiotemporal.readthedocs.io/en/latest/). For local access, you can build it from the `docs` directory.

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

Thanks to all contributors! Check the [Contributing guidelines](https://github.com/TorchSpatiotemporal/tsl/blob/dev/.github/CONTRIBUTING.md) and help us build a better <img src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo.svg" width="25px" align="center"/> tsl.

<a href="https://github.com/TorchSpatiotemporal/tsl/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TorchSpatiotemporal/tsl" />
</a>
