<div align="center">
    <br><br>
    <img alt="Torch Spatiotemporal" src="https://raw.githubusercontent.com/TorchSpatiotemporal/tsl/main/docs/source/_static/img/tsl_logo_text.svg" width="85%"/>
    <h3>Neural spatiotemporal forecasting with PyTorch</h3>
    <hr>
    <p>
    <a href='https://pypi.org/project/torch-spatiotemporal/'><img alt="PyPI" src="https://img.shields.io/pypi/v/torch-spatiotemporal"></a>
    <img alt="PyPI - Python Version" src="https://img.shields.io/badge/python-%3E%3D3.8-blue">
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
