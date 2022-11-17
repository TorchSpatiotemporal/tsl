<div align="center">
    <img src="docs/source/_static/img/tsl_logo_text.svg" width="60%"/>
</div>

**tsl** *(Torch Spatiotemporal)* is a library built to accelerate research on neural spatiotemporal data processing 
methods, with a focus on Graph Neural Networks.

`tsl` is built on several libraries of the *Python* scientific computing ecosystem, with the final objective of providing a straightforward process that goes from data preprocessing to model prototyping.
In particular, `tsl` offers a wide range of utilities to develop neural networks in *PyTorch* for processing spatiotemporal data signals.

## Installation

`tsl` is compatible with Python>=3.7. We recommend installation from source to be up-to-date with the latest version:

```bash
git clone https://github.com/TorchSpatiotemporal/tsl.git
cd tsl
python setup.py install  # Or 'pip install .'
```

To solve all dependencies, we recommend using Anaconda and the provided environment configuration by running the command:

```bash
conda env create -f tsl_env.yml
```

Alternatively, you can install the library from pip:

```bash
pip install torch-spatiotemporal
```

Please refer to [PyG installation guidelines](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for installation of PyG ecosystem without conda.

## Tutorial

The best way to start using `tsl` is by following the tutorial notebook in `examples/notebooks/a_gentle_introduction_to_tsl.ipynb`.

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
