# tsl: a PyTorch library for spatiotemporal data processing


**tsl** *(Torch Spatiotemporal)* is a library built to accelerate research on neural spatiotemporal data processing 
methods, with a focus on Graph Neural Networks.

`tsl` is built on several libraries of the *Python* scientific computing ecosystem, with the final objective of providing a straightforward process that goes from data preprocessing to model prototyping.
In particular, `tsl` offers a wide range of utilities to develop neural networks in *PyTorch* for processing spatiotemporal data signals.

## Installation
`tsl` is compatible with Python 3.7. The simplest way to install the library is from pip:

```bash
pip install torch-spatiotemporal
```

We recommend using Anaconda and the provided environment configuration by running the command:

```bash
conda env create -f tsl_env.yml
```

To install `tsl` from source, run this in a terminal:

```bash
git clone https://github.com/TorchSpatiotemporal/tsl.git
cd tsl
python setup.py install  # Or 'pip install .'
```

## Tutorial

The best way to start using `tsl` is by following the tutorial notebook in `examples/notebooks/a_gentle_introduction_to_tsl.ipynb`.

## Documentation

The documentation is hosted on [readthedocs](https://torch-spatiotemporal.readthedocs.io/en/latest/). For local access, you can build it from the `docs` directory.

## Credits

[Andrea Cini](https://andreacini.github.io/), [Ivan Marisca](https://marshka.github.io/)
