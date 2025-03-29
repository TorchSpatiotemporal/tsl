from setuptools import find_packages, setup

__version__ = '0.9.5'
URL = 'https://github.com/TorchSpatiotemporal/tsl'

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    'einops',
    'numpy>1.20.3,<2',
    'pandas>=1.4',
    'pytorch_lightning>=1.8',
    'PyYAML',
    'scikit_learn',
    'scipy',
    'tables',
    'torchmetrics>=0.7',
    'tqdm',
]

plot_requires = [
    'matplotlib',
    'mpld3',
]

experiment_requires = [
    'hydra-core',
    'omegaconf',
]

full_install_requires = plot_requires + experiment_requires + [
    'holidays',
    'neptune-client>=0.14',
]

doc_requires = full_install_requires + [
    'docutils',
    'sphinx==7',
    'sphinx-design',
    'sphinx-copybutton',
    'sphinxext-opengraph',
    'sphinx-hoverxref',
    'myst-nb',
    'furo==2024.04.27',
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
setup(
    name='torch_spatiotemporal',
    version=__version__,
    description='A PyTorch library for spatiotemporal data processing',
    author='Andrea Cini, Ivan Marisca',
    author_email='andrea.cini@usi.ch, ivan.marisca@usi.ch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    download_url=f'{URL}/archive/v{__version__}.tar.gz',
    license="MIT",
    classifiers=classifiers,
    keywords=[
        'pytorch', 'pytorch-geometric', 'geometric-deep-learning',
        'graph-neural-networks', 'temporal-graph-networks',
        'spatiotemporal-graph-neural-networks', 'spatiotemporal-processing',
        'neural-spatiotemporal-forecasting', 'time-series-analysis',
        'forecasting'
    ],
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require={
        'experiment': experiment_requires,
        'full': full_install_requires,
        'doc': doc_requires,
    },
    packages=find_packages(exclude=['examples*']),
)
