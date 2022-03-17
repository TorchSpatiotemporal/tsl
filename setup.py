from setuptools import find_packages, setup

__version__ = '0.1.0'
URL = 'https://github.com/TorchSpatiotemporal/tsl'

with open("README.md", "r") as fh:
    long_description = fh.read()


install_requires = [
    'einops',
    'numpy',
    'pandas',
    'pytorch_lightning>=1.5',
    'PyYAML',
    'scikit_learn',
    'scipy',
    'tables',
    'test_tube',
    'torch>=1.9',
    'torch_geometric>=2.0',
    'torchmetrics>=0.7',
    'tqdm',
]

full_install_requires = [
    'matplotlib',
    'mpld3',
    'neptune-client>=0.14',
    'pytorch_fast_transformers'
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
    license="MIT",
    keywords=[
        'pytorch',
        'pytorch-geometric',
        'geometric-deep-learning',
        'graph-neural-networks',
        'graph-convolutional-networks',
        'temporal-graph-networks',
        'spatiotemporal-processing',
    ],
    python_requires='>=3.7',
    install_requires=install_requires,
    extras_require={
        'full': full_install_requires,
    },
    packages=find_packages(exclude=['examples*']),
)
