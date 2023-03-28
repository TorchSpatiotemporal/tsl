# Building Documentation

To build the documentation from tsl root directory:

1. Install PyTorch and PyG via `pip install -r docs/requirements.txt`.
2. Install tsl and [Sphinx](https://www.sphinx-doc.org/en/master/) requirements
   via `pip install .[doc]`
3. Generate the documentation file via:

```bash
cd docs
make html
```

The documentation is now available to view by opening
`docs/build/html/index.html`.
