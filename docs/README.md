# Building Documentation

To build the documentation from the tsl root directory:

1. Install the project and its documentation dependencies with
   [uv](https://docs.astral.sh/uv/):

```bash
uv venv && uv pip install -e ".[doc]" --torch-backend=auto
```

2. Generate the documentation via:

```bash
cd docs
make html
```

The documentation is now available to view by opening
`docs/build/html/index.html`.
