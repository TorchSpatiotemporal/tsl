# testing

Tests are collected in files, classes and functions, using the naming convention: `test_*.py`, `TestClass`, `test_func`

1. Install `pytest` in your tsl environment
```
conda activate tsl
conda install pytest
```

2. From the tsl root folder, invoke all tests
```
pytest --verbose
```

Markers are used to select or exclude some tests using the -m flag
- Exclude slow tests
```
pytest -v -m 'not slow'
```
- Run only integration tests
```
pytest -v -m integration
```
