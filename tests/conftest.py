"""Shared pytest configuration for the test suite."""

from importlib.util import find_spec

import pytest

OPTIONAL_TEST_DEPENDENCIES = {
    'holidays': 'holidays',
    'torch_sparse': 'torch-sparse',
    'torch_scatter': 'torch-scatter',
}


def pytest_addoption(parser):
    parser.addoption(
        "--run-datasets",
        action="store_true",
        default=False,
        help="run tests that download real datasets (marked dataset_download)",
    )


def pytest_collection_modifyitems(config, items):
    skip_download = None
    if not config.getoption("--run-datasets"):
        skip_download = pytest.mark.skip(
            reason="needs --run-datasets to download/load real datasets"
        )

    unavailable_markers = {
        marker: package
        for marker, package in OPTIONAL_TEST_DEPENDENCIES.items()
        if find_spec(marker) is None
    }
    for item in items:
        if skip_download is not None and "dataset_download" in item.keywords:
            item.add_marker(skip_download)
        for marker, package in unavailable_markers.items():
            if marker in item.keywords:
                item.add_marker(
                    pytest.mark.skip(reason=f"requires optional dependency {package}")
                )
