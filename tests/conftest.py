"""Shared pytest configuration for the test suite."""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-datasets",
        action="store_true",
        default=False,
        help="run tests that download real datasets (marked dataset_download)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-datasets"):
        return

    skip_download = pytest.mark.skip(
        reason="needs --run-datasets to download/load real datasets")
    for item in items:
        if "dataset_download" in item.keywords:
            item.add_marker(skip_download)
