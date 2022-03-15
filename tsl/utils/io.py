import os
import pickle
import zipfile
from typing import Any

from tsl import logger


def extract_zip(path: str, folder: str, log: bool = True):
    r"""Extracts a zip archive to a specific folder.

    Args:
        path (string): The path to the zip archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not log anything.
            (default: :obj:`True`)
    """
    if log:
        logger.info(f"Extracting {path}")
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)


def save_pickle(obj: Any, filename: str) -> str:
    """Save obj to path as pickle.

    Args:
        obj: Object to be saved.
        filename (string): Where to save the file.

    Returns:
        path (string): The absolute path to the saved pickle
    """
    abspath = os.path.abspath(filename)
    directory = os.path.dirname(abspath)
    os.makedirs(directory, exist_ok=True)
    with open(abspath, 'wb') as fp:
        pickle.dump(obj, fp)
    return abspath


def load_pickle(filename: str) -> Any:
    """Load object from pickle filename.

    Args:
        filename (string): The absolute path to the saved pickle.

    Returns:
        data (any): The loaded object.
    """
    with open(filename, 'rb') as fp:
        data = pickle.load(fp)
    return data


def save_figure(fig, filename: str, as_html=False, as_pickle=False):
    if filename.endswith('html'):
        as_html = True
        filename = filename[:-5]
    elif filename.endswith('pkl'):
        as_pickle = True
        filename = filename[:-4]
    if not (as_html or as_pickle):
        as_html = False  # save as html if nothing is specified
    if as_html:
        import mpld3
        with open(filename + '.html', 'w') as fp:
            mpld3.save_html(fig, fp)
    if as_pickle:
        import pickle
        with open(filename + '.pkl', 'wb') as fp:
            pickle.dump(fig, fp)