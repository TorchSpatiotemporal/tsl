import os
import pickle
import tarfile
import urllib.request
import zipfile
from typing import Any, Optional

from tqdm import tqdm

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


def extract_tar(path: str, folder: str, log: bool = True):
    r"""Extracts a tar (or tar.gz) archive to a specific folder.

    Args:
        path (string): The path to the tar(gz) archive.
        folder (string): The destination folder.
        log (bool, optional): If :obj:`False`, will not log anything.
            (default: :obj:`True`)
    """
    if log:
        logger.info(f"Extracting {path}")
    with tarfile.open(path, 'r') as tar:
        for member in tqdm(iterable=tar.getmembers(),
                           total=len(tar.getmembers())):
            tar.extract(member=member, path=folder)


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
    if filename.endswith('.html'):
        as_html = True
        filename = filename[:-5]
    elif filename.endswith('.pkl'):
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


class DownloadProgressBar(tqdm):
    # From https://stackoverflow.com/a/53877507
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str,
                 folder: str,
                 filename: Optional[str] = None,
                 log: bool = True):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        filename (string, optional): The filename. If :obj:`None`, inferred from
            url.
        log (bool, optional): If :obj:`False`, will not log anything.
            (default: :obj:`True`)
    """
    if filename is None:
        filename = url.rpartition('/')[2].split('?')[0]
    path = os.path.join(folder, filename)

    if os.path.exists(path):
        if log:
            logger.warning(f'Using existing file {filename}')
        return path

    if log:
        logger.info(f'Downloading {url}')

    os.makedirs(folder, exist_ok=True)

    # From https://stackoverflow.com/a/53877507
    with DownloadProgressBar(unit='B',
                             unit_scale=True,
                             miniters=1,
                             desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=path, reporthook=t.update_to)
    return path
