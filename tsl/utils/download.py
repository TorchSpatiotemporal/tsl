import os
import urllib.request
from typing import Optional

from tqdm import tqdm

from tsl import logger


class DownloadProgressBar(tqdm):
    # From https://stackoverflow.com/a/53877507
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, folder: str, filename: Optional[str] = None,
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
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=path, reporthook=t.update_to)
    return path
