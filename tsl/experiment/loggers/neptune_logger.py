import os
from typing import Optional, Union, List, Mapping

import numpy as np
import pandas as pd
from einops import rearrange
from pytorch_lightning.loggers import NeptuneLogger as LightningNeptuneLogger
from torch import Tensor

from tsl.utils.io import save_figure
from tsl.utils.python_utils import ensure_list


class NeptuneLogger(LightningNeptuneLogger):
    """Extensions of PyTorch Lightning
    :class:`~pytorch_lightning.loggers.NeptuneLogger` with useful logging
    functionalities.

    Args:
        api_key (str, optional): Neptune API token, found on https://neptune.ai
            upon registration. Read: `how to find and set Neptune API token
            <https://docs.neptune.ai/administration/security-and-privacy/how-to-find-and-set-neptune-api-token>`_.
            It is recommended to keep it in the :obj:`NEPTUNE_API_TOKEN`
            environment variable, then you can drop :attr:`api_key=None`.
            (default: :obj:`None`)
        project_name (str, optional): Name of a project in a form of
            "my_workspace/my_project". If :obj:`None`, the value of
            `NEPTUNE_PROJECT` environment variable will be taken.
            You need to create the project in https://neptune.ai first.
            (default: :obj:`None`)
        experiment_name (str, optional): Editable name of the run.
            Run name appears in the "all metadata/sys" section in Neptune UI.
            (default: :obj:`None`)
        tags (list, optional): List of tags of the run.
            (default: :obj:`None`)
        params (Mapping, optional): Mapping of the run's parameters (are logged
            as :obj:`"parameters"` on Neptune).
            (default: :obj:`None`)
        debug (bool): If :obj:`True`, then do not log online (i.e., log in
            :obj:`"debug"` mode). Otherwise log online in :obj:`"async"` mode.
            (default: :obj:`False`)
        prefix (str, optional): Root namespace for all metadata logging.
            (default: :obj:`"logs"`)
        upload_stdout (bool): If :obj:`True`, then log also :obj:`stdout` on
            Neptune.
            (default: :obj:`False`)
        **kwargs: Additional parameters for
            :class:`~pytorch_lightning.loggers.NeptuneLogger`.
    """

    def __init__(self, api_key: Optional[str] = None,
                 project_name: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 tags: Optional[Union[str, List]] = None,
                 params: Optional[Mapping] = None,
                 debug: bool = False,
                 prefix: Optional[str] = 'logs',
                 upload_stdout: bool = False,
                 **kwargs):
        prefix = prefix or ""
        if tags is not None:
            kwargs['tags'] = ensure_list(tags)
        mode = 'debug' if debug else 'async'
        super(NeptuneLogger, self).__init__(
            api_key=api_key,
            project=project_name,
            name=experiment_name,
            log_model_checkpoints=False,
            prefix=prefix,
            capture_stdout=upload_stdout,
            mode=mode,
            **kwargs)
        if params is not None:
            self.run['parameters'] = params
        # self.log_hyperparams(params=params)

    def log_metric(self, metric_name: str,
                   metric_value: Union[Tensor, float, str],
                   step: Optional[int] = None):
        # todo log metric series at once
        self.run[f'logs/{metric_name}'].log(metric_value, step)

    def log_artifact(self, filename: str, artifact_name: Optional[str] = None,
                     delete_after: bool = False):
        if artifact_name is None:
            # './dir/file.ext' -> 'file.ext'
            artifact_name = os.path.basename(filename)
        if delete_after:
            self.run[f"artifacts/{artifact_name}"].upload(filename, wait=True)
            os.unlink(filename)
        else:
            self.run[f"artifacts/{artifact_name}"].upload(filename)

    def log_numpy(self, array, name="array"):
        if not name.endswith('.npy'):
            name += '.npy'
        np.save(name, array)
        self.log_artifact(name, delete_after=True)

    def log_pred_df(self, name, idx, y, yhat, label_y='true',
                    label_yhat='pred'):
        """
        Log a csv containing predictions and true values. Only works for univariate timeseries.

        :param name: name of the file
        :param idx: dataframe idx
        :param y: true values
        :param yhat: predictions
        :param label_y: true values
        :param label_yhat: predictions
        :return:
        """
        y = rearrange(y, 'b ... -> b (...)')
        yhat = rearrange(yhat, 'b ... -> b (...)')
        if isinstance(label_y, str):
            label_y = [f'{label_y}_{i}' for i in range(y.shape[1])]
        if isinstance(label_yhat, str):
            label_yhat = [f'{label_yhat}_{i}' for i in range(yhat.shape[1])]
        df = pd.DataFrame(data=np.concatenate([y, yhat], axis=-1),
                          columns=label_y + label_yhat,
                          index=idx)
        df.to_csv(name, index=True, index_label='datetime')
        self.experiment.log_artifact(name)
        os.remove(name)

    def log_dataframe(self, df: pd.DataFrame, name: str = 'dataframe'):
        """Log a dataframe as csv.

        Args:
            df (DataFrame): The dataframe to be logged.
            name (str): The name of the file. (default: :obj:`'dataframe'`)
        """
        if not name.endswith('.csv'):
            name += '.csv'
        df.to_csv(name, index=True, index_label='index')
        self.log_artifact(name, delete_after=True)

    def log_figure(self, fig, name: str = 'figure'):
        """Log a matplotlib figure as html.

        Args:
            fig (Figure): The matplotlib figure to be logged
            name (str): The name of the file. (default: :obj:`'figure'`)
        """
        if not name.endswith('.html'):
            name += '.html'
        save_figure(fig, name)
        self.log_artifact(name, delete_after=True)
