# Running the examples

In this folder, we provide example scripts on how to use TSL for running
consistent and reproducible experiments on neural spatiotemporal forecasting
and imputation.

The experiment code and its configuration are handled by the [`Experiment`](https://torch-spatiotemporal.readthedocs.io/en/latest/modules/experiment.html)
object in `tsl.experiment`, which makes use of [Hydra](https://hydra.cc/) as a
configuration manager. Please refer to [Hydra's documentation](https://hydra.cc/docs/intro/)
for more usage information.

In both subfolders, you will find an example script and a `config` directory.
We now show an example of how to launch a traffic forecasting experiment (the
same procedure applies to the imputation experiment).

Running a forecasting experiment using one of the baseline models on a provided
benchmark dataset is as simple as that:

```bash
python forecasting/run_traffic_experiment.py model=dcrnn dataset=la logger=wandb
```

Under the hood, we are telling the program to run the traffic experiment using
the default config `config/traffic/default.yaml` instantiated with model config
`config/traffic/model/dcrnn.yaml`, dataset config `config/traffic/dataset/la.yaml`,
and logger config `config/traffic/logger/wandb.yaml`. In practice, any parameter
defined in the subordinate configs `dcrnn.yaml`, `la.yaml`, and `wandb.yaml`
config files overrides any same parameter in `default.yaml`.

By passing as argument `dataset=la` we run the forecasting
experiment on the [MetrLA](https://torch-spatiotemporal.readthedocs.io/en/latest/modules/datasets_in_tsl.html#tsl.datasets.MetrLA)
dataset. Other available dataset options are `bay`, `pems3`, `pems4`, `pems7`,
and `pems8`.

Check the code in the script file to see other possible options for the
configuration arguments!
