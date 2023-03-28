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
python forecasting/run_traffic_experiment.py config=dcrnn dataset.name=la
```

Under the hood, we are telling the program to run the traffic experiment using
the `config/traffic/dcrnn.yaml` config, which is a particular instance of the
`config/traffic/default.yaml` config. In practice, any parameter not defined in
the primary `dcrnn.yaml` config file is inherited from the fallback config
`default.yaml`.

Note that in this latter config, the rows
```yaml
dataset:
  name: ???
```
indicate that the name of the dataset is required in order to run the
experiment. By passing as argument `dataset.name=la` we run the forecasting
experiment on the [MetrLA](https://torch-spatiotemporal.readthedocs.io/en/latest/modules/datasets_in_tsl.html#tsl.datasets.MetrLA)
dataset.

Check the code in the script file to see other possible options for the
configuration arguments!
