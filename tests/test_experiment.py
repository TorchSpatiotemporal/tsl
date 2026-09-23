from pathlib import Path

from omegaconf import OmegaConf

from tsl.experiment import Experiment


def test_experiment_is_available_with_mandatory_hydra_dependencies(tmp_path):
    experiment = Experiment(lambda cfg: cfg, config_path=str(tmp_path))

    assert experiment.config_path == str(Path(tmp_path))
    assert OmegaConf.has_resolver('prod')
