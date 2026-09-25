import os
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tsl.experiment import Experiment


def test_experiment_is_available_with_mandatory_hydra_dependencies(tmp_path):
    experiment = Experiment(lambda cfg: cfg, config_path=str(tmp_path))

    assert experiment.config_path == str(Path(tmp_path))
    assert OmegaConf.has_resolver('prod')


def test_experiment_uses_hydra_13_working_directory_defaults(tmp_path, monkeypatch):
    output_dir = tmp_path / 'output'
    (tmp_path / 'config.yaml').write_text(
        f'hydra:\n  run:\n    dir: {output_dir}\n', encoding='utf-8'
    )
    monkeypatch.setattr(sys, 'argv', ['test_experiment'])

    experiment = Experiment(
        lambda cfg: os.getcwd(), config_path=str(tmp_path), config_name='config'
    )

    assert experiment.run() == os.getcwd()
    assert Path(experiment.run_dir) == output_dir
    assert (output_dir / 'config.yaml').is_file()


@pytest.mark.parametrize(
    ('interpolation', 'expected'),
    [
        ('${neg:-4}', 4),
        ('${prod:1,2,3,4}', 24),
        ('${ternary:true,yes,no}', 'yes'),
        ('${lower:HELLO}', 'hello'),
        ('${upper:hello}', 'HELLO'),
        ('${title:hello world}', 'Hello World'),
        ('${capitalize:hello WORLD}', 'Hello world'),
    ],
)
def test_custom_resolvers(interpolation, expected):
    cfg = OmegaConf.create({'value': interpolation})

    assert cfg.value == expected
