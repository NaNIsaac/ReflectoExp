"""Evaluation scripts read the run their own main.py trained (ReflectoExp#3).

The scripts run training or plotting under ``if __name__ == "__main__"`` and need data on disk, so
this checks the one line that picks the folder: it must come from the experiment's CONFIG, the same
place main.py takes it from, and not from a path hard-coded to another experiment.
"""

import importlib.util
from pathlib import Path

import pytest

RUNS = Path(__file__).resolve().parents[1] / "runs"
EXPERIMENTS = ["exp02_one_layer", "exp04_one_genx"]
SCRIPTS = ["evaluate.py", "evaluate_with_real_data.py"]


@pytest.mark.parametrize("exp", EXPERIMENTS)
@pytest.mark.parametrize("script", SCRIPTS)
def test_evaluate_script_takes_the_run_folder_from_config(exp, script):
    text = (RUNS / exp / script).read_text(encoding="utf-8")
    assert 'CONFIG["base_dir"] / CONFIG["exp_name"]' in text
    # The folder another experiment trained into, which these scripts used to read.
    assert 'r"XRR_AI\data\one_layer\run"' not in text


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_main_trains_into_the_same_folder(exp):
    text = (RUNS / exp / "main.py").read_text(encoding="utf-8")
    assert 'CONFIG["base_dir"] / CONFIG["exp_name"]' in text


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_config_defines_the_run_folder(exp, monkeypatch, tmp_path):
    # exp02 builds base_dir from $DATA; exp04 writes a fixed path. Either way it must be a Path.
    monkeypatch.setenv("DATA", str(tmp_path))
    spec = importlib.util.spec_from_file_location(f"{exp}_config", RUNS / exp / "config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run_dir = module.CONFIG["base_dir"] / module.CONFIG["exp_name"]
    assert run_dir.parent == module.CONFIG["base_dir"]
    assert run_dir.name == module.CONFIG["exp_name"]
