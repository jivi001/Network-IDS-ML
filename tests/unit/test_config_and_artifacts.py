import json
from pathlib import Path

import pytest
from sklearn.dummy import DummyClassifier

from src.utils.artifacts import log_to_mlflow, persist_training_bundle
from src.utils.config import load_yaml_config


def test_load_yaml_config(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("a: 1\n", encoding="utf-8")
    config = load_yaml_config(config_file)
    assert config["a"] == 1


def test_persist_training_bundle(tmp_path: Path) -> None:
    model = DummyClassifier(strategy="most_frequent")
    output_dir = tmp_path / "bundle"
    metadata = {"metrics": {"f1": 0.9}, "features": ["a"], "config": {"x": 1}}
    persist_training_bundle(output_dir, model, metadata)
    assert (output_dir / "model.joblib").exists()
    assert (
        json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))["f1"]
        == 0.9
    )


def test_log_to_mlflow_without_dependency_raises() -> None:
    with pytest.raises(RuntimeError):
        log_to_mlflow("file:./x", "e", "r", {}, {}, Path("."), model=object())
