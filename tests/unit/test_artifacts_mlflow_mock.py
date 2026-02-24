import sys
from pathlib import Path
from types import SimpleNamespace

from src.utils.artifacts import log_to_mlflow


class _RunCtx:
    def __enter__(self):
        return SimpleNamespace(info=SimpleNamespace(run_id="abc123"))

    def __exit__(self, exc_type, exc, tb):
        return False


def test_log_to_mlflow_with_mocked_module(monkeypatch, tmp_path: Path) -> None:
    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda *_: None,
        set_experiment=lambda *_: None,
        start_run=lambda **_: _RunCtx(),
        log_params=lambda *_: None,
        log_metric=lambda *_: None,
        log_artifacts=lambda *_, **__: None,
        sklearn=SimpleNamespace(log_model=lambda *_, **__: None),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    run_id = log_to_mlflow(
        "file:./x", "exp", "run", {"a": 1}, {"m": 0.9}, tmp_path, object()
    )
    assert run_id == "abc123"
