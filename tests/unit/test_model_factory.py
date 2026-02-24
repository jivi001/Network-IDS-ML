import pytest

from src.models.factory import build_model


def test_random_forest_factory() -> None:
    model = build_model("random_forest", {"n_estimators": 5}, random_state=42)
    assert model.__class__.__name__ == "RandomForestClassifier"


def test_other_factory_models() -> None:
    assert (
        build_model("svm", {"C": 1.0, "kernel": "linear"}, 42).__class__.__name__
        == "SVC"
    )
    assert (
        build_model("isolation_forest", {"n_estimators": 10}, 42).__class__.__name__
        == "IsolationForest"
    )


def test_factory_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        build_model("xgboost", {}, 42)
