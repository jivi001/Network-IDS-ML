from src.training.training import _build_search


def test_build_search_passthrough_when_disabled() -> None:
    dummy_pipeline = object()
    config = {"training": {"cv_folds": 2, "search": {"enabled": False}}}
    out = _build_search(dummy_pipeline, config, random_state=42)
    assert out is dummy_pipeline
