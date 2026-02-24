import pytest

pytest.importorskip("pydantic")

from pydantic import ValidationError

from src.api.schemas import ExplainRequest, PredictRequest


def test_predict_request_threshold_bounds() -> None:
    request = PredictRequest(records=[{"a": 1}], threshold_override=0.5)
    assert request.threshold_override == 0.5


def test_explain_request_validates_top_k() -> None:
    try:
        ExplainRequest(record={"a": 1}, top_k=0)
    except ValidationError:
        return
    assert False, "Expected validation error"
