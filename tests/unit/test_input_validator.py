import pandas as pd

from src.inference.input_validator import validate_payload


def test_validate_payload_missing_feature() -> None:
    frame = pd.DataFrame([{"a": 1, "b": 2}])
    result = validate_payload(frame, ["a", "c"])
    assert not result.is_valid
    assert any("Missing features" in error for error in result.errors)
