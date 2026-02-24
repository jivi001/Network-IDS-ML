from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)
    threshold_override: float | None = Field(default=None, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    results: list[dict[str, Any]]


class ExplainRequest(BaseModel):
    record: dict[str, Any]
    top_k: int = Field(default=5, ge=1, le=20)
