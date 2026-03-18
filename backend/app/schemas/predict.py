from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(description="Rows to infer, each row is a feature dictionary")


class PredictResponse(BaseModel):
    model_id: str
    predictions: list[Any]
