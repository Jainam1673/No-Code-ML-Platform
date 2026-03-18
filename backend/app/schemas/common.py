from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


JobState = Literal["queued", "running", "succeeded", "failed"]


class JobStatus(BaseModel):
    job_id: str
    state: JobState
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    model_id: str | None = None
    message: str | None = None
    error: str | None = None


class ModelInfo(BaseModel):
    model_id: str
    label_column: str
    problem_type: str
    eval_metric: str
    created_at: datetime
    model_path: str
    source_dataset: str = Field(description="CSV source used for training")
