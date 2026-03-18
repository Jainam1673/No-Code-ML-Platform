from __future__ import annotations

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    dataset_path: str = Field(description="Path to training CSV file")
    target_column: str = Field(description="Target column in CSV")
    presets: str = Field(default="medium_quality", description="AutoGluon fit presets")
    time_limit: int | None = Field(default=None, ge=1, description="Optional training limit in seconds")


class TrainSubmissionResponse(BaseModel):
    job_id: str
    state: str
    message: str
