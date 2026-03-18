from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.common import JobStatus, ModelInfo
from app.schemas.predict import PredictRequest, PredictResponse
from app.schemas.train import TrainRequest, TrainSubmissionResponse
from app.services.job_service import job_service
from app.services.model_service import model_service

router = APIRouter(prefix="/v1", tags=["models"])


@router.post("/models/train", response_model=TrainSubmissionResponse)
def submit_training(request: TrainRequest) -> TrainSubmissionResponse:
    job = job_service.submit_training_job(request)
    return TrainSubmissionResponse(job_id=job.job_id, state=job.state, message=job.message or "")


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    try:
        return job_service.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}") from exc


@router.get("/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    return model_service.list_models()


@router.get("/models/{model_id}", response_model=ModelInfo)
def model_details(model_id: str) -> ModelInfo:
    try:
        return model_service.get_model_info(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/models/{model_id}/predict", response_model=PredictResponse)
def predict(model_id: str, request: PredictRequest) -> PredictResponse:
    try:
        predictions = model_service.predict(model_id=model_id, rows=request.rows)
        return PredictResponse(model_id=model_id, predictions=predictions)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
