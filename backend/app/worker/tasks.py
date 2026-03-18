from __future__ import annotations

from app.schemas.train import TrainRequest
from app.services.job_service import job_service
from app.services.model_service import model_service
from app.worker.celery_app import celery_app


@celery_app.task(
    bind=True,
    name="app.worker.train_model_task",
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=5,
)
def train_model_task(self, job_id: str, request_payload: dict) -> str:
    request = TrainRequest(**request_payload)
    job_service.mark_running(job_id)

    try:
        model_info = model_service.train(request)
        job_service.mark_succeeded(job_id=job_id, model_id=model_info.model_id)
        return model_info.model_id
    except Exception as exc:  # noqa: BLE001
        job_service.mark_failed(job_id=job_id, error=str(exc))
        raise
