from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import cast
from uuid import uuid4

from app.core.config import settings
from app.db.models import JobRecordORM
from app.db.session import get_session
from app.schemas.common import JobState, JobStatus
from app.schemas.train import TrainRequest


@dataclass(slots=True)
class JobRecord:
    job_id: str
    state: JobState = "queued"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    model_id: str | None = None
    message: str | None = None
    error: str | None = None

    def to_status(self) -> JobStatus:
        return JobStatus(
            job_id=self.job_id,
            state=self.state,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            model_id=self.model_id,
            message=self.message,
            error=self.error,
        )

    @classmethod
    def from_orm(cls, db_job: JobRecordORM) -> "JobRecord":
        return cls(
            job_id=db_job.job_id,
            state=cast(JobState, db_job.state),
            created_at=db_job.created_at,
            started_at=db_job.started_at,
            completed_at=db_job.completed_at,
            model_id=db_job.model_id,
            message=db_job.message,
            error=db_job.error,
        )


class JobService:
    """Coordinates asynchronous training jobs and lifecycle persistence."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=settings.max_training_workers)

    def submit_training_job(self, request: TrainRequest) -> JobStatus:
        job_id = str(uuid4())
        job = JobRecord(job_id=job_id, message="Training job queued")
        with self._lock:
            self._jobs[job_id] = job
        self._upsert_db(job)

        if settings.execution_backend == "celery":
            from app.worker.tasks import train_model_task

            try:
                train_model_task.delay(job_id=job_id, request_payload=request.model_dump())
            except Exception:  # noqa: BLE001
                job.message = "Queue unavailable, running job in local worker"
                self._upsert_db(job)
                self._executor.submit(self._execute_job_local, job_id, request)
        else:
            self._executor.submit(self._execute_job_local, job_id, request)

        return self.get_job(job_id)

    def _execute_job_local(self, job_id: str, request: TrainRequest) -> None:
        from app.services.model_service import model_service

        self.mark_running(job_id)

        try:
            model_info = model_service.train(request)
            self.mark_succeeded(job_id=job_id, model_id=model_info.model_id)
        except Exception as exc:  # noqa: BLE001
            self.mark_failed(job_id=job_id, error=str(exc))

    def mark_running(self, job_id: str) -> None:
        job = self._get_or_create_job(job_id)
        job.state = "running"
        job.started_at = datetime.now(timezone.utc)
        job.message = "Training started"
        self._upsert_db(job)

    def mark_succeeded(self, job_id: str, model_id: str) -> None:
        job = self._get_or_create_job(job_id)
        job.state = "succeeded"
        job.model_id = model_id
        job.completed_at = datetime.now(timezone.utc)
        job.message = "Training completed"
        self._upsert_db(job)

    def mark_failed(self, job_id: str, error: str) -> None:
        job = self._get_or_create_job(job_id)
        job.state = "failed"
        job.error = error
        job.completed_at = datetime.now(timezone.utc)
        job.message = "Training failed"
        self._upsert_db(job)

    def get_job(self, job_id: str) -> JobStatus:
        with get_session() as session:
            db_job = session.get(JobRecordORM, job_id)
            if db_job is not None:
                return JobRecord.from_orm(db_job).to_status()

        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id].to_status()

    def _get_or_create_job(self, job_id: str) -> JobRecord:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return job

        with get_session() as session:
            db_job = session.get(JobRecordORM, job_id)
            if db_job is None:
                raise KeyError(job_id)

            hydrated = JobRecord.from_orm(db_job)

        with self._lock:
            self._jobs[job_id] = hydrated
            return hydrated

    @staticmethod
    def _upsert_db(job: JobRecord) -> None:
        with get_session() as session:
            existing = session.get(JobRecordORM, job.job_id)
            if existing is None:
                existing = JobRecordORM(job_id=job.job_id, state=job.state, created_at=job.created_at)
                session.add(existing)

            existing.state = job.state
            existing.created_at = job.created_at
            existing.started_at = job.started_at
            existing.completed_at = job.completed_at
            existing.model_id = job.model_id
            existing.message = job.message
            existing.error = job.error
            session.commit()


job_service = JobService()
