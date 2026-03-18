from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "No-Code ML Backend"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    auto_create_tables: bool = False
    run_migrations_on_startup: bool = False
    metrics_enabled: bool = True
    max_training_workers: int = 2
    execution_backend: Literal["threadpool", "celery"] = "celery"
    database_url: str = "sqlite:///./artifacts/platform.db"
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"
    celery_queue: str = "training"
    celery_task_soft_time_limit_seconds: int = 3600
    celery_task_time_limit_seconds: int = 3900
    celery_worker_prefetch_multiplier: int = 1
    artifacts_root: Path = Path("artifacts")
    models_root: Path = Path("artifacts/models")
    registry_root: Path = Path("artifacts/registry")

    model_config = SettingsConfigDict(env_prefix="NOCODEML_", env_file=".env", extra="ignore")


settings = Settings()


def ensure_paths() -> None:
    settings.artifacts_root.mkdir(parents=True, exist_ok=True)
    settings.models_root.mkdir(parents=True, exist_ok=True)
    settings.registry_root.mkdir(parents=True, exist_ok=True)
