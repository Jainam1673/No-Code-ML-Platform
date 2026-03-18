from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings
from app.services.capability_service import capability_service

router = APIRouter(prefix="/v1/system", tags=["system"])


@router.get("/capabilities")
def capabilities() -> dict[str, object]:
    caps = capability_service.detect()
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "runtime": {
            "python_version": caps.python_version,
            "os": caps.os_name,
            "arch": caps.arch,
        },
        "accelerators": {
            "hint": caps.accelerator_hint,
            "cuda_available": caps.cuda_available,
            "nvidia_smi_available": caps.nvidia_smi_available,
            "gpu_count": caps.gpu_count,
        },
        "execution": {
            "backend": settings.execution_backend,
            "celery_queue": settings.celery_queue,
            "max_training_workers": settings.max_training_workers,
        },
    }
