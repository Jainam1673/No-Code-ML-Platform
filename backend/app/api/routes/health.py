from __future__ import annotations

from fastapi import APIRouter

from app.services.health_service import health_service

router = APIRouter(tags=["health"])


@router.get("/livez")
def livez() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readyz() -> dict[str, object]:
    return health_service.readiness_payload()


@router.get("/health")
def health() -> dict[str, object]:
    return health_service.readiness_payload()
