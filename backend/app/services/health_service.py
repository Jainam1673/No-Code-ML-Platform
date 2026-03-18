from __future__ import annotations

from dataclasses import dataclass

from redis import Redis
from sqlalchemy import text

from app.core.config import settings
from app.db.session import get_session


@dataclass(frozen=True, slots=True)
class DependencyStatus:
    database: bool
    redis: bool

    @property
    def overall(self) -> str:
        return "ok" if self.database and self.redis else "degraded"


class HealthService:
    """Encapsulates health and readiness dependency checks."""

    def check_dependencies(self) -> DependencyStatus:
        return DependencyStatus(database=self._check_database(), redis=self._check_redis())

    @staticmethod
    def _check_database() -> bool:
        try:
            with get_session() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _check_redis() -> bool:
        try:
            return bool(
                Redis.from_url(
                    settings.redis_url,
                    socket_connect_timeout=0.3,
                    socket_timeout=0.3,
                    retry_on_timeout=False,
                ).ping()
            )
        except Exception:  # noqa: BLE001
            return False

    def readiness_payload(self) -> dict[str, object]:
        status = self.check_dependencies()
        return {
            "status": status.overall,
            "service": settings.app_name,
            "version": settings.app_version,
            "dependencies": {
                "database": "ok" if status.database else "down",
                "redis": "ok" if status.redis else "down",
            },
        }


health_service = HealthService()
