from __future__ import annotations

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.routes.health import router as health_router
from app.api.routes.models import router as models_router
from app.api.routes.system import router as system_router
from app.core.config import ensure_paths, settings
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware
from app.db.base import Base
import app.db.models  # noqa: F401
from app.db.session import engine


def create_app() -> FastAPI:
    configure_logging()
    ensure_paths()
    if settings.run_migrations_on_startup:
        from app.db.migrations import upgrade_head

        upgrade_head()

    if settings.auto_create_tables:
        Base.metadata.create_all(bind=engine)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(RequestContextMiddleware)

    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(system_router)

    if settings.metrics_enabled:
        Instrumentator().instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")

    return app


app = create_app()
