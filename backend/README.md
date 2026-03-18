# Backend

Backend runtime for the platform, built around typed service boundaries and asynchronous ML training orchestration.

## Technology

- Python 3.12
- uv for dependency management and command execution
- FastAPI for API endpoints
- Celery + Redis for distributed training jobs
- SQLAlchemy + Alembic for metadata persistence and schema migrations
- AutoGluon Tabular as the training engine

## Backend Architecture

- app/api/routes: HTTP entrypoints
- app/services: business logic and orchestration
- app/db: models, sessions, migrations integration
- app/worker: Celery app and task execution
- app/core: config, middleware, logging

Artifact layout:

- artifacts/models/<model_id>
- artifacts/registry/<model_id>.json

## Local Setup (uv)

```bash
uv sync
cp .env.example .env
```

Run migrations:

```bash
uv run python main.py db-upgrade
```

## Run Modes

API server:

```bash
uv run python main.py serve
```

Worker:

```bash
uv run python main.py worker
```

Installation check:

```bash
uv run python main.py check
```

## Operational Endpoints

- /docs
- /redoc
- /metrics
- /livez
- /readyz
- /health

## Core API

- POST /v1/models/train
- GET /v1/jobs/{job_id}
- GET /v1/models
- GET /v1/models/{model_id}
- POST /v1/models/{model_id}/predict
- GET /v1/system/capabilities

## Reliability Defaults

- Job state transitions persisted in SQL
- Celery queue prefetch multiplier tuned for fair dispatch
- Late acknowledgements enabled
- Configurable soft/hard task time limits
- Readiness requires both PostgreSQL and Redis availability

## Container Notes

Backend Docker build uses uv in multi-stage mode and produces a runtime image with only application code and resolved dependencies.
