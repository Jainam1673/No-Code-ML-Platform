# Backend

Backend runtime for model training orchestration, metadata persistence, and inference serving.

Design goals:

- Keep domain logic explicit and testable.
- Isolate request handling from orchestration.
- Support async training with durable job lifecycle tracking.
- Preserve deployability from local development to Kubernetes.

## Technology

- Python 3.12
- `uv` for dependency resolution and command execution
- FastAPI for HTTP API surface
- Celery + Redis for asynchronous training
- SQLAlchemy + Alembic for persistence and schema migrations
- AutoGluon Tabular for model training/inference

## Package Layout

- `app/api/routes`: HTTP endpoints, request/response mapping
- `app/services`: business orchestration and domain workflows
- `app/schemas`: typed contracts for API payloads
- `app/db`: ORM models, sessions, migration integration
- `app/worker`: Celery app and background task definitions
- `app/core`: config, middleware, logging

Artifacts and model registry:

- `artifacts/models/<model_id>`
- `artifacts/registry/<model_id>.json`

## Local Setup

```bash
uv sync
cp .env.example .env
```

Apply migrations:

```bash
uv run python main.py db-upgrade
```

## Runtime Commands

Run API server:

```bash
uv run python main.py serve
```

Run worker:

```bash
uv run python main.py worker
```

Verify AutoGluon installation:

```bash
uv run python main.py check
```

## Operational Endpoints

- `GET /docs`
- `GET /redoc`
- `GET /metrics`
- `GET /livez`
- `GET /readyz`
- `GET /health`

## Core API Endpoints

- `POST /v1/models/train`
- `GET /v1/jobs/{job_id}`
- `GET /v1/models`
- `GET /v1/models/{model_id}`
- `POST /v1/models/{model_id}/predict`
- `GET /v1/system/capabilities`

## Training Lifecycle

1. Client submits training request with dataset path and target column.
2. API persists queued job status.
3. Job dispatches via Celery (or local threadpool fallback).
4. Worker trains AutoGluon model and writes artifacts.
5. Model metadata is persisted in PostgreSQL and JSON registry.
6. Client polls job status and uses model for prediction.

## Reliability Characteristics

- Durable job state transitions in SQL.
- Celery late acknowledgements for safer task handling.
- Configurable soft and hard task time limits.
- Worker prefetch tuning for fairer queue distribution.
- Readiness checks both PostgreSQL and Redis dependencies.

## Configuration Surface

Primary config is read from environment variables with `NOCODEML_` prefix. Key controls include:

- Server host and port
- DB and Redis URLs
- Celery broker/result backend/queue
- Worker concurrency
- Metrics toggle
- Runtime migration and table creation toggles

See `app/core/config.py` for the full settings model.

## Testing and Quality

```bash
uv run pytest
uv run ruff check app tests main.py
uv run mypy app
```

## Container Notes

The backend Docker image uses a multi-stage build with `uv` to produce a lean runtime image while preserving deterministic dependency resolution.
