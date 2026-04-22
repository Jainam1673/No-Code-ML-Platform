# No-Code ML Platform

Production-grade ML platform reference implementation with a clear separation of concerns between online serving and offline model training.

This repository is designed for interview demonstrations of system design, backend engineering, cloud-native operations, and practical MLOps execution.

## Why This Project

- Shows end-to-end ML system ownership: API, async training, metadata persistence, and frontend integration.
- Uses modern, deterministic toolchains: `uv` for Python and `Bun` for frontend.
- Demonstrates production concerns: health/readiness, migrations, autoscaling, rollout safety, and operational runbooks.
- Keeps implementation readable and extensible for future enterprise features.

## Architecture At A Glance

- API plane: FastAPI service for train/predict requests and metadata access.
- Training plane: Celery workers running AutoGluon Tabular training jobs.
- Data plane: PostgreSQL for durable records, Redis for queueing.
- Artifact plane: persisted model files and JSON model registry.
- Experience plane: Next.js frontend for status and platform entry.
- Deployment plane: Docker Compose for local parity and Kubernetes for production orchestration.

See detailed docs:

- `docs/ARCHITECTURE.md`
- `docs/OPERATIONS.md`
- `docs/ROLE_MATRIX.md`
- `docs/RELEASE_CHECKLIST.md`
- `CONTRIBUTING.md`

## Tech Stack

- Backend: Python 3.12, FastAPI, SQLAlchemy, Alembic, Celery, AutoGluon Tabular
- Frontend: Next.js 16, React 19, TypeScript, Tailwind CSS
- Infra: PostgreSQL 16, Redis 7, NGINX, Docker Compose, Kubernetes
- Tooling policy:
  - Python dependencies and execution via `uv`
  - Frontend dependencies and runtime via `Bun`
  - No pip-based or npm-based application dependency flow in Dockerfiles

## Quickstart (Local Native)

Prerequisites:

- Python 3.12
- `uv`
- `bun`
- Docker (optional)

1. Backend setup and migration

```bash
make backend-sync
make backend-db-upgrade
```

2. Run backend API

```bash
make backend-serve
```

3. Run async training worker (new terminal)

```bash
make backend-worker
```

4. Run frontend (new terminal)

```bash
make frontend-install
make frontend-dev
```

Access points:

- Frontend: http://localhost:3000
- Backend docs: http://localhost:8000/docs
- Backend health: http://localhost:8000/health
- Backend metrics: http://localhost:8000/metrics

## Quickstart (Docker Compose)

Start all services:

```bash
docker compose up --build
```

Apply migrations before first API/worker usage:

```bash
docker compose run --rm migrate
```

Compose gateway and services:

- Frontend gateway: http://localhost
- Backend health via gateway: http://localhost/api/health
- Backend docs: http://localhost/docs
- PostgreSQL: localhost:5432
- Redis: localhost:6379

## Core API Surface

- `GET /livez`
- `GET /readyz`
- `GET /health`
- `GET /metrics`
- `GET /v1/system/capabilities`
- `POST /v1/models/train`
- `GET /v1/jobs/{job_id}`
- `GET /v1/models`
- `GET /v1/models/{model_id}`
- `POST /v1/models/{model_id}/predict`

## Repository Layout

```text
backend/   FastAPI API, Celery worker, DB models/migrations, services
frontend/  Next.js application
infra/     Docker and Kubernetes manifests
docs/      Architecture, operations, release, role matrix, references
```

## Interview Demo Flow

1. Explain architecture separation: API plane vs training plane.
2. Show reliability endpoints (`/livez`, `/readyz`, `/metrics`).
3. Submit a training request and poll job status.
4. List trained models and run prediction.
5. Walk through deployment safety controls in Kubernetes manifests.
6. Highlight operational readiness with `docs/OPERATIONS.md` and `docs/RELEASE_CHECKLIST.md`.

## Kubernetes Deployment Baseline

1. Set real image tags in `infra/k8s/base/backend.yaml` and `infra/k8s/base/frontend.yaml`.
2. Create real secrets from `infra/k8s/base/secret.example.yaml`.
3. Apply manifests:

```bash
kubectl apply -k infra/k8s/base
```

4. Trigger migration job:

```bash
kubectl -n nocodeml create job --from=job/backend-migrate backend-migrate-$(date +%s)
```

5. Validate rollout:

```bash
kubectl -n nocodeml get deploy,pods,svc,hpa,pdb
```

Optional GPU workers:

```bash
kubectl apply -k infra/k8s/overlays/gpu-worker
```

## Current Production Readiness Features

- Non-root containers and reduced Linux capabilities
- Readiness/liveness/startup probes
- Rolling updates with disruption controls
- PodDisruptionBudgets and autoscaling
- Migration-first deployment path
- Queue-backed asynchronous training execution

## What Is Next

Planned enhancements for enterprise-hardening:

- Authentication and authorization boundaries
- Object store-backed artifact registry
- Tracing and SLO-driven alerting dashboards
- Multi-tenant resource controls and governance policies
