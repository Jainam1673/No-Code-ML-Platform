# No Code ML Platform

Production-focused ML systems project designed to demonstrate practical AI infrastructure and platform architecture.

## Core Stack

- Backend API and worker: FastAPI + Celery + AutoGluon Tabular
- Metadata and queue infrastructure: PostgreSQL + Redis
- Frontend: Next.js 16
- Build/runtime tooling:
  - Python dependency management and execution with uv
  - Frontend dependency management/build/runtime with Bun
- Orchestration: Docker Compose and Kubernetes

Architecture and operations documentation:

- docs/ARCHITECTURE.md
- docs/OPERATIONS.md
- docs/ROLE_MATRIX.md
- docs/RELEASE_CHECKLIST.md
- CONTRIBUTING.md

## Toolchain Consistency

This repository intentionally uses uv and Bun across local and container workflows.

- No npm commands are used in Dockerfiles.
- No pip-based dependency installation is used in Dockerfiles for project dependencies.
- Backend containers resolve dependencies with uv.
- Frontend containers build and run with Bun.

## Local Development

Prerequisites:

- Python 3.12
- uv
- Bun
- Docker (optional, for compose workflow)

Convenience workflow (recommended):

```bash
make backend-sync
make backend-db-upgrade
make backend-serve
make backend-worker
```

In another terminal:

```bash
make frontend-install
make frontend-dev
```

## Docker Compose Workflow

Start platform services:

```bash
docker compose up --build
```

Apply migrations before first API/worker usage:

```bash
docker compose run --rm migrate
```

Service entry points:

- Frontend gateway: http://localhost
- Backend health: http://localhost/api/health
- Backend docs: http://localhost/docs
- PostgreSQL: localhost:5432
- Redis: localhost:6379

## Runtime Endpoints

- GET /livez
- GET /readyz
- GET /health
- GET /metrics
- GET /v1/system/capabilities

## Kubernetes Deployment

1. Update backend/frontend image references in:
   - infra/k8s/base/backend.yaml
   - infra/k8s/base/frontend.yaml
2. Create real secrets using infra/k8s/base/secret.example.yaml as template.
3. Apply base manifests:

```bash
kubectl apply -k infra/k8s/base
```

4. Trigger migration job:

```bash
kubectl -n nocodeml create job --from=job/backend-migrate backend-migrate-$(date +%s)
```

5. Verify resources:

```bash
kubectl -n nocodeml get deploy,pods,svc,hpa,pdb
```

Optional GPU worker overlay:

```bash
kubectl apply -k infra/k8s/overlays/gpu-worker
```

## Production Baseline Included

- Non-root containers
- Rolling updates for app workloads
- Startup/readiness/liveness probes
- PodDisruptionBudgets
- Worker/API autoscaling
- Migration-first deployment path

## Scope

This codebase is a strong production-oriented foundation for AI platform work. It is intentionally extensible for enterprise features such as authn/authz, policy control, and advanced observability pipelines.
