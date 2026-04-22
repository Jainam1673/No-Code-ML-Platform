# AI Platform Architecture

## Purpose

This architecture is designed to demonstrate strong engineering fundamentals for ML platform development:

- clean service boundaries
- operational reliability
- asynchronous workload isolation
- deterministic developer experience

## Primary Quality Attributes

- Reliability: dependency-aware readiness checks and migration-first rollouts.
- Scalability: independent scaling of API and worker planes.
- Operability: health and metrics endpoints plus runbooks.
- Security baseline: non-root containers and constrained runtime posture.
- Maintainability: typed contracts and service-centric backend organization.

## System Planes

- API Plane (FastAPI)
  - Accepts train and predict requests.
  - Persists and serves job/model metadata.
  - Enqueues training tasks to Celery.
- Training Plane (Celery Worker)
  - Pulls queued tasks from Redis.
  - Executes AutoGluon Tabular training.
  - Writes model artifacts and metadata updates.
- Data Plane
  - PostgreSQL for durable job/model records.
  - Redis for broker and result backend.
  - Persistent artifact volume for model binaries.
- Experience Plane
  - Next.js frontend that surfaces platform status and API entrypoints.

## Request and Training Flow

1. Client submits `POST /v1/models/train` with dataset path and target column.
2. API creates `job_records` entry with queued status.
3. API dispatches training task to Celery queue.
4. Worker marks job running and starts AutoGluon training.
5. Trained model is written to artifact storage.
6. Worker stores model metadata in `model_records` and JSON registry.
7. Worker marks job succeeded (or failed with error details).
8. Client polls `GET /v1/jobs/{job_id}` and performs inference via `POST /v1/models/{model_id}/predict`.

## Deployment Topologies

- Local topology
  - Docker Compose with gateway, frontend, backend, worker, Redis, PostgreSQL.
- Cluster topology
  - Kubernetes deployments for backend, worker, frontend.
  - Migration job for controlled schema rollout.
  - Ingress and service resources for traffic management.

## Reliability Model

- `GET /livez`: process liveness only.
- `GET /readyz`: validates database and Redis connectivity.
- `GET /metrics`: Prometheus scrape endpoint.
- HPAs for API, worker, and frontend.
- PodDisruptionBudgets to reduce voluntary downtime.
- Rollout safety via rolling updates and probes.

## Toolchain and Build Policy

- Backend dependency and execution workflow is `uv` only.
- Frontend workflow is `bun` only.
- Container builds preserve the same policy to reduce local-vs-prod drift.

## Runtime and Storage Model

- Metadata model
  - `job_records`: asynchronous lifecycle state
  - `model_records`: model identity and provenance
- Artifacts
  - Filesystem-backed model output under `artifacts/models`
  - JSON registry snapshots under `artifacts/registry`

## Security and Isolation Baseline

- Non-root runtime users in containers.
- Reduced Linux capabilities.
- Network policy baseline in Kubernetes manifests.
- Secrets externalized from plain manifests via secret references.

## Extensibility Path

- Add tenant-aware authn/authz and policy layers.
- Move artifact registry to object storage.
- Add distributed tracing and SLO alerting integrations.
- Introduce dedicated node pools/queues for accelerator-specific workloads.

## Accelerator Strategy

- CPU worker deployment is the baseline.
- Optional GPU worker overlay exists in `infra/k8s/overlays/gpu-worker`.
- Runtime introspection endpoint `GET /v1/system/capabilities` reports host capabilities.
- TPU can be added as separate worker pools for TPU-compatible training stacks.

## Reference Notes

AutoGluon API reference snapshots are stored under `docs/reference/autogluon/` for offline browsing and implementation support.
