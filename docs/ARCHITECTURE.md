# AI Platform Architecture

## Design Goals

- Separate online inference/API concerns from heavy offline training workloads.
- Preserve model lineage and job history with durable metadata.
- Keep runtime deployable in both local compose and Kubernetes.
- Provide clear operational hooks for SRE (health, metrics, migration workflow).

## Logical Components

- API Plane (FastAPI):
  - Receives train/predict requests.
  - Stores and exposes job/model metadata.
  - Enqueues training jobs to Redis/Celery.
- Training Plane (Celery Worker):
  - Pulls queued training tasks.
  - Runs AutoGluon tabular training.
  - Persists artifacts and model registry metadata.
- Data Plane:
  - PostgreSQL for job/model records.
  - Redis for queue broker/result backend.
  - Artifact volume for trained model binaries.
- Experience Plane:
  - Next.js frontend with API integration.

## Reliability Model

- Liveness endpoint (`/livez`) checks process health only.
- Readiness endpoint (`/readyz`) checks DB and Redis dependencies.
- Prometheus metrics endpoint (`/metrics`) for scrape-based telemetry.
- Kubernetes HPA on API, worker, and frontend deployments.
- PodDisruptionBudgets to reduce voluntary outage during maintenance.

## Deployment Topology

- Local: Docker Compose with gateway, API, worker, PostgreSQL, Redis, frontend.
- Cluster: Kubernetes deployments + migration job + ingress.
- Migration-first strategy: schema upgrades before API/worker rollout.

## Container Strategy

- Backend: Python multi-stage image with uv-managed dependencies resolved during build.
- Frontend: Bun-based Next.js build and Bun runtime with standalone output.
- Runtime containers execute as non-root to reduce attack surface.

## Toolchain Policy

- Python workflows use uv in local development and Docker builds.
- Frontend workflows use Bun in local development and Docker builds.
- Dockerfiles avoid npm and pip dependency workflows for application packages.

## Kubernetes Runtime Baseline

- RollingUpdate strategy with `maxUnavailable: 0` for app deployments.
- Startup/readiness/liveness probes for safer rollout behavior.
- Pod anti-affinity and topology spread constraints for resilience.
- PodDisruptionBudgets to preserve availability during node events.

## Model Lifecycle (Current)

1. Client submits `/v1/models/train`.
2. API creates job record and enqueues Celery task.
3. Worker trains model and writes artifacts to persistent volume.
4. Worker updates model metadata in PostgreSQL and registry JSON.
5. Client polls `/v1/jobs/{job_id}` and uses `/v1/models/{model_id}/predict`.

## Reference Artifacts

- AutoGluon API surface exports are kept under `docs/reference/autogluon/` for offline reference.

## Scale-Out Path

- Isolate training onto dedicated node pools with taints/tolerations.
- Add external object store-backed artifact registry.
- Add distributed tracing and SLO-driven alerting.
- Introduce authn/authz boundaries and tenant isolation.

## Accelerator Strategy

- Baseline CPU worker deployment in base manifests.
- Optional GPU worker overlay under `infra/k8s/overlays/gpu-worker`.
- Runtime capability introspection endpoint (`/v1/system/capabilities`) for accelerator-aware operations.
- TPU workloads can be introduced as dedicated worker pools where model stack requires TPU-compatible frameworks.
