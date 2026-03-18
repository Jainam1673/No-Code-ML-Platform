# Role Matrix: Personal and Professional Usage

## SWE

- Extend API routes in backend/app/api/routes.
- Add service-layer logic in backend/app/services with typed contracts.
- Use uv for reproducible local and CI execution.

## SRE

- Operate liveness/readiness/metrics endpoints.
- Apply docs/OPERATIONS.md for incident response and rollback.
- Use Kubernetes probes, PDBs, HPA, and NetworkPolicies from infra/k8s/base.

## DevOps

- Build and publish container images from Dockerfiles.
- Use compose for local parity and kustomize for cluster deploys.
- Apply migration-first release process via backend-migrate job.

## Platform Engineer

- Add overlays under infra/k8s/overlays for environment-specific policy.
- Enable GPU training with infra/k8s/overlays/gpu-worker.
- Integrate secret managers and policy engines in cluster baseline.

## Data Scientist / Data Engineer

- Submit training jobs with dataset path and target via /v1/models/train.
- Inspect model metadata and inference endpoints through v1 APIs.
- Use artifact and registry outputs for experiment handoff.

## ML Engineer / Scientist

- Scale workers independently from API for long-running training.
- Tune queue behavior (Celery config, worker concurrency, queue names).
- Add model-specific preprocessing and evaluation logic in services.

## GPU / TPU Engineers

- GPU path: worker overlay requests nvidia.com/gpu and dedicated queue.
- TPU path: platform exposes runtime capability endpoint and can be extended via accelerator-specific worker pools.
- Capability introspection endpoint: /v1/system/capabilities.

## Personal Use

- Run complete local stack with Docker Compose.
- Iterate rapidly with uv (backend) and Bun (frontend).

## Professional Use

- Deploy to Kubernetes with secure baseline and upgrade-safe migrations.
- Evolve into enterprise controls: authn/authz, tracing, governance, cost controls.
