# Role Matrix

This matrix clarifies how different engineering roles interact with this platform. It is intentionally written to support interview discussion across software, infrastructure, and ML domains.

## Software Engineer

- Implement request contracts in `backend/app/schemas`.
- Keep HTTP handlers thin in `backend/app/api/routes`.
- Place orchestration logic in `backend/app/services`.
- Validate behavior with backend tests and static checks.

## Site Reliability Engineer (SRE)

- Operate service health via `/livez`, `/readyz`, `/health`, and `/metrics`.
- Use `docs/OPERATIONS.md` for incident response and mitigation.
- Tune autoscaling, disruption budgets, and rollout controls in Kubernetes manifests.

## DevOps Engineer

- Build and publish backend/frontend images.
- Maintain local parity with Docker Compose.
- Enforce migration-first deployment path via migration job.
- Keep CI quality gates aligned with repository standards.

## Platform Engineer

- Extend baseline manifests with environment-specific overlays.
- Integrate cluster policy controls, secret management, and network security.
- Add accelerator-aware worker pools where required.

## ML Engineer

- Extend training workflows and model lifecycle behavior.
- Tune Celery queue configuration and worker concurrency.
- Add model-specific preprocessing and evaluation pipelines.
- Improve artifact lineage and reproducibility metadata.

## Data Scientist / Analytics Engineer

- Submit training jobs using `POST /v1/models/train`.
- Track job lifecycle using `GET /v1/jobs/{job_id}`.
- Consume model metadata and prediction APIs for experiments.

## Accelerator Engineer (GPU/TPU)

- Use GPU overlay under `infra/k8s/overlays/gpu-worker`.
- Validate runtime capabilities with `GET /v1/system/capabilities`.
- Introduce hardware-specific worker pools and queue routing strategy.

## Interview Positioning

Use this repository to demonstrate:

- Full-stack ownership from API to deployment.
- Practical MLOps architecture with asynchronous training.
- Production operations mindset with reliability safeguards.
- Clear extension path toward enterprise-scale platform controls.
