# Operations Runbook

Operational guidance for running this platform in production-like environments.

## Service Objectives (Initial Targets)

- API availability: 99.9%
- P95 inference latency: < 800ms
- Queue lag: < 5 minutes under normal load
- Training success rate: > 98%

These are baseline targets and should be tuned with real traffic and model profiles.

## Key Signals

- Traffic: request rate on `GET/POST /v1/*`
- Latency: p50/p95 API duration
- Errors: 4xx/5xx rates and failure distribution by endpoint
- Saturation:
  - API CPU and memory
  - Worker CPU and memory
  - Redis queue depth
  - Worker concurrency utilization

## Alert Recommendations

- API health degraded for > 5 minutes
- Readiness failures due to PostgreSQL or Redis dependencies
- Worker deployment unavailable or crash looping
- Queue lag above threshold for > 10 minutes
- Elevated training failure ratio over rolling time window

## Deployment Procedure

1. Confirm target image tags and release notes.
2. Ensure infrastructure dependencies are healthy (PostgreSQL, Redis).
3. Run migration job before app rollout.
4. Deploy backend API and workers.
5. Deploy frontend and ingress/gateway.
6. Validate probes and health endpoints.
7. Execute smoke test for training and inference.
8. Monitor error rate and queue depth for at least one observation window.

## Post-Deploy Verification

- `GET /livez` is stable.
- `GET /readyz` reports database and Redis as healthy.
- `GET /metrics` is scrape-ready.
- Training job lifecycle transitions complete (queued -> running -> succeeded/failed).
- Inference endpoint returns valid predictions for a known model.

## Rollback Procedure

1. Roll back backend and worker images.
2. Evaluate schema compatibility with rolled-back application version.
3. If required, perform controlled Alembic downgrade.
4. Re-validate health, readiness, and smoke tests.
5. Publish incident summary and remediation tasks.

## Incident Response Checklist

1. Check `GET /health` for dependency state.
2. Compare `GET /livez` and `GET /readyz` behavior to identify dependency impact.
3. Inspect backend logs for request ID correlated failures.
4. Inspect worker logs for task exceptions and retry patterns.
5. Check Redis queue depth and broker availability.
6. Inspect PostgreSQL connectivity, lock contention, and resource pressure.
7. Apply mitigations:
   - scale workers if backlog is rising
   - reduce training concurrency if memory pressure is high
   - roll back faulty release if regression is confirmed

## Capacity Planning Guidelines

- Start with 2 API and 2 worker replicas.
- Tune worker concurrency only after measuring memory per training job.
- Keep artifact storage persistent and sized for model lifecycle retention.
- Isolate long-running training from inference nodes under heavier workloads.

## Security and Compliance Baseline

- Never commit plaintext credentials.
- Use Kubernetes secrets or managed external secret providers.
- Restrict network access to data stores.
- Run vulnerability scanning in CI/CD.
- Keep containers non-root with minimal privileges.

## Common Failure Modes

- Queue backlog growth
  - Symptoms: rising job wait time and stale queued state
  - Actions: scale workers, inspect Redis health, verify worker startup/config
- Migration mismatch
  - Symptoms: startup failures or ORM errors after deploy
  - Actions: verify migration order, apply missing revision, roll back app if needed
- Resource exhaustion during training
  - Symptoms: OOM kills, worker restart loops, increased task failures
  - Actions: lower concurrency, adjust memory limits, split workloads by model profile
