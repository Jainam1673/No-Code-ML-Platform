# Operations Runbook

## Core SLO Targets

- API availability: 99.9%
- P95 API latency: < 800ms for inference endpoints
- Training queue lag: < 5 minutes at normal load
- Training job success rate: > 98%

## Golden Signals

- Traffic: requests per second on `/v1/*`
- Latency: p50/p95 request duration
- Errors: 4xx and 5xx rates
- Saturation: CPU, memory, queue depth, worker concurrency utilization

## Baseline Alerts

- API health degraded for 5 minutes
- Database unreachable from API/worker
- Redis unreachable from API/worker
- Worker replicas unavailable
- Queue lag high for 10 minutes
- Pod restart loop (CrashLoopBackOff)

## Deployment Procedure

1. Deploy infrastructure dependencies: PostgreSQL and Redis.
2. Run database migration job.
3. Deploy backend API and worker.
4. Deploy frontend and ingress.
5. Verify health (`/health`) and metrics (`/metrics`).
6. Execute smoke test training and prediction requests.

## Rollback Procedure

1. Roll back backend and worker image tags.
2. If migration is backward-compatible, keep schema and resume traffic.
3. If migration requires rollback, execute a controlled down migration from Alembic.
4. Re-run smoke tests.

## Capacity Guidelines

- Start with 2 API replicas and 2 worker replicas.
- Increase worker concurrency only after measuring memory usage per training job.
- Keep model artifacts on persistent shared storage.
- For heavy workloads, isolate training and inference onto separate node pools.

## Security Baseline

- Do not store plaintext credentials in manifests.
- Use Kubernetes secrets or external secret manager.
- Restrict network access to PostgreSQL and Redis.
- Enable image scanning in CI.
- Run containers as non-root where possible.

## Incident Checklist

1. Check `/health` dependency statuses.
2. Check `/readyz` and `/livez` endpoint behavior.
3. Check backend logs for request failures and request IDs.
4. Check worker logs for task failures.
5. Check Redis availability and queue pressure.
6. Check database connectivity and lock contention.
7. Scale worker replicas if backlog is rising and resources are available.
