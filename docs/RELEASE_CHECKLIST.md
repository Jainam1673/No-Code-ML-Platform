# Release Checklist

Use this checklist for controlled, low-risk releases.

## 1. Pre-Release Readiness

- Confirm release scope and linked change notes.
- Verify backend and frontend image tags are immutable and published.
- Ensure schema-impacting changes include Alembic migration artifacts.
- Confirm documentation updates for behavior, API, or ops changes.
- Validate secrets/config dependencies for target environment.

## 2. Quality Gates

Run all required checks before rollout:

```bash
make backend-lint
make backend-test
make backend-check
make frontend-build
```

Record pass/fail evidence in release notes.

## 3. Deployment Sequence

1. Apply infra/config updates.
2. Run migration job to completion.
3. Roll out backend and worker.
4. Roll out frontend and ingress/gateway.
5. Verify rollout health and service readiness.

## 4. Post-Deploy Validation

- `GET /livez` and `GET /readyz` return expected status.
- `GET /metrics` remains scrape-ready.
- API error rates remain within baseline.
- Queue lag and worker health remain stable.
- Smoke flow succeeds:
  - submit training job
  - verify lifecycle transition
  - run prediction on resulting model

## 5. Rollback Plan

- Roll back application image tags first.
- Evaluate migration compatibility with rolled-back code.
- Perform controlled Alembic downgrade only when needed.
- Re-run smoke and health checks after rollback.

## 6. Release Closeout

- Publish summary: shipped changes, incidents, mitigations.
- Capture follow-up tasks for reliability/performance debt.
- Update runbook or checklist if process gaps were found.
