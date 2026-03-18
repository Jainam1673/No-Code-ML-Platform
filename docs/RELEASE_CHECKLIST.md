# Release Checklist

## Pre-Release

- Update container image tags for backend and frontend.
- Confirm migration revision is present for schema changes.
- Validate docs impacted by API or infra updates.

## Verification Commands

```bash
make backend-lint
make backend-test
make backend-check
make frontend-build
```

## Deployment Sequence

1. Apply cluster manifests.
2. Run migration job.
3. Roll out backend and worker.
4. Roll out frontend and ingress.
5. Verify health and metrics endpoints.

## Post-Release

- Confirm no elevated 5xx rates.
- Confirm worker queue drain behavior is stable.
- Confirm model training and prediction smoke flows succeed.
