# Contributing

## Development Standards

- Backend: Python 3.12 with uv.
- Frontend: Next.js with Bun.
- Keep service-layer logic in backend/app/services.
- Keep API handlers thin and schema-driven.

## Local Setup

1. Backend:

```bash
make backend-sync
make backend-db-upgrade
```

2. Frontend:

```bash
make frontend-install
```

## Quality Gates

Run before opening a pull request:

```bash
make backend-lint
make backend-test
make backend-check
make frontend-build
```

## Pull Request Expectations

- Include a concise change summary.
- Add or update tests for behavioral changes.
- Update relevant docs in README or docs/.
- Keep changes scoped; avoid unrelated refactors.

## Architecture Conventions

- Add new backend capabilities via services first, then expose routes.
- Preserve typed boundaries between schemas, services, and persistence.
- Use migration-first schema evolution with Alembic.
