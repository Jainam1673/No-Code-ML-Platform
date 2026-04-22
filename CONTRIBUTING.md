# Contributing

Thank you for improving this repository. The objective is to keep it production-oriented, deterministic to run, and easy to explain in technical interviews.

## Engineering Principles

- Prefer clarity over cleverness.
- Keep API layer thin and business logic in services.
- Preserve typed boundaries between schemas, services, and persistence.
- Design for operability: every behavior should be observable and testable.
- Keep changes small, scoped, and reviewable.

## Required Tooling

- Backend: Python 3.12 and `uv`
- Frontend: `bun`
- Optional local parity: Docker + Docker Compose

Toolchain policy is strict:

- Use `uv` for backend dependency management and command execution.
- Use `bun` for frontend dependency management and scripts.
- Do not introduce pip- or npm-based project workflows for app dependencies.

## Local Setup

Backend:

```bash
make backend-sync
make backend-db-upgrade
```

Frontend:

```bash
make frontend-install
```

## Development Workflow

1. Sync dependencies.
2. Implement change with tests.
3. Run quality gates.
4. Update docs for any user-facing, API, or operational impact.
5. Open a focused pull request.

## Quality Gates

Run all checks before opening a pull request:

```bash
make backend-lint
make backend-test
make backend-check
make frontend-build
```

If your change affects deployment or runtime behavior, validate relevant runbook steps in `docs/OPERATIONS.md`.

## Pull Request Requirements

- Clear title and concise summary of intent.
- Behavior change description, including migration or rollout impact.
- Tests added or updated for changed behavior.
- Documentation updated (`README.md`, `docs/*`, or module READMEs).
- No unrelated refactors.

Suggested PR template sections:

- Context
- Change summary
- Test evidence
- Rollout and rollback notes
- Follow-up items

## Code Organization Conventions

- `backend/app/api/routes`: request parsing and response shaping only.
- `backend/app/services`: orchestration and domain logic.
- `backend/app/schemas`: typed request/response models.
- `backend/app/db`: persistence and migration integration.
- `backend/app/worker`: asynchronous task execution.
- `frontend/app`: UI and API-facing presentation layer.

## Schema and Migration Rules

- Schema updates require an Alembic migration.
- Prefer backward-compatible, migration-first deployment sequencing.
- Do not merge schema-dependent runtime code without migration artifacts.

## Documentation Rules

- Every meaningful capability change must update at least one doc.
- API contract changes should be reflected in root `README.md` and backend README.
- Operational changes should update `docs/OPERATIONS.md` and release checklist as needed.

## Security and Reliability Baseline

- Never commit secrets.
- Keep health endpoints and readiness semantics intact.
- Preserve non-root runtime expectations in containers/manifests.
- Fail safely with actionable error messages.
