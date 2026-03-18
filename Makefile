SHELL := /bin/bash

.PHONY: backend-sync backend-serve backend-worker backend-check backend-db-upgrade backend-test backend-lint backend-typecheck frontend-install frontend-dev frontend-build compose-up compose-down k8s-apply

backend-sync:
	cd backend && uv sync

backend-serve:
	cd backend && uv run python main.py serve

backend-worker:
	cd backend && uv run python main.py worker

backend-check:
	cd backend && uv run python main.py check

backend-db-upgrade:
	cd backend && uv run python main.py db-upgrade

backend-test:
	cd backend && uv run pytest

backend-lint:
	cd backend && uv run ruff check app tests main.py

backend-typecheck:
	cd backend && uv run mypy app

frontend-install:
	cd frontend && bun install --frozen-lockfile

frontend-dev:
	cd frontend && bun run dev

frontend-build:
	cd frontend && bun run build

compose-up:
	docker compose up --build

compose-down:
	docker compose down

k8s-apply:
	kubectl apply -k infra/k8s/base
