from __future__ import annotations

import argparse
from importlib.metadata import version

from app.core.config import settings


def run_check() -> None:
    print(f"AutoGluon Tabular version: {version('autogluon-tabular')}")
    print("AutoGluon Tabular is ready.")


def run_serve() -> None:
    from app.server import run as run_server

    run_server()


def run_worker() -> None:
    from app.worker.celery_app import celery_app

    celery_app.worker_main(
        [
            "worker",
            "--loglevel=INFO",
            f"--concurrency={settings.max_training_workers}",
            f"--queues={settings.celery_queue}",
        ]
    )


def run_db_upgrade() -> None:
    from app.db.migrations import upgrade_head

    upgrade_head()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="No-Code ML backend")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Verify AutoGluon installation")
    check_parser.set_defaults(handler=lambda _: run_check())

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI backend service")
    serve_parser.set_defaults(handler=lambda _: run_serve())

    worker_parser = subparsers.add_parser("worker", help="Run the Celery training worker")
    worker_parser.set_defaults(handler=lambda _: run_worker())

    db_upgrade_parser = subparsers.add_parser("db-upgrade", help="Apply Alembic migrations to head")
    db_upgrade_parser.set_defaults(handler=lambda _: run_db_upgrade())

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
