from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from app.core.config import settings


def _config() -> Config:
    config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
    return config


def upgrade_head() -> None:
    config = _config()
    engine = create_engine(settings.database_url)

    with engine.connect() as connection:
        inspector = inspect(connection)
        tables = set(inspector.get_table_names())
        alembic_revision = None
        if "alembic_version" in tables:
            row = connection.execute(text("SELECT version_num FROM alembic_version LIMIT 1")).fetchone()
            alembic_revision = row[0] if row else None

    has_app_tables = {"job_records", "model_records"}.issubset(tables)
    has_alembic_version = "alembic_version" in tables
    needs_stamp = has_app_tables and (not has_alembic_version or alembic_revision is None)

    if needs_stamp:
        command.stamp(config, "head")
        return

    command.upgrade(config, "head")
