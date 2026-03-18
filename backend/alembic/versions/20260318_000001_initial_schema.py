"""Initial schema

Revision ID: 20260318_000001
Revises: None
Create Date: 2026-03-18 00:00:01
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260318_000001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "job_records",
        sa.Column("job_id", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("model_id", sa.String(length=64), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("job_id"),
    )

    op.create_table(
        "model_records",
        sa.Column("model_id", sa.String(length=64), nullable=False),
        sa.Column("label_column", sa.String(length=255), nullable=False),
        sa.Column("problem_type", sa.String(length=64), nullable=False),
        sa.Column("eval_metric", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("model_path", sa.Text(), nullable=False),
        sa.Column("source_dataset", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("model_id"),
    )


def downgrade() -> None:
    op.drop_table("model_records")
    op.drop_table("job_records")
