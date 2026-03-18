from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class JobRecordORM(Base):
    __tablename__ = "job_records"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    state: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    model_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class ModelRecordORM(Base):
    __tablename__ = "model_records"

    model_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    label_column: Mapped[str] = mapped_column(String(255), nullable=False)
    problem_type: Mapped[str] = mapped_column(String(64), nullable=False)
    eval_metric: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    model_path: Mapped[str] = mapped_column(Text, nullable=False)
    source_dataset: Mapped[str] = mapped_column(Text, nullable=False)
