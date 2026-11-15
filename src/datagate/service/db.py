"""Persistence layer — the system of record from the design doc's data model.

SQLAlchemy ORM so the same models run on SQLite (dev) and Postgres (prod).
JSON columns hold the engine's Pydantic payloads verbatim: the engine owns
those schemas, the database just stores them.

There is no migration framework yet; ``init_db`` adds any column a model has
gained since the database was created (nullable/defaulted columns only), which
is enough to carry a local database across releases without data loss.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
    inspect,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.types import JSON

from datagate.service.config import settings


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class DealState(str, Enum):
    LISTED = "listed"
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    WITHDRAWN = "withdrawn"


class Base(DeclarativeBase):
    pass


class DatasetRecord(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(512))
    storage_ref: Mapped[str] = mapped_column(String(1024))
    format: Mapped[str] = mapped_column(String(16))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    n_rows: Mapped[int] = mapped_column(Integer, default=0)
    n_cols: Mapped[int] = mapped_column(Integer, default=0)
    fingerprint: Mapped[str] = mapped_column(String(64), default="")
    profile: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SpecRecord(Base):
    __tablename__ = "specs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    dataset_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("datasets.id"), nullable=True)
    spec: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class JobRecord(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id"))
    spec_id: Mapped[str] = mapped_column(String(36), ForeignKey("specs.id"))
    reference_dataset_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("datasets.id"), nullable=True)
    deal_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    mode: Mapped[str] = mapped_column(String(1), default="A")
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.QUEUED.value)
    progress: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    report_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True)


class ReportRecord(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(String(36), ForeignKey("jobs.id"))
    verdict: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    safety_score: Mapped[float] = mapped_column(Float, default=0.0)
    fit_score: Mapped[float] = mapped_column(Float, default=0.0)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # Blob holding the complete quarantine list (the payload embeds a capped one).
    quarantine_ref: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    blind: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, default=0)
    engine_version: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class DealRecord(Base):
    """A blind-evaluation listing (marketplace preview — no payments)."""

    __tablename__ = "deals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id"))
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    asking_price: Mapped[str] = mapped_column(String(64), default="")
    state: Mapped[str] = mapped_column(String(16), default=DealState.LISTED.value)
    report_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    history: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


_connect_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)
engine = create_engine(settings.database_url, connect_args=_connect_args, future=True)

if settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record) -> None:
        # WAL lets the API read while a worker writes; the timeout rides out
        # brief write contention instead of failing with "database is locked".
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def _add_missing_columns() -> None:
    """Bring an older database up to the current models (additive changes only)."""
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                ddl_type = column.type.compile(dialect=engine.dialect)
                conn.execute(text(
                    f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl_type}'))


def init_db() -> None:
    Base.metadata.create_all(engine)
    _add_missing_columns()


def session_scope() -> Session:
    return SessionLocal()
