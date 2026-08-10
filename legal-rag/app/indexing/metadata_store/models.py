"""SQLAlchemy records for streaming ingestion and legal chunk metadata."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.utcnow()


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class DocumentRecord(Base):
    __tablename__ = "documents"
    document_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_link: Mapped[str | None] = mapped_column(Text, nullable=True)
    checksum: Mapped[str] = mapped_column(String(64), index=True)
    has_passage: Mapped[bool] = mapped_column(Boolean)
    chunking_version: Mapped[str] = mapped_column(String(64))
    embedding_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_dimension: Mapped[int | None] = mapped_column(Integer, nullable=True)
    index_version: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utc_now, onupdate=utc_now
    )


class ParentChunkRecord(Base):
    __tablename__ = "parent_chunks"
    parent_id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE"), index=True
    )
    document_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_link: Mapped[str | None] = mapped_column(Text, nullable=True)
    chapter: Mapped[str | None] = mapped_column(String, nullable=True)
    section: Mapped[str | None] = mapped_column(String, nullable=True)
    article: Mapped[str | None] = mapped_column(String, nullable=True)
    position: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunking_method: Mapped[str] = mapped_column(String(32))


class ChildChunkRecord(Base):
    __tablename__ = "child_chunks"
    __table_args__ = (
        Index("ix_child_chunks_document_article", "document_id", "article"),
    )
    child_id: Mapped[str] = mapped_column(String, primary_key=True)
    parent_id: Mapped[str] = mapped_column(
        ForeignKey("parent_chunks.parent_id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE"), index=True
    )
    document_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_link: Mapped[str | None] = mapped_column(Text, nullable=True)
    chapter: Mapped[str | None] = mapped_column(String, nullable=True)
    section: Mapped[str | None] = mapped_column(String, nullable=True)
    article: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    clause: Mapped[str | None] = mapped_column(String, nullable=True)
    point: Mapped[str | None] = mapped_column(String, nullable=True)
    position: Mapped[int] = mapped_column(Integer, index=True)
    previous_child_id: Mapped[str | None] = mapped_column(String, nullable=True)
    next_child_id: Mapped[str | None] = mapped_column(String, nullable=True)
    text: Mapped[str] = mapped_column(Text)
    embedding_text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunking_method: Mapped[str] = mapped_column(String(32))


class IngestionJobRecord(Base):
    __tablename__ = "ingestion_jobs"
    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    index_version: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String(32))
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    documents_processed: Mapped[int] = mapped_column(Integer, default=0)
    documents_failed: Mapped[int] = mapped_column(Integer, default=0)
    chunks_created: Mapped[int] = mapped_column(Integer, default=0)
    children_created: Mapped[int] = mapped_column(Integer, default=0)


class IngestionCheckpointRecord(Base):
    __tablename__ = "ingestion_checkpoints"
    job_id: Mapped[str] = mapped_column(
        ForeignKey("ingestion_jobs.job_id"), primary_key=True
    )
    index_version: Mapped[str] = mapped_column(String)
    last_processed_document_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    last_processed_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    documents_processed: Mapped[int] = mapped_column(Integer, default=0)
    documents_failed: Mapped[int] = mapped_column(Integer, default=0)
    chunks_created: Mapped[int] = mapped_column(Integer, default=0)
    children_created: Mapped[int] = mapped_column(Integer, default=0)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    status: Mapped[str] = mapped_column(String(32))


class IngestionErrorRecord(Base):
    __tablename__ = "ingestion_errors"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String, index=True)
    document_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_path: Mapped[str] = mapped_column(Text)
    error_type: Mapped[str] = mapped_column(String(128))
    error_message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class IndexVersionRecord(Base):
    __tablename__ = "index_versions"
    index_version: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    manifest_path: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
