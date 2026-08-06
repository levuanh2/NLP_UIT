"""SQLAlchemy persistence models; domain models remain infrastructure-free."""

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class ParentChunkRecord(Base):
    __tablename__ = "parent_chunks"
    parent_id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[int] = mapped_column(Integer, index=True)
    document_name: Mapped[str] = mapped_column(String)
    source_link: Mapped[str] = mapped_column(Text)
    chapter: Mapped[str | None] = mapped_column(String, nullable=True)
    section: Mapped[str | None] = mapped_column(String, nullable=True)
    article: Mapped[str | None] = mapped_column(String, nullable=True)
    clause: Mapped[str | None] = mapped_column(String, nullable=True)
    point: Mapped[str | None] = mapped_column(String, nullable=True)
    text: Mapped[str] = mapped_column(Text)


class ChildChunkRecord(Base):
    __tablename__ = "child_chunks"
    child_id: Mapped[str] = mapped_column(String, primary_key=True)
    parent_id: Mapped[str] = mapped_column(
        ForeignKey("parent_chunks.parent_id"), index=True
    )
    document_id: Mapped[int] = mapped_column(Integer, index=True)
    document_name: Mapped[str] = mapped_column(String)
    source_link: Mapped[str] = mapped_column(Text)
    chapter: Mapped[str | None] = mapped_column(String, nullable=True)
    section: Mapped[str | None] = mapped_column(String, nullable=True)
    article: Mapped[str | None] = mapped_column(String, nullable=True)
    clause: Mapped[str | None] = mapped_column(String, nullable=True)
    point: Mapped[str | None] = mapped_column(String, nullable=True)
    original_text: Mapped[str] = mapped_column(Text)
    embedding_text: Mapped[str | None] = mapped_column(Text, nullable=True)
