"""Parent-child chunk models."""

from pydantic import BaseModel

from app.domain.metadata import LegalMetadata


class ParentChunk(BaseModel):
    parent_id: str
    document_id: int
    chapter: str | None
    section: str | None
    article: str | None
    text: str
    token_count: int | None


class ChildChunk(BaseModel):
    child_id: str
    parent_id: str
    document_id: int
    chapter: str | None
    section: str | None
    article: str | None
    clause: str | None
    point: str | None
    original_text: str
    embedding_text: str | None
    token_count: int | None
    metadata: LegalMetadata
