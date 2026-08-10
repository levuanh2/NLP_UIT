"""Parent-child chunk models with stable hierarchy and adjacency metadata."""

from pydantic import AliasChoices, BaseModel, Field

from app.domain.metadata import LegalMetadata


class ParentChunk(BaseModel):
    parent_id: str
    document_id: int
    document_name: str | None = None
    source_link: str | None = None
    chapter: str | None
    section: str | None
    article: str | None
    position: int = 0
    text: str
    token_count: int | None
    chunking_method: str = "article"


class ChildChunk(BaseModel):
    child_id: str
    parent_id: str
    document_id: int
    document_name: str | None = None
    source_link: str | None = None
    chapter: str | None
    section: str | None
    article: str | None
    clause: str | None
    point: str | None
    position: int = 0
    previous_child_id: str | None = None
    next_child_id: str | None = None
    text: str = Field(validation_alias=AliasChoices("text", "original_text"))
    embedding_text: str
    token_count: int | None
    chunking_method: str = "structured"
    metadata: LegalMetadata | None = None

    @property
    def original_text(self) -> str:
        """Backward-compatible access for the original scaffold contract."""
        return self.text
