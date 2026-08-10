"""Legal metadata originating from context JSON and structure extraction."""

from pydantic import BaseModel


class LegalMetadata(BaseModel):
    document_id: int
    document_name: str | None = None
    source_link: str | None = None
    chapter: str | None
    section: str | None
    article: str | None
    clause: str | None
    point: str | None
