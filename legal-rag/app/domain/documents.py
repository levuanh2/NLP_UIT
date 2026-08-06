"""Legal context document and extracted hierarchy models."""

from pydantic import BaseModel, Field


class LegalPoint(BaseModel):
    point_id: str
    point_label: str
    text: str


class LegalClause(BaseModel):
    clause_id: str
    clause_number: str
    text: str
    points: list[LegalPoint]


class LegalArticle(BaseModel):
    article_id: str
    article_number: str
    title: str | None
    clauses: list[LegalClause]


class LegalSection(BaseModel):
    section_id: str
    title: str
    articles: list[LegalArticle]


class LegalChapter(BaseModel):
    chapter_id: str
    title: str
    sections: list[LegalSection]


class LegalStructure(BaseModel):
    """Extracted legal hierarchy for one context passage."""

    chapters: list[LegalChapter] = Field(default_factory=list)


class LegalDocument(BaseModel):
    document_id: int
    document_name: str
    source_link: str
    raw_text: str
    cleaned_text: str | None = None
    structure: LegalStructure | None = None
