"""Legal context document and extracted hierarchy models."""

from pydantic import BaseModel, Field


class LegalPoint(BaseModel):
    point_id: str
    point_label: str
    text: str
    position: int = 0


class LegalClause(BaseModel):
    clause_id: str
    clause_number: str
    text: str
    points: list[LegalPoint] = Field(default_factory=list)
    position: int = 0


class LegalArticle(BaseModel):
    article_id: str
    article_number: str
    title: str | None
    clauses: list[LegalClause] = Field(default_factory=list)
    text: str = ""
    position: int = 0


class LegalSection(BaseModel):
    section_id: str
    title: str
    articles: list[LegalArticle] = Field(default_factory=list)
    position: int = 0


class LegalChapter(BaseModel):
    chapter_id: str
    title: str
    sections: list[LegalSection] = Field(default_factory=list)
    articles: list[LegalArticle] = Field(default_factory=list)
    position: int = 0


class LegalStructure(BaseModel):
    """Extracted legal hierarchy for one context passage."""

    chapters: list[LegalChapter] = Field(default_factory=list)
    sections: list[LegalSection] = Field(default_factory=list)
    articles: list[LegalArticle] = Field(default_factory=list)


class LegalDocument(BaseModel):
    document_id: int
    document_name: str | None = None
    source_link: str | None = None
    raw_text: str
    cleaned_text: str | None = None
    structure: LegalStructure | None = None
