"""Legal hierarchy extraction tests."""

from app.domain.documents import LegalDocument
from app.ingestion.structure.extractor import LegalStructureExtractor
from app.ingestion.structure.validator import LegalStructureValidator


def _document() -> LegalDocument:
    text = (
        "CHƯƠNG I\nMỤC 1\nĐIỀU 1. Phạm vi\n"
        "1. Nội dung khoản một.\na) Nội dung điểm a.\n"
        "ĐIỀU 2. Trách nhiệm\n1. Nội dung trách nhiệm."
    )
    return LegalDocument(
        document_id=1,
        document_name="Luật mẫu",
        source_link="https://example.test",
        raw_text=text,
        cleaned_text=text,
    )


def test_structure_extractor_extracts_chapter() -> None:
    result = LegalStructureExtractor().extract(_document())
    assert result.structure is not None
    assert result.structure.chapters[0].title == "CHƯƠNG I"


def test_structure_extractor_extracts_article_and_clause() -> None:
    result = LegalStructureExtractor().extract(_document())
    articles = [
        article
        for chapter in result.structure.chapters  # type: ignore[union-attr]
        for section in chapter.sections
        for article in section.articles
    ]
    assert [item.article_number for item in articles] == ["0", "1", "2"]
    assert articles[1].clauses[0].clause_number == "1"
    assert articles[1].clauses[0].points[0].point_label == "a"
    assert LegalStructureValidator().validate(result) == []
