"""Vietnamese legal hierarchy extraction tests."""

from app.ingestion.structure.extractor import LegalStructureExtractor

TEXT = """Chương I
QUY ĐỊNH CHUNG
Mục 1
Phạm vi
Điều 1. Phạm vi điều chỉnh
1. Khoản thứ nhất.
a) Điểm thứ nhất.
b) Điểm thứ hai.
2. Khoản thứ hai.
Điều 2. Hiệu lực
Nội dung hiệu lực.
"""


def test_structure_extractor_chapter() -> None:
    structure = LegalStructureExtractor().extract(TEXT)
    assert structure.chapters[0].title.startswith("Chương I")


def test_structure_extractor_article() -> None:
    structure = LegalStructureExtractor().extract(TEXT)
    articles = structure.chapters[0].sections[0].articles
    assert [article.article_number for article in articles] == ["1", "2"]


def test_structure_extractor_clause() -> None:
    structure = LegalStructureExtractor().extract(TEXT)
    article = structure.chapters[0].sections[0].articles[0]
    assert [clause.clause_number for clause in article.clauses] == ["1", "2"]


def test_structure_extractor_point() -> None:
    structure = LegalStructureExtractor().extract(TEXT)
    clause = structure.chapters[0].sections[0].articles[0].clauses[0]
    assert [point.point_label for point in clause.points] == ["a", "b"]
