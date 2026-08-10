"""Deterministic legal-reference extraction tests."""

from app.retrieval.query_analyzer import QueryAnalyzer


def test_query_analyzer_article() -> None:
    metadata = QueryAnalyzer().analyze("Theo Điều 37 Luật Doanh nghiệp...")

    assert metadata.article == "Điều 37"
    assert metadata.document_name == "Luật Doanh nghiệp"


def test_query_analyzer_clause() -> None:
    metadata = QueryAnalyzer().analyze("Áp dụng khoản 2 Điều 5")

    assert metadata.clause == "Khoản 2"
    assert metadata.article == "Điều 5"


def test_query_analyzer_point() -> None:
    metadata = QueryAnalyzer().analyze("Theo điểm a khoản 1 Điều 3")

    assert metadata.point == "Điểm a"


def test_query_analyzer_does_not_confuse_dieu_kien_with_article() -> None:
    metadata = QueryAnalyzer().analyze("Điều kiện để doanh nghiệp được hoạt động")

    assert metadata.article is None
    assert metadata.document_name is None
