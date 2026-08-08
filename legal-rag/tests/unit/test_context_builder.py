"""Legal context rendering tests."""

from app.domain.retrieval import LegalEvidence
from app.retrieval.context.context_builder import (
    LegalContextBuilder,
    _display_document_name,
)


def test_context_builder_preserves_legal_metadata() -> None:
    evidence = LegalEvidence(
        evidence_id="parent-1",
        document_id=1,
        document_name="Nghị định 1/2026/NĐ-CP",
        source_link="https://example.test",
        chapter="Chương I",
        section="Mục 1",
        article="7",
        clause="2",
        point="a",
        text="Nội dung căn cứ.",
    )
    context = LegalContextBuilder().build("Mức phạt?", [evidence])
    assert "Điều 7, khoản 2, điểm a" in context.formatted_context
    assert "parent-1" in context.formatted_context
    assert context.token_count


def test_display_document_name_normalizes_legal_slug() -> None:
    assert (
        _display_document_name(
            "Nghi-dinh-01-2021-ND-CP-dang-ky-doanh-nghiep-283247"
        )
        == "Nghị định 01/2021/NĐ-CP"
    )
