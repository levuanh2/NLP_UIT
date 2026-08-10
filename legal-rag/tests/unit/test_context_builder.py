"""Context budget and rendering tests."""

from app.domain.retrieval import LegalEvidence
from app.retrieval.context.context_budget import ContextBudgetManager
from app.retrieval.context.context_builder import LegalContextBuilder


def evidence(identifier: str, words: int) -> LegalEvidence:
    return LegalEvidence(
        evidence_id=identifier,
        document_id=1,
        document_name="Luật mẫu",
        source_link="https://example.test",
        chapter="Chương I",
        section=None,
        article="Điều 1",
        clause=None,
        point=None,
        text=" ".join([identifier] * words),
    )


def test_context_budget_preserves_evidence_boundaries() -> None:
    fitted = ContextBudgetManager().fit(
        [evidence("a", 4), evidence("b", 4), evidence("c", 4)], 8
    )

    assert [item.evidence_id for item in fitted] == ["a", "b"]
    assert sum(len(item.text.split()) for item in fitted) <= 8


def test_context_builder_preserves_legal_metadata() -> None:
    context = LegalContextBuilder(max_tokens=10).build("query", [evidence("e1", 3)])

    assert "Luật mẫu > Chương I > Điều 1" in context.formatted_context
    assert "[e1]" in context.formatted_context
