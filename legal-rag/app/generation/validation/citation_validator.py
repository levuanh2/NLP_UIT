"""Citation validation for generated legal answers."""

from app.domain.generation import GeneratedAnswer
from app.domain.retrieval import LegalContext


class CitationValidator:
    def validate(self, answer: GeneratedAnswer, context: LegalContext) -> bool:
        available = {item.evidence_id for item in context.evidences}
        return all(
            citation.evidence_id is not None and citation.evidence_id in available
            for citation in answer.citations
        ) and set(answer.evidence_ids).issubset(available)
