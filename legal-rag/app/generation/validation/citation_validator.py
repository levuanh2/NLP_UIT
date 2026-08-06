"""Citation validation skeleton."""

from app.domain.generation import GeneratedAnswer
from app.domain.retrieval import LegalContext


class CitationValidator:
    def validate(self, answer: GeneratedAnswer, context: LegalContext) -> bool:
        # TODO(phase-implementation):
        # Verify every citation resolves to supplied legal evidence.
        raise NotImplementedError
