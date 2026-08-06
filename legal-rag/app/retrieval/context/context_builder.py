"""Legal evidence context builder skeleton."""

from app.domain.retrieval import LegalContext, LegalEvidence


class LegalContextBuilder:
    def build(self, query: str, evidences: list[LegalEvidence]) -> LegalContext:
        """Format legal evidence for the LLM."""
        # TODO(phase-implementation):
        # Render stable evidence labels, metadata, and token accounting.
        raise NotImplementedError
