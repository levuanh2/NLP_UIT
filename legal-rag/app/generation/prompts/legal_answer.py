"""Legal answer prompt builder skeleton."""

from app.domain.retrieval import LegalContext


class LegalPromptBuilder:
    def build(self, question: str, context: LegalContext) -> str:
        """Build system instruction, legal evidence, and user question."""
        # TODO(phase-implementation):
        # Render bounded evidence and explicit response instructions.
        raise NotImplementedError
