"""Insufficient-evidence abstention skeleton."""

from app.domain.retrieval import LegalContext


class AbstentionValidator:
    def __init__(self, abstention_message: str) -> None:
        self.abstention_message = abstention_message

    def should_abstain(self, context: LegalContext) -> bool:
        # TODO(phase-implementation):
        # Define evidence sufficiency policy without external knowledge.
        raise NotImplementedError
