"""Insufficient-evidence abstention validation."""

from app.domain.retrieval import LegalContext


class AbstentionValidator:
    def __init__(self, abstention_message: str) -> None:
        self.abstention_message = abstention_message

    def should_abstain(self, context: LegalContext) -> bool:
        return not context.evidences or not context.formatted_context.strip()
