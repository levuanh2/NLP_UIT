"""Grounding validation skeleton."""

from app.domain.generation import GeneratedAnswer
from app.domain.retrieval import LegalContext


class GroundingValidator:
    def validate(self, answer: GeneratedAnswer, context: LegalContext) -> bool:
        # TODO(phase-implementation):
        # Check that legal claims are supported by supplied evidence.
        raise NotImplementedError
