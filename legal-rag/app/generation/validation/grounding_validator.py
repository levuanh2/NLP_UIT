"""Conservative lexical grounding validation."""

import re

from app.domain.generation import GeneratedAnswer
from app.domain.retrieval import LegalContext


class GroundingValidator:
    def validate(self, answer: GeneratedAnswer, context: LegalContext) -> bool:
        if not answer.answer.strip() or not context.evidences:
            return False
        tokens = _tokens(answer.answer)
        evidence_tokens = _tokens(context.formatted_context)
        if not tokens:
            return False
        return len(tokens & evidence_tokens) / len(tokens) >= 0.45


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[^\W_]+", text.casefold()) if len(token) > 2
    }
