"""Bounded context selection that preserves evidence/citation boundaries."""

from collections.abc import Callable

from app.domain.retrieval import LegalEvidence


class ContextBudgetManager:
    def fit(
        self,
        evidences: list[LegalEvidence],
        max_tokens: int,
        *,
        token_cost: Callable[[LegalEvidence], int] | None = None,
        truncate_first: bool = True,
    ) -> list[LegalEvidence]:
        """Select ranked evidence without exceeding an approximate token budget."""
        if max_tokens <= 0:
            return []
        selected: list[LegalEvidence] = []
        used = 0
        for evidence in evidences:
            token_count = (
                token_cost(evidence)
                if token_cost is not None
                else len(evidence.text.split())
            )
            if used + token_count <= max_tokens:
                selected.append(evidence)
                used += token_count
                continue
            if not selected and truncate_first and token_cost is None:
                words = evidence.text.split()
                selected.append(
                    evidence.model_copy(update={"text": " ".join(words[:max_tokens])})
                )
            break
        return selected
