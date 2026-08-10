"""Legal evidence context builder skeleton."""

from collections.abc import Callable

from app.domain.retrieval import LegalContext, LegalEvidence
from app.retrieval.context.context_budget import ContextBudgetManager


class LegalContextBuilder:
    def __init__(
        self,
        budget_manager: ContextBudgetManager | None = None,
        max_tokens: int = 6000,
    ) -> None:
        self.budget_manager = budget_manager or ContextBudgetManager()
        self.max_tokens = max_tokens

    def build(
        self,
        query: str,
        evidences: list[LegalEvidence],
        *,
        max_tokens: int | None = None,
        token_cost: Callable[[LegalEvidence], int] | None = None,
        truncate_first: bool = True,
    ) -> LegalContext:
        """Format legal evidence for the LLM."""
        selected = self.budget_manager.fit(
            evidences,
            max_tokens if max_tokens is not None else self.max_tokens,
            token_cost=token_cost,
            truncate_first=truncate_first,
        )
        rendered: list[str] = []
        for evidence in selected:
            hierarchy = " > ".join(
                value
                for value in (
                    evidence.document_name,
                    evidence.chapter,
                    evidence.section,
                    evidence.article,
                    evidence.clause,
                    evidence.point,
                )
                if value
            )
            rendered.append(
                f"[{evidence.evidence_id}] {hierarchy}\n{evidence.text}".strip()
            )
        formatted = "\n\n".join(rendered)
        return LegalContext(
            query=query,
            evidences=selected,
            formatted_context=formatted,
            token_count=len(formatted.split()),
        )
