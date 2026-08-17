"""Structured, bounded legal prompt construction."""

from collections.abc import Callable

from pydantic import BaseModel

from app.domain.retrieval import LegalContext, LegalEvidence, RetrievalResult
from app.generation.prompts.system import LEGAL_SYSTEM_PROMPT
from app.retrieval.context.context_builder import LegalContextBuilder


class GroundedPrompt(BaseModel):
    prompt: str
    evidences: list[LegalEvidence]
    token_count: int
    context_token_count: int


class LegalPromptBuilder:
    def __init__(
        self,
        context_builder: LegalContextBuilder | None = None,
        max_context_tokens: int = 4096,
        reserved_generation_tokens: int = 192,
    ) -> None:
        self.context_builder = context_builder or LegalContextBuilder()
        self.max_context_tokens = max_context_tokens
        self.reserved_generation_tokens = reserved_generation_tokens

    def prepare(
        self,
        question: str,
        source: RetrievalResult | LegalContext,
        *,
        token_counter: Callable[[str], int] | None = None,
    ) -> GroundedPrompt:
        """Select complete evidence blocks and render a grounded prompt."""
        count = token_counter or (lambda text: len(text.split()))
        evidences = source.evidences
        empty_prompt = self._render(question, [])
        fixed_cost = count(empty_prompt)
        available = (
            self.max_context_tokens - self.reserved_generation_tokens - fixed_cost - 32
        )
        if available <= 0:
            raise RuntimeError("Configured model context leaves no room for evidence")

        def evidence_cost(evidence: LegalEvidence) -> int:
            # The index is display-only here; all metadata and legal text remain intact.
            return count(self._render_evidence(1, evidence)) + 2

        context = self.context_builder.build(
            question,
            evidences,
            max_tokens=available,
            token_cost=evidence_cost,
            truncate_first=False,
        )
        prompt = self._render(question, context.evidences)
        prompt_tokens = count(prompt)
        if prompt_tokens + self.reserved_generation_tokens > self.max_context_tokens:
            raise RuntimeError(
                "Grounded prompt exceeds configured model context window"
            )
        return GroundedPrompt(
            prompt=prompt,
            evidences=context.evidences,
            token_count=prompt_tokens,
            context_token_count=sum(
                count(self._render_evidence(index, evidence))
                for index, evidence in enumerate(context.evidences, start=1)
            ),
        )

    def build(self, question: str, context: RetrievalResult | LegalContext) -> str:
        """Backward-compatible prompt-only API."""
        return self.prepare(question, context).prompt

    def _render(self, question: str, evidences: list[LegalEvidence]) -> str:
        blocks = "\n\n".join(
            self._render_evidence(index, evidence)
            for index, evidence in enumerate(evidences, start=1)
        )
        # No "use citation [n]" instruction: the answer is scored on token
        # overlap with expert prose, which carries no bracket markers.
        answer_heading = "### Trả lời:"
        return (
            f"{LEGAL_SYSTEM_PROMPT}\n"
            f"CONTEXT:\n### Ngữ cảnh:\n{blocks}\n\n"
            f"QUESTION:\n### Câu hỏi:\n{question.strip()}\n\n"
            f"ANSWER:\n{answer_heading}\n"
        )

    @staticmethod
    def _render_evidence(index: int, evidence: LegalEvidence) -> str:
        fields = (
            ("Document ID", str(evidence.document_id)),
            ("Tên văn bản", evidence.document_name),
            ("Nguồn", evidence.source_link),
            ("Chương", evidence.chapter),
            ("Điều", evidence.article),
            ("Khoản", evidence.clause),
            ("Điểm", evidence.point),
            ("Child ID", evidence.child_id),
        )
        metadata = "\n".join(f"{label}: {value}" for label, value in fields if value)
        return (
            f"[{index}]\nDOCUMENT:\n"
            f"{metadata}\n\n"
            f"Nội dung:\n{evidence.text}\n[/{index}]"
        )
