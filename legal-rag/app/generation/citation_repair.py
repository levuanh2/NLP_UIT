"""Bounded local-LLM citation repair without citation injection."""

from app.domain.retrieval import RetrievalResult
from app.generation.llm.base import BaseLLMGenerator
from app.generation.prompts.citation_repair import CitationRepairPromptBuilder


class CitationRepair:
    MAX_REPAIR_NEW_TOKENS = 192

    def __init__(
        self,
        generator: BaseLLMGenerator,
        prompt_builder: CitationRepairPromptBuilder | None = None,
        *,
        enabled: bool = True,
        max_retries: int = 1,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        max_context_tokens: int = 4096,
    ) -> None:
        if max_retries < 0 or max_retries > 1:
            raise ValueError("citation repair max_retries must be between 0 and 1")
        self.generator = generator
        self.prompt_builder = prompt_builder or CitationRepairPromptBuilder()
        self.enabled = enabled
        self.max_retries = max_retries
        # Repair is a formatting correction, not a second unconstrained answer.
        # Keep this optional CPU-heavy formatting pass below the concise answer
        # budget while preserving enough room for a short supported response.
        self.max_new_tokens = min(max_new_tokens, self.MAX_REPAIR_NEW_TOKENS)
        self.temperature = temperature
        self.max_context_tokens = max_context_tokens

    def repair(
        self,
        *,
        question: str,
        answer: str,
        retrieval_result: RetrievalResult,
    ) -> str:
        """Run at most one additional local generation using the same evidence."""
        if not self.enabled or self.max_retries == 0:
            return answer
        if not retrieval_result.evidences:
            return answer
        repaired = answer
        for _ in range(self.max_retries):
            prompt = self.prompt_builder.build(
                question=question,
                answer=repaired,
                evidences=retrieval_result.evidences,
            )
            prompt_tokens = self.generator.count_tokens(prompt)
            if prompt_tokens + self.max_new_tokens > self.max_context_tokens:
                raise RuntimeError(
                    "Citation repair prompt exceeds configured model context window: "
                    f"prompt={prompt_tokens}, generation={self.max_new_tokens}, "
                    f"limit={self.max_context_tokens}"
                )
            repaired = self.generator.generate(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            ).strip()
            if not repaired:
                raise RuntimeError("Local LLM returned an empty citation repair")
        return repaired
