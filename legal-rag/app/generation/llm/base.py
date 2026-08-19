"""LLM generator contract."""

from abc import ABC, abstractmethod


class BaseLLMGenerator(ABC):
    @abstractmethod
    def load(self) -> None:
        """Load local LLM."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate legal answer."""

    def generate_many(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[str]:
        """Answer several prompts. One at a time unless a backend pools them."""
        return [
            self.generate(
                prompt, max_new_tokens=max_new_tokens, temperature=temperature
            )
            for prompt in prompts
        ]

    def count_tokens(self, text: str) -> int:
        """Count prompt tokens; test adapters may use this conservative fallback."""
        return len(text.split())

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""
