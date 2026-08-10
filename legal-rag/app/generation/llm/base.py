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

    def count_tokens(self, text: str) -> int:
        """Count prompt tokens; test adapters may use this conservative fallback."""
        return len(text.split())

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""
