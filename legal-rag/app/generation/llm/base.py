"""LLM generator contract."""

from abc import ABC, abstractmethod


class BaseLLMGenerator(ABC):
    @abstractmethod
    def load(self) -> None:
        """Load local LLM."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate legal answer."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""
