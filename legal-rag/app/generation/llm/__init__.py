"""Public local language-model contracts and loaders."""

from app.generation.llm.base import BaseLLMGenerator
from app.generation.llm.qwen_generator import QwenGenerator

LocalLLM = QwenGenerator

__all__ = ["BaseLLMGenerator", "LocalLLM", "QwenGenerator"]
