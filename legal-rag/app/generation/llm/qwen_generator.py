"""Local Vi-Qwen generator skeleton."""

from typing import Any

from app.generation.llm.base import BaseLLMGenerator


class QwenGenerator(BaseLLMGenerator):
    def __init__(
        self,
        model_name: str,
        device: str,
        dtype: str,
        local_files_only: bool,
        trust_remote_code: bool,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def load(self) -> None:
        # TODO(phase-implementation):
        # Lazily load configured Vi-Qwen model/tokenizer from local files only.
        raise NotImplementedError

    def generate(self, prompt: str) -> str:
        # TODO(phase-implementation):
        # Generate with configured decoding and no external service calls.
        raise NotImplementedError

    def unload(self) -> None:
        # TODO(phase-implementation):
        # Release model resources and device memory.
        raise NotImplementedError
