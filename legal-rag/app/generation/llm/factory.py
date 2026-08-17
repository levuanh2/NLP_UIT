"""LLM generator factory."""

from app.generation.llm.base import BaseLLMGenerator
from app.generation.llm.qwen_generator import QwenGenerator


class LLMGeneratorFactory:
    @staticmethod
    def create(
        provider: str,
        model_name: str,
        device: str,
        dtype: str,
        local_files_only: bool,
        trust_remote_code: bool,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        min_new_tokens: int,
        repetition_penalty: float,
        quantization: str = "none",
    ) -> BaseLLMGenerator:
        if provider != "local_transformers":
            raise ValueError(f"Unsupported LLM provider: {provider}")
        return QwenGenerator(
            model_name=model_name,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            quantization=quantization,
        )
