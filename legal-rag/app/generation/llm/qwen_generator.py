"""Reusable local Transformers adapter for the approved Vi-Qwen model."""

import gc
import os
import time
from typing import Any

from app.core.exceptions import ConfigurationError, ModelNotLoadedError
from app.domain.generation import GenerationMetrics
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
        min_new_tokens: int,
        repetition_penalty: float,
        quantization: str = "none",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.quantization = quantization
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.min_new_tokens = min_new_tokens
        self.repetition_penalty = repetition_penalty
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._resolved_device: str | None = None
        self.last_generation_metrics: GenerationMetrics | None = None
        # Ephemeral diagnostics for an explicitly traced local inference. This
        # is never serialized into GeneratedAnswer/submission artifacts.
        self.debug_enabled = False
        self.last_generation_debug: dict[str, Any] | None = None

    def load(self) -> None:
        """Load tokenizer/model once from an existing local path or HF cache."""
        if self._model is not None and self._tokenizer is not None:
            return
        if not self.local_files_only:
            raise ConfigurationError("Local LLM requires local_files_only=true")
        if self.trust_remote_code:
            raise ConfigurationError("Remote model code is disabled")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._resolved_device = self._resolve_device(torch)
        torch_dtype = self._resolve_dtype(torch)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=True,
                trust_remote_code=False,
            )
            if not getattr(self._tokenizer, "chat_template", None):
                raise ConfigurationError(
                    f"Tokenizer for {self.model_name!r} has no chat template"
                )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=torch_dtype,
                **self._quantization_options(torch),
            )
        except Exception as exc:
            self._model = None
            self._tokenizer = None
            raise ConfigurationError(
                f"Could not load approved local LLM {self.model_name!r}: {exc}"
            ) from exc
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if self.quantization == "none":
            self._model.to(self._resolved_device)
        self._model.eval()

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        model, tokenizer = self._require_loaded()
        import torch

        requested_tokens = max_new_tokens or self.max_new_tokens
        requested_temperature = self.temperature if temperature is None else temperature
        if requested_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        # Vi-Qwen2-RAG is instruction-tuned with Qwen's chat template.  Feeding
        # the rendered legal prompt as bare completion text puts the model
        # outside its documented inference format and materially degrades
        # grounding/citation instruction following.
        tokenize_started = time.perf_counter()
        messages = self._conversation(prompt)
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(chat_prompt, return_tensors="pt", truncation=False)
        tokenize_seconds = time.perf_counter() - tokenize_started
        input_tokens = int(encoded["input_ids"].shape[-1])
        model_limit = self._model_context_limit(model, tokenizer)
        if input_tokens + requested_tokens > model_limit:
            raise RuntimeError(
                "Grounded prompt exceeds local model context window: "
                f"prompt={input_tokens}, generation={requested_tokens}, "
                f"limit={model_limit}"
            )
        encoded = {
            key: value.to(self._resolved_device) for key, value in encoded.items()
        }
        generation_options: dict[str, Any] = {
            "max_new_tokens": requested_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "min_new_tokens": self.min_new_tokens,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.do_sample:
            generation_options.update(
                temperature=requested_temperature,
                top_p=self.top_p,
            )
        generation_started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**encoded, **generation_options)
        generation_seconds = time.perf_counter() - generation_started
        generated_ids = output[0, input_tokens:]
        generated_tokens = int(generated_ids.shape[-1])
        decode_started = time.perf_counter()
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        decode_seconds = time.perf_counter() - decode_started
        self.last_generation_debug = (
            {
                "source_prompt": prompt,
                "chat_messages": messages,
                "rendered_chat_prompt": chat_prompt,
                "chat_template_used": True,
                "input_tokens": input_tokens,
                "raw_output_token_ids": generated_ids.detach().cpu().tolist(),
                "decoded_output": answer,
            }
            if self.debug_enabled
            else None
        )
        self.last_generation_metrics = GenerationMetrics(
            input_tokens=input_tokens,
            max_new_tokens=requested_tokens,
            generated_tokens=generated_tokens,
            tokenize_seconds=tokenize_seconds,
            generation_seconds=generation_seconds,
            decode_seconds=decode_seconds,
            tokens_per_second=(
                generated_tokens / generation_seconds if generation_seconds else 0.0
            ),
        )
        if not answer:
            raise RuntimeError("Local LLM returned an empty answer")
        return answer

    @staticmethod
    def _conversation(prompt: str) -> list[dict[str, str]]:
        """Map the canonical grounded prompt onto Vi-Qwen's documented roles."""
        if prompt.startswith("SYSTEM:\n"):
            for boundary, user_heading in (
                ("\nCONTEXT:\n", "CONTEXT:\n"),
                ("\n### Câu hỏi\n", "### Câu hỏi\n"),
            ):
                if boundary in prompt:
                    system, user = prompt.split(boundary, maxsplit=1)
                    return [
                        {
                            "role": "system",
                            "content": system.removeprefix("SYSTEM:\n"),
                        },
                        {"role": "user", "content": f"{user_heading}{user}"},
                    ]
        return [{"role": "user", "content": prompt}]

    def count_tokens(self, text: str) -> int:
        _, tokenizer = self._require_loaded()
        return len(tokenizer.encode(text, add_special_tokens=True))

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._resolved_device = None
        self.last_generation_metrics = None
        self.last_generation_debug = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _require_loaded(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            raise ModelNotLoadedError(
                f"Approved local LLM {self.model_name!r} has not been loaded"
            )
        return self._model, self._tokenizer

    def _resolve_device(self, torch: Any) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ConfigurationError("CUDA was configured but is not available")
        return self.device

    def _quantization_options(self, torch: Any) -> dict[str, Any]:
        """Fit the 3B model in 6 GB of VRAM; fp16 weights leave no room for KV cache."""
        if self.quantization == "none":
            return {}
        if self.quantization != "nf4":
            raise ConfigurationError(
                f"Unsupported MODEL_QUANTIZATION: {self.quantization}"
            )
        if self._resolved_device != "cuda":
            raise ConfigurationError("NF4 quantization requires a CUDA device")
        from transformers import BitsAndBytesConfig

        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
            "device_map": {"": 0},
        }

    def _resolve_dtype(self, torch: Any) -> Any:
        if self.dtype == "auto":
            return "auto"
        values = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if self.dtype not in values:
            raise ConfigurationError(f"Unsupported MODEL_DTYPE: {self.dtype}")
        return values[self.dtype]

    @staticmethod
    def _model_context_limit(model: Any, tokenizer: Any) -> int:
        configured = getattr(model.config, "max_position_embeddings", None)
        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        limits = [
            int(value)
            for value in (configured, tokenizer_limit)
            if isinstance(value, int) and 0 < value < 1_000_000
        ]
        if not limits:
            raise ConfigurationError("Local LLM context window is not configured")
        return min(limits)
