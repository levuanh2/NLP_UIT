"""Local Vi-Qwen causal language-model generator."""

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
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(self.dtype)
        if dtype is None:
            raise ValueError(f"Unsupported model dtype: {self.dtype}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=dtype,
        )
        self._model.to(self._resolved_device())
        self._model.eval()

    def generate(self, prompt: str) -> str:
        if self._model is None or self._tokenizer is None:
            self.load()
        import torch

        if hasattr(self._tokenizer, "apply_chat_template"):
            input_text = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            input_text = prompt
        encoded = self._tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        encoded = {
            key: value.to(self._resolved_device()) for key, value in encoded.items()
        }
        options: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if self.do_sample:
            options.update(temperature=self.temperature, top_p=self.top_p)
        with torch.inference_mode():
            output = self._model.generate(**encoded, **options)
        prompt_length = encoded["input_ids"].shape[1]
        return self._tokenizer.decode(
            output[0, prompt_length:], skip_special_tokens=True
        ).strip()

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
