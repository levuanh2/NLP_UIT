"""Local Vietnamese reranker."""

from typing import Any

from app.domain.retrieval import RetrievalCandidate
from app.retrieval.reranking.base import BaseReranker


class VietnameseReranker(BaseReranker):
    def __init__(
        self,
        model_name: str,
        device: str,
        local_files_only: bool,
        trust_remote_code: bool,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for reranking.") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        self._model.to(self._resolved_device())
        self._model.eval()

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        if top_k <= 0 or not candidates:
            return []
        if self._model is None or self._tokenizer is None:
            self.load()
        import torch

        scores: list[float] = []
        for start in range(0, len(candidates), 8):
            batch = candidates[start : start + 8]
            encoded = self._tokenizer(
                [query] * len(batch),
                [item.text for item in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self._resolved_device()) for key, value in encoded.items()
            }
            with torch.inference_mode():
                logits = self._model(**encoded).logits
            if logits.shape[-1] == 1:
                batch_scores = torch.sigmoid(logits[:, 0])
            else:
                batch_scores = torch.softmax(logits, dim=-1)[:, -1]
            scores.extend(float(value) for value in batch_scores.cpu())
        ranked = sorted(
            (
                candidate.model_copy(update={"rerank_score": scores[index]})
                for index, candidate in enumerate(candidates)
            ),
            key=lambda item: (-(item.rerank_score or 0.0), item.child_id),
        )
        return ranked[:top_k]

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
