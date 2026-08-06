"""Local Vietnamese reranker skeleton."""

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
        # TODO(phase-implementation):
        # Lazily load tokenizer/model from local files only.
        raise NotImplementedError

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        # TODO(phase-implementation):
        # Score query/chunk pairs and return the configured top-k.
        raise NotImplementedError

    def unload(self) -> None:
        # TODO(phase-implementation):
        # Release reranker resources and device memory.
        raise NotImplementedError
