"""Local Vietnamese legal embedding loader skeleton."""

from typing import Any

import numpy as np

from app.indexing.embeddings.base import BaseEmbeddingModel


class VietnameseLegalEmbeddingModel(BaseEmbeddingModel):
    def __init__(
        self,
        model_name: str,
        device: str,
        local_files_only: bool,
        query_prefix: str,
        passage_prefix: str,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.local_files_only = local_files_only
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self._model: Any | None = None

    def load(self) -> None:
        # TODO(phase-implementation):
        # Lazily import SentenceTransformer and load only local model files.
        raise NotImplementedError

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        # TODO(phase-implementation):
        # Generate normalized passage embeddings with the configured prefix.
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        # TODO(phase-implementation):
        # Generate one normalized query embedding.
        raise NotImplementedError

    def dimension(self) -> int:
        # TODO(phase-implementation):
        # Read dimension from the loaded local model.
        raise NotImplementedError

    def unload(self) -> None:
        # TODO(phase-implementation):
        # Release local model resources and device memory.
        raise NotImplementedError
