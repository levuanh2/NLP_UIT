"""Local Vietnamese legal embedding loader."""

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
        batch_size: int = 128,
        max_sequence_length: int = 128,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.local_files_only = local_files_only
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self._model: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required.") from exc
        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            local_files_only=self.local_files_only,
        )
        self._model.max_seq_length = self.max_sequence_length

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            self.load()
        if not texts:
            return np.empty((0, self.dimension()), dtype=np.float32)
        values = self._model.encode(
            [f"{self.passage_prefix}{text}" for text in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(values, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_queries([query])[0]

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """Embed queries in GPU-friendly batches."""
        if self._model is None:
            self.load()
        if not queries:
            return np.empty((0, self.dimension()), dtype=np.float32)
        values = self._model.encode(
            [f"{self.query_prefix}{query}" for query in queries],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(values, dtype=np.float32)

    def dimension(self) -> int:
        if self._model is None:
            self.load()
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise RuntimeError("Embedding model did not report a dimension.")
        return int(dimension)

    def unload(self) -> None:
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
