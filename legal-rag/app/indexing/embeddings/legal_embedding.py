"""Local Vietnamese legal embedding loader skeleton."""

import gc
import os
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
        if self.local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            local_files_only=self.local_files_only,
            trust_remote_code=False,
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        model = self._require_model()
        return np.asarray(
            model.encode(
                [f"{self.passage_prefix}{text}" for text in texts],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def embed_query(self, query: str) -> np.ndarray:
        model = self._require_model()
        vector = model.encode(
            f"{self.query_prefix}{query}",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype=np.float32)

    def dimension(self) -> int:
        return int(self._require_model().get_sentence_embedding_dimension())

    def unload(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _require_model(self) -> Any:
        if self._model is None:
            raise RuntimeError("Embedding model has not been loaded")
        return self._model
