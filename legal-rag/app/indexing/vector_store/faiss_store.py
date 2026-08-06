"""FAISS vector-store skeleton with lazy infrastructure loading."""

from pathlib import Path
from typing import Any

import numpy as np

from app.indexing.vector_store.base import BaseVectorStore


class FAISSVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self._index: Any | None = None
        self._ids: list[str] = []

    def create(self, dimension: int) -> None:
        # TODO(phase-implementation):
        # Lazily import FAISS and construct the configured local index.
        raise NotImplementedError

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        # TODO(phase-implementation):
        # Validate aligned vectors/IDs and add only vectors plus the external
        # child-ID mapping to FAISS; legal metadata remains in SQLite.
        raise NotImplementedError

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        # TODO(phase-implementation):
        # Search FAISS and enforce optional metadata-filtered IDs.
        raise NotImplementedError

    def save(self, path: Path) -> None:
        # TODO(phase-implementation):
        # Atomically persist FAISS data and the ID mapping.
        raise NotImplementedError

    def load(self, path: Path) -> None:
        # TODO(phase-implementation):
        # Load local FAISS data and validate its ID mapping.
        raise NotImplementedError
