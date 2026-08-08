"""Vector store contract."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseVectorStore(ABC):
    @abstractmethod
    def create(self, dimension: int) -> None:
        """Create a vector index."""

    @abstractmethod
    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        """Add vectors to index."""

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Search vectors."""

    def search_many(
        self,
        query_vectors: np.ndarray,
        top_k: int,
        allowed_ids: list[set[str] | None],
    ) -> list[list[tuple[str, float]]]:
        """Search multiple vectors; implementations may override for batching."""
        if len(query_vectors) != len(allowed_ids):
            raise ValueError("Query vectors and allowed-ID sets must be aligned.")
        return [
            self.search(vector, top_k, allowed)
            for vector, allowed in zip(query_vectors, allowed_ids, strict=True)
        ]

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist vector index."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load vector index."""
