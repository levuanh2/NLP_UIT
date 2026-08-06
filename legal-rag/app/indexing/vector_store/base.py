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

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist vector index."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load vector index."""
