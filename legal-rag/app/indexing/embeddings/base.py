"""Embedding model contract."""

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def load(self) -> None:
        """Load embedding model locally."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed legal document chunks."""

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed one legal query."""

    @abstractmethod
    def dimension(self) -> int:
        """Return embedding vector dimension."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""
