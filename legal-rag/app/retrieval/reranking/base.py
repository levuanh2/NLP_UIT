"""Reranker contract."""

from abc import ABC, abstractmethod

from app.domain.retrieval import RetrievalCandidate


class BaseReranker(ABC):
    @abstractmethod
    def load(self) -> None:
        """Load reranker locally."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        """Rerank retrieval candidates."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""
