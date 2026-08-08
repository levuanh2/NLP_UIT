"""Deterministic no-model reranker for explicitly disabled reranking."""

from app.domain.retrieval import RetrievalCandidate
from app.retrieval.reranking.base import BaseReranker


class IdentityReranker(BaseReranker):
    def load(self) -> None:
        return None

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        del query
        return candidates[: max(0, top_k)]

    def unload(self) -> None:
        return None
