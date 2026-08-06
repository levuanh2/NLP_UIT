"""Reciprocal-rank fusion skeleton."""

from app.domain.retrieval import RetrievalCandidate


class ReciprocalRankFusion:
    def __init__(self, rrf_k: int) -> None:
        self.rrf_k = rrf_k

    def fuse(
        self,
        ranked_lists: list[list[RetrievalCandidate]],
        top_n: int,
    ) -> list[RetrievalCandidate]:
        """Fuse dense and BM25 results."""
        # TODO(phase-implementation):
        # Implement RRF with deterministic deduplication and tie ordering.
        raise NotImplementedError
