"""Deterministic reciprocal-rank fusion by stable child ID."""

from app.domain.retrieval import RetrievalCandidate


class RRFFusion:
    def fuse(
        self,
        dense_results: list[RetrievalCandidate],
        bm25_results: list[RetrievalCandidate],
        *,
        k: int = 60,
        top_k: int = 30,
    ) -> list[RetrievalCandidate]:
        if k <= 0:
            raise ValueError("RRF k must be positive")
        if top_k <= 0:
            return []
        scores: dict[str, float] = {}
        details: dict[str, dict[str, float]] = {}
        for results, score_field in (
            (dense_results, "dense_score"),
            (bm25_results, "bm25_score"),
        ):
            seen: set[str] = set()
            for position, candidate in enumerate(results, start=1):
                if candidate.child_id in seen:
                    continue
                seen.add(candidate.child_id)
                rank = candidate.rank if candidate.rank > 0 else position
                scores[candidate.child_id] = scores.get(candidate.child_id, 0.0) + (
                    1.0 / (k + rank)
                )
                raw_score = (
                    candidate.dense_score
                    if score_field == "dense_score"
                    else candidate.bm25_score
                )
                if raw_score is None:
                    raw_score = candidate.score
                details.setdefault(candidate.child_id, {})[score_field] = raw_score
        ordered = sorted(scores, key=lambda child_id: (-scores[child_id], child_id))[
            :top_k
        ]
        return [
            RetrievalCandidate(
                child_id=child_id,
                score=scores[child_id],
                source="rrf",
                rank=rank,
                fusion_score=scores[child_id],
                **details.get(child_id, {}),
            )
            for rank, child_id in enumerate(ordered, start=1)
        ]


class ReciprocalRankFusion(RRFFusion):
    """Backward-compatible adapter for the earlier composition contract."""

    def __init__(self, rrf_k: int = 60) -> None:
        self.rrf_k = rrf_k

    def fuse(  # type: ignore[override]
        self,
        ranked_lists: list[list[RetrievalCandidate]],
        top_n: int,
    ) -> list[RetrievalCandidate]:
        dense = ranked_lists[0] if ranked_lists else []
        bm25 = ranked_lists[1] if len(ranked_lists) > 1 else []
        return super().fuse(dense, bm25, k=self.rrf_k, top_k=top_n)
