"""Reciprocal-rank fusion."""

from app.domain.retrieval import RetrievalCandidate


class ReciprocalRankFusion:
    def __init__(self, rrf_k: int) -> None:
        if rrf_k <= 0:
            raise ValueError("RRF k must be positive.")
        self.rrf_k = rrf_k

    def fuse(
        self,
        ranked_lists: list[list[RetrievalCandidate]],
        top_n: int,
    ) -> list[RetrievalCandidate]:
        """Fuse dense and BM25 results."""
        if top_n <= 0:
            return []
        scores: dict[str, float] = {}
        candidates: dict[str, RetrievalCandidate] = {}
        first_seen: dict[str, int] = {}
        order = 0
        for ranked in ranked_lists:
            seen_in_list: set[str] = set()
            for rank, candidate in enumerate(ranked, start=1):
                if candidate.child_id in seen_in_list:
                    continue
                seen_in_list.add(candidate.child_id)
                scores[candidate.child_id] = scores.get(candidate.child_id, 0.0) + (
                    1.0 / (self.rrf_k + rank)
                )
                if candidate.child_id not in candidates:
                    candidates[candidate.child_id] = candidate
                    first_seen[candidate.child_id] = order
                    order += 1
                else:
                    current = candidates[candidate.child_id]
                    candidates[candidate.child_id] = current.model_copy(
                        update={
                            "dense_score": current.dense_score or candidate.dense_score,
                            "bm25_score": current.bm25_score or candidate.bm25_score,
                        }
                    )
        ranked_ids = sorted(
            scores,
            key=lambda item: (-scores[item], first_seen[item], item),
        )[:top_n]
        return [
            candidates[item].model_copy(update={"fusion_score": scores[item]})
            for item in ranked_ids
        ]
