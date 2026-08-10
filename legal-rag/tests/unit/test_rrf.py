"""Reciprocal-rank fusion tests."""

from app.domain.retrieval import RetrievalCandidate
from app.retrieval.fusion.rrf import RRFFusion


def candidate(child_id: str, source: str, rank: int) -> RetrievalCandidate:
    return RetrievalCandidate(
        child_id=child_id,
        score=1.0 / rank,
        source=source,  # type: ignore[arg-type]
        rank=rank,
    )


def test_rrf_merge() -> None:
    fused = RRFFusion().fuse(
        [candidate("a", "dense", 1), candidate("b", "dense", 2)],
        [candidate("b", "bm25", 1), candidate("c", "bm25", 2)],
    )

    assert [item.child_id for item in fused] == ["b", "a", "c"]


def test_rrf_duplicate_child() -> None:
    fused = RRFFusion().fuse(
        [candidate("a", "dense", 1), candidate("a", "dense", 2)],
        [candidate("a", "bm25", 1)],
    )

    assert len(fused) == 1
    assert fused[0].score == (1 / 61) + (1 / 61)


def test_rrf_deterministic_order() -> None:
    dense = [candidate("z", "dense", 1)]
    bm25 = [candidate("a", "bm25", 1)]

    first = RRFFusion().fuse(dense, bm25)
    second = RRFFusion().fuse(dense, bm25)

    assert [item.child_id for item in first] == ["a", "z"]
    assert first == second
