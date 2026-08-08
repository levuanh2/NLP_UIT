"""Reciprocal-rank fusion tests."""

from app.domain.metadata import LegalMetadata
from app.domain.retrieval import RetrievalCandidate
from app.retrieval.fusion.rrf import ReciprocalRankFusion


def _candidate(identifier: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        child_id=identifier,
        text=identifier,
        metadata=LegalMetadata(
            document_id=1,
            document_name="doc",
            source_link="",
            chapter=None,
            section=None,
            article=None,
            clause=None,
            point=None,
        ),
    )


def test_rrf_fuses_dense_and_bm25_results() -> None:
    a, b, c = (_candidate(value) for value in ("a", "b", "c"))
    result = ReciprocalRankFusion(60).fuse([[a, b], [b, c]], top_n=3)
    assert [item.child_id for item in result] == ["b", "a", "c"]
    assert result[0].fusion_score and result[0].fusion_score > result[1].fusion_score
