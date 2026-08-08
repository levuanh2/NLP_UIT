"""Hybrid retrieval orchestration tests."""

from app.domain.metadata import LegalMetadata
from app.domain.queries import LegalQuery, QueryAnalysis, QueryMetadata
from app.domain.retrieval import LegalEvidence, RetrievalCandidate
from app.retrieval.context.context_builder import LegalContextBuilder
from app.retrieval.fusion.rrf import ReciprocalRankFusion
from app.retrieval.pipeline import RetrievalPipeline


def _candidate(identifier: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        child_id=identifier,
        text="Nội dung",
        metadata=LegalMetadata(
            document_id=1,
            document_name="doc",
            source_link="",
            chapter=None,
            section=None,
            article="1",
            clause=None,
            point=None,
        ),
    )


class _Analyzer:
    def analyze(self, query: str) -> QueryAnalysis:
        return QueryAnalysis(
            original_query=query,
            normalized_query=query,
            intent="test",
            metadata=QueryMetadata(article="1", confidence=1.0),
            should_apply_metadata_filter=True,
        )


class _Filter:
    def allowed_ids(self, metadata: QueryMetadata) -> set[str]:
        del metadata
        return {"allowed"}


class _Retriever:
    def __init__(self) -> None:
        self.allowed: set[str] | None = None
        self.embedding_model = self

    def unload(self) -> None:
        return None

    def retrieve(self, query: str, top_n: int, allowed: set[str] | None):
        del query, top_n
        self.allowed = allowed
        return [_candidate("allowed")]

    def retrieve_many(
        self, queries: list[str], top_n: int, allowed: list[set[str] | None]
    ):
        return [
            self.retrieve(query, top_n, allowed_ids)
            for query, allowed_ids in zip(queries, allowed, strict=True)
        ]


class _Reranker:
    def rerank(self, query: str, candidates: list[RetrievalCandidate], top_k: int):
        del query
        return candidates[:top_k]

    def unload(self) -> None:
        return None


class _Expander:
    def expand(self, candidates: list[RetrievalCandidate]):
        return [
            LegalEvidence(
                evidence_id=candidates[0].child_id,
                document_id=1,
                document_name="doc",
                source_link="",
                chapter=None,
                section=None,
                article="1",
                clause=None,
                point=None,
                text="Nội dung",
            )
        ]


def test_retrieval_pipeline_filters_before_vector_search() -> None:
    dense, lexical = _Retriever(), _Retriever()
    pipeline = RetrievalPipeline(
        _Analyzer(),
        _Filter(),
        dense,
        lexical,
        ReciprocalRankFusion(60),  # type: ignore[arg-type]
        _Reranker(),
        _Expander(),
        LegalContextBuilder(),  # type: ignore[arg-type]
    )
    context = pipeline.retrieve(LegalQuery(question_id="1", question="Điều 1?"))
    assert dense.allowed == lexical.allowed == {"allowed"}
    assert context.evidences[0].evidence_id == "allowed"
