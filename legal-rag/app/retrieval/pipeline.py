"""Hybrid retrieval orchestration ending at ranked, budgeted evidence."""

import logging

from app.domain.queries import LegalQuery
from app.domain.retrieval import RetrievalCandidate, RetrievalResult
from app.retrieval.context.context_builder import LegalContextBuilder
from app.retrieval.context.parent_expander import ParentContextExpander
from app.retrieval.dense.retriever import DenseRetriever
from app.retrieval.filters.metadata_filter import MetadataFilter
from app.retrieval.fusion.rrf import RRFFusion
from app.retrieval.lexical.retriever import BM25Retriever
from app.retrieval.query.analyzer import QueryAnalyzer
from app.retrieval.reranking.base import BaseReranker

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(
        self,
        query_analyzer: QueryAnalyzer,
        metadata_filter: MetadataFilter,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        fusion: RRFFusion,
        reranker: BaseReranker | None,
        parent_expander: ParentContextExpander,
        context_builder: LegalContextBuilder,
        *,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        rrf_k: int = 60,
        rrf_top_k: int = 30,
        reranker_enabled: bool = True,
        reranker_top_k: int = 10,
        trace: bool = False,
        active_index_version: str | None = None,
    ) -> None:
        self.query_analyzer = query_analyzer
        self.metadata_filter = metadata_filter
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.fusion = fusion
        self.reranker = reranker
        self.parent_expander = parent_expander
        self.context_builder = context_builder
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.rrf_top_k = rrf_top_k
        self.reranker_enabled = reranker_enabled
        self.reranker_top_k = reranker_top_k
        self.trace = trace
        dense_version = getattr(dense_retriever, "active_index_version", None)
        bm25_version = getattr(bm25_retriever, "active_index_version", None)
        if dense_version and bm25_version and dense_version != bm25_version:
            raise RuntimeError(
                f"Dense/BM25 active versions differ: {dense_version} != {bm25_version}"
            )
        self.active_index_version = (
            active_index_version
            or dense_version
            or bm25_version
            or "injected-test-index"
        )

    def retrieve(self, query: str | LegalQuery) -> RetrievalResult:
        """Run filtering, hybrid ranking, expansion, and context budgeting."""
        raw_query = query.question if isinstance(query, LegalQuery) else query
        metadata = self.query_analyzer.analyze(raw_query)
        retrieval_filter = self.metadata_filter.build_filter(metadata)
        candidate_ids = retrieval_filter.candidate_ids
        fallback = False
        if (
            retrieval_filter.applied
            and not candidate_ids
            and not retrieval_filter.authoritative
        ):
            candidate_ids = None
            fallback = True

        self._trace(
            "query_analysis",
            query=raw_query,
            metadata=metadata.model_dump(exclude_none=True),
            filter_fields=retrieval_filter.fields,
            filter_matches=retrieval_filter.matched_count,
            filter_fallback=fallback,
        )
        dense = self.dense_retriever.retrieve(
            raw_query, candidate_ids=candidate_ids, top_k=self.dense_top_k
        )
        bm25 = self.bm25_retriever.retrieve(
            raw_query, candidate_ids=candidate_ids, top_k=self.bm25_top_k
        )
        fused = self.fusion.fuse(
            dense,
            bm25,
            k=self.rrf_k,
            top_k=self.rrf_top_k,
        )
        ranked = self._rerank(raw_query, fused)
        expanded = self.parent_expander.expand(ranked)
        context = self.context_builder.build(raw_query, expanded)
        self._trace_ids("dense", dense)
        self._trace_ids("bm25", bm25)
        self._trace_ids("rrf", fused)
        self._trace_ids("reranker", ranked)
        self._trace(
            "context",
            expanded_count=len(expanded),
            final_evidence_count=len(context.evidences),
            final_token_count=context.token_count,
        )
        return RetrievalResult(
            query=raw_query,
            query_metadata=metadata,
            candidates=ranked,
            evidences=context.evidences,
            active_index_version=self.active_index_version,
            dense_count=len(dense),
            bm25_count=len(bm25),
            fused_count=len(fused),
            reranked_count=len(ranked),
            metadata_filter_applied=retrieval_filter.applied and not fallback,
            metadata_filter_fallback=fallback,
        )

    def _rerank(
        self, query: str, fused: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        if not self.reranker_enabled:
            return fused[: self.reranker_top_k]
        if self.reranker is None:
            raise RuntimeError(
                "Reranking is enabled but no local reranker is configured"
            )
        return self.reranker.rerank(query, fused, top_k=self.reranker_top_k)

    def _trace_ids(self, stage: str, candidates: list[RetrievalCandidate]) -> None:
        self._trace(
            stage,
            count=len(candidates),
            child_ids=[candidate.child_id for candidate in candidates],
        )

    def _trace(self, stage: str, **details: object) -> None:
        if self.trace:
            logger.info("retrieval_trace stage=%s details=%s", stage, details)
