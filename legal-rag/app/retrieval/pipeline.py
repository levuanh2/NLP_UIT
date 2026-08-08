"""Hybrid retrieval pipeline composition root."""

from app.domain.queries import LegalQuery
from app.domain.retrieval import LegalContext
from app.retrieval.context.context_builder import LegalContextBuilder
from app.retrieval.context.parent_expander import ParentContextExpander
from app.retrieval.dense.retriever import DenseRetriever
from app.retrieval.filters.metadata_filter import MetadataFilter
from app.retrieval.fusion.rrf import ReciprocalRankFusion
from app.retrieval.lexical.retriever import LexicalRetriever
from app.retrieval.query.analyzer import QueryAnalyzer
from app.retrieval.reranking.base import BaseReranker


class RetrievalPipeline:
    def __init__(
        self,
        query_analyzer: QueryAnalyzer,
        metadata_filter: MetadataFilter,
        dense_retriever: DenseRetriever,
        lexical_retriever: LexicalRetriever,
        fusion: ReciprocalRankFusion,
        reranker: BaseReranker,
        parent_expander: ParentContextExpander,
        context_builder: LegalContextBuilder,
        dense_top_n: int = 20,
        lexical_top_n: int = 20,
        fusion_top_n: int = 30,
        rerank_top_k: int = 5,
    ) -> None:
        self.query_analyzer = query_analyzer
        self.metadata_filter = metadata_filter
        self.dense_retriever = dense_retriever
        self.lexical_retriever = lexical_retriever
        self.fusion = fusion
        self.reranker = reranker
        self.parent_expander = parent_expander
        self.context_builder = context_builder
        self.dense_top_n = dense_top_n
        self.lexical_top_n = lexical_top_n
        self.fusion_top_n = fusion_top_n
        self.rerank_top_k = rerank_top_k

    def retrieve(self, query: LegalQuery) -> LegalContext:
        """Run analysis, filtering, hybrid retrieval, reranking, and expansion."""
        return self.retrieve_many([query])[0]

    def retrieve_many(self, queries: list[LegalQuery]) -> list[LegalContext]:
        """Batch model stages so each local model is loaded only once."""
        if not queries:
            return []
        analyses = [self.query_analyzer.analyze(item.question) for item in queries]
        allowed_sets = [
            self.metadata_filter.allowed_ids(analysis.metadata) for analysis in analyses
        ]
        dense_lists = self.dense_retriever.retrieve_many(
            [analysis.normalized_query for analysis in analyses],
            self.dense_top_n,
            allowed_sets,
        )
        self.dense_retriever.embedding_model.unload()
        lexical_lists = [
            self.lexical_retriever.retrieve(
                analysis.normalized_query, self.lexical_top_n, allowed
            )
            for analysis, allowed in zip(analyses, allowed_sets, strict=True)
        ]
        fused_lists = [
            self.fusion.fuse([dense, lexical], self.fusion_top_n)
            for dense, lexical in zip(dense_lists, lexical_lists, strict=True)
        ]
        reranked_lists = [
            self.reranker.rerank(analysis.normalized_query, fused, self.rerank_top_k)
            for analysis, fused in zip(analyses, fused_lists, strict=True)
        ]
        self.reranker.unload()
        return [
            self.context_builder.build(
                analysis.normalized_query,
                self.parent_expander.expand(reranked),
            )
            for analysis, reranked in zip(analyses, reranked_lists, strict=True)
        ]
