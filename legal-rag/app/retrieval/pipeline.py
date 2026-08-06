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
    ) -> None:
        self.query_analyzer = query_analyzer
        self.metadata_filter = metadata_filter
        self.dense_retriever = dense_retriever
        self.lexical_retriever = lexical_retriever
        self.fusion = fusion
        self.reranker = reranker
        self.parent_expander = parent_expander
        self.context_builder = context_builder

    def retrieve(self, query: LegalQuery) -> LegalContext:
        """Run analysis, filtering, hybrid retrieval, reranking, and expansion."""
        # TODO(phase-implementation):
        # Implement orchestration with empty-filter fallback to full corpus.
        raise NotImplementedError
