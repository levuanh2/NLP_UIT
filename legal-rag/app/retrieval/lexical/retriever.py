"""BM25 retriever skeleton."""

from app.domain.retrieval import RetrievalCandidate
from app.indexing.lexical.bm25_index import BM25Index
from app.indexing.metadata_store.repository import LegalRepository


class LexicalRetriever:
    def __init__(self, index: BM25Index, repository: LegalRepository) -> None:
        self.index = index
        self.repository = repository

    def retrieve(
        self,
        query: str,
        top_n: int,
        allowed_ids: set[str] | None = None,
    ) -> list[RetrievalCandidate]:
        """Run BM25 search."""
        # TODO(phase-implementation):
        # Implement BM25 retrieval and domain candidate mapping.
        raise NotImplementedError
