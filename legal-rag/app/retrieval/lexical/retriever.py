"""BM25 lexical retriever."""

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
        results: list[RetrievalCandidate] = []
        for child_id, score in self.index.search(query, top_n, allowed_ids):
            child = self.repository.get_child(child_id)
            if child is not None:
                results.append(
                    RetrievalCandidate(
                        child_id=child_id,
                        text=child.original_text,
                        metadata=child.metadata,
                        bm25_score=score,
                    )
                )
        return results
