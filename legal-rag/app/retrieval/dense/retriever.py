"""Dense retriever."""

from app.domain.retrieval import RetrievalCandidate
from app.indexing.embeddings.base import BaseEmbeddingModel
from app.indexing.metadata_store.repository import LegalRepository
from app.indexing.vector_store.base import BaseVectorStore


class DenseRetriever:
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        repository: LegalRepository,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.repository = repository

    def retrieve(
        self,
        query: str,
        top_n: int,
        allowed_ids: set[str] | None = None,
    ) -> list[RetrievalCandidate]:
        """Embed query, search FAISS, and load chunk metadata."""
        return self.retrieve_many([query], top_n, [allowed_ids])[0]

    def retrieve_many(
        self,
        queries: list[str],
        top_n: int,
        allowed_ids: list[set[str] | None],
    ) -> list[list[RetrievalCandidate]]:
        """Batch query embedding, then search and hydrate each ranked result."""
        if len(queries) != len(allowed_ids):
            raise ValueError("Queries and allowed-ID sets must be aligned.")
        if not queries:
            return []
        vectors = self.embedding_model.embed_queries(queries)
        grouped: list[list[RetrievalCandidate]] = []
        matches_by_query = self.vector_store.search_many(vectors, top_n, allowed_ids)
        all_child_ids = list(
            dict.fromkeys(
                child_id
                for matches in matches_by_query
                for child_id, _ in matches
            )
        )
        children = self.repository.get_children(all_child_ids)
        for matches in matches_by_query:
            results: list[RetrievalCandidate] = []
            for child_id, score in matches:
                child = children.get(child_id)
                if child is not None:
                    results.append(
                    RetrievalCandidate(
                        child_id=child_id,
                        text=child.original_text,
                        metadata=child.metadata,
                        dense_score=score,
                    )
                )
            grouped.append(results)
        return grouped
