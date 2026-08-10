"""Disk-backed BM25/FTS5 retrieval."""

from pathlib import Path

from app.domain.retrieval import RetrievalCandidate
from app.indexing.lexical.bm25_index import BM25Index
from app.indexing.metadata_store.repository import LegalRepository
from app.retrieval.active_index import ActiveIndex


class BM25Retriever:
    def __init__(
        self,
        index: BM25Index | None = None,
        repository: LegalRepository | None = None,
        *,
        index_root: Path | None = None,
    ) -> None:
        self.repository = repository
        self.active_index = ActiveIndex(index_root) if index_root is not None else None
        self.index = index or BM25Index()
        if self.active_index is not None:
            self.index.load(self.active_index.bm25_path)
        elif index is None:
            raise ValueError(
                "BM25Retriever requires an active index or loaded BM25 index"
            )

    @property
    def active_index_version(self) -> str | None:
        return self.active_index.version if self.active_index else None

    def retrieve(
        self,
        query: str,
        *,
        candidate_ids: set[str] | None = None,
        top_k: int = 20,
    ) -> list[RetrievalCandidate]:
        hits = self.index.search(query, top_k, candidate_ids)
        return [
            RetrievalCandidate(
                child_id=child_id,
                score=score,
                source="bm25",
                rank=rank,
                bm25_score=score,
            )
            for rank, (child_id, score) in enumerate(hits, start=1)
        ]


# Compatibility name retained for the previous scaffold.
LexicalRetriever = BM25Retriever
