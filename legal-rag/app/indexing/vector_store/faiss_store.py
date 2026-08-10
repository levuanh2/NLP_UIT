"""Local FAISS vector store with an external stable child-ID mapping."""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from app.indexing.vector_store.base import BaseVectorStore


class FAISSVectorStore(BaseVectorStore):
    def __init__(
        self,
        index_type: str = "auto",
        metric: str = "cosine",
        normalize_embeddings: bool = True,
    ) -> None:
        self.index_type = index_type
        self.metric = metric
        self.normalize_embeddings = normalize_embeddings
        self.resolved_index_type = index_type
        self._index: Any | None = None
        self._ids: list[str] = []

    def create(self, dimension: int) -> None:
        import faiss

        if dimension <= 0:
            raise ValueError("FAISS dimension must be positive")
        if self.metric != "cosine":
            raise ValueError(f"Unsupported metric: {self.metric}")
        if self.index_type == "flat":
            self._index = faiss.IndexFlatIP(dimension)
            self.resolved_index_type = "flat"
        elif self.index_type in {"auto", "hnsw"}:
            self._index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
            self.resolved_index_type = "hnsw"
        elif self.index_type in {"ivf", "pq", "ivfpq"}:
            raise NotImplementedError(
                "IVF/PQ requires corpus-aware training; configure flat/hnsw for now"
            )
        else:
            raise ValueError(f"Unsupported FAISS index type: {self.index_type}")
        self._ids = []

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        index = self._require_index()
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(ids):
            raise ValueError("FAISS vectors and IDs must have aligned rows")
        if matrix.shape[1] != index.d:
            raise ValueError("FAISS vector dimension mismatch")
        if len(set(ids)) != len(ids) or set(ids).intersection(self._ids):
            raise ValueError("Duplicate child ID supplied to FAISS")
        if self.normalize_embeddings:
            import faiss

            faiss.normalize_L2(matrix)
        index.add(matrix)
        self._ids.extend(ids)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        index = self._require_index()
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if self.normalize_embeddings:
            import faiss

            faiss.normalize_L2(query)
        # FAISS selectors are index-type specific. Searching the shard and filtering
        # stable IDs guarantees recall for arbitrary SQLite-produced candidate sets.
        requested = (
            index.ntotal if allowed_ids is not None else min(index.ntotal, top_k)
        )
        scores, positions = index.search(query, requested)
        results: list[tuple[str, float]] = []
        for position, score in zip(positions[0], scores[0], strict=True):
            if position < 0:
                continue
            child_id = self._ids[int(position)]
            if allowed_ids is not None and child_id not in allowed_ids:
                continue
            results.append((child_id, float(score)))
            if len(results) == top_k:
                break
        return results

    def save(self, path: Path) -> None:
        import faiss

        index = self._require_index()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        faiss.write_index(index, str(temporary))
        os.replace(temporary, path)
        ids_path = path.with_suffix(path.suffix + ".ids.json")
        ids_temporary = ids_path.with_suffix(ids_path.suffix + ".tmp")
        ids_temporary.write_text(json.dumps(self._ids), encoding="utf-8")
        os.replace(ids_temporary, ids_path)

    def load(self, path: Path) -> None:
        import faiss

        ids_path = path.with_suffix(path.suffix + ".ids.json")
        self._index = faiss.read_index(str(path))
        self._ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if self._index.ntotal != len(self._ids):
            raise ValueError("FAISS index and ID mapping counts differ")

    @property
    def count(self) -> int:
        return len(self._ids)

    def _require_index(self) -> Any:
        if self._index is None:
            raise RuntimeError("FAISS index has not been created or loaded")
        return self._index
