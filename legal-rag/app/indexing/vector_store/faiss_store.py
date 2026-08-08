"""Local cosine-similarity FAISS vector store with stable external IDs."""

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from app.indexing.vector_store.base import BaseVectorStore


class FAISSVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self._index: Any | None = None
        self._ids: list[str] = []

    def create(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("FAISS dimension must be positive.")
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss-cpu is required.") from exc
        self._index = faiss.IndexFlatIP(dimension)
        self._ids = []

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        if self._index is None:
            raise RuntimeError("FAISS index has not been created.")
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(ids):
            raise ValueError("FAISS vectors and IDs are not aligned.")
        if len(set(ids)) != len(ids) or set(ids) & set(self._ids):
            raise ValueError("FAISS IDs must be globally unique.")
        if matrix.shape[1] != self._index.d:
            raise ValueError("FAISS vector dimension mismatch.")
        self._index.add(matrix)
        self._ids.extend(ids)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        if self._index is None:
            raise RuntimeError("FAISS index has not been loaded.")
        if top_k <= 0 or not self._ids:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if allowed_ids is not None:
            return self._search_allowed(query[0], top_k, allowed_ids)
        search_k = min(len(self._ids), max(top_k, top_k * 10))
        while True:
            scores, positions = self._index.search(query, search_k)
            results = [
                (self._ids[int(position)], float(score))
                for score, position in zip(scores[0], positions[0], strict=True)
                if position >= 0
            ]
            if len(results) >= top_k or search_k == len(self._ids):
                return results[:top_k]
            search_k = min(len(self._ids), search_k * 2)

    def _search_allowed(
        self, query: np.ndarray, top_k: int, allowed_ids: set[str]
    ) -> list[tuple[str, float]]:
        positions = np.fromiter(
            (
                position
                for position, identifier in enumerate(self._ids)
                if identifier in allowed_ids
            ),
            dtype=np.int64,
        )
        if not len(positions):
            return []
        vectors = self._index.reconstruct_batch(positions)
        scores = np.asarray(vectors, dtype=np.float32) @ query
        limit = min(top_k, len(positions))
        if limit < len(positions):
            selected = np.argpartition(scores, -limit)[-limit:]
        else:
            selected = np.arange(len(positions))
        return sorted(
            (
                (self._ids[int(positions[index])], float(scores[index]))
                for index in selected
            ),
            key=lambda item: (-item[1], item[0]),
        )[:limit]

    def search_many(
        self,
        query_vectors: np.ndarray,
        top_k: int,
        allowed_ids: list[set[str] | None],
    ) -> list[list[tuple[str, float]]]:
        """Search unrestricted queries in OpenMP-friendly small batches."""
        if self._index is None:
            raise RuntimeError("FAISS index has not been loaded.")
        matrix = np.ascontiguousarray(query_vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(allowed_ids):
            raise ValueError("Query vectors and allowed-ID sets must be aligned.")
        grouped: list[list[tuple[str, float]]] = [[] for _ in allowed_ids]
        unrestricted = [
            position for position, allowed in enumerate(allowed_ids) if allowed is None
        ]
        if unrestricted and top_k > 0 and self._ids:
            # FAISS switches to BLAS at 20 queries. Some Windows CPU wheels use a
            # single-threaded BLAS, while the direct path below uses OpenMP.
            for start in range(0, len(unrestricted), 16):
                query_positions = unrestricted[start : start + 16]
                scores, positions = self._index.search(
                    matrix[query_positions], top_k
                )
                for result_position, query_position in enumerate(query_positions):
                    grouped[query_position] = [
                        (self._ids[int(position)], float(score))
                        for score, position in zip(
                            scores[result_position],
                            positions[result_position],
                            strict=True,
                        )
                        if position >= 0
                    ]
        for query_position, allowed in enumerate(allowed_ids):
            if allowed is not None:
                grouped[query_position] = self.search(
                    matrix[query_position], top_k, allowed
                )
        return grouped

    def save(self, path: Path) -> None:
        if self._index is None:
            raise RuntimeError("FAISS index has not been created.")
        import faiss

        path.parent.mkdir(parents=True, exist_ok=True)
        ids_path = path.with_suffix(path.suffix + ".ids.json")
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".faiss", delete=False
        ) as stream:
            temporary_index = Path(stream.name)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False
        ) as stream:
            temporary_ids = Path(stream.name)
            json.dump(self._ids, stream, ensure_ascii=False)
        try:
            faiss.write_index(self._index, str(temporary_index))
            temporary_index.replace(path)
            temporary_ids.replace(ids_path)
        finally:
            temporary_index.unlink(missing_ok=True)
            temporary_ids.unlink(missing_ok=True)

    def load(self, path: Path) -> None:
        import faiss

        ids_path = path.with_suffix(path.suffix + ".ids.json")
        if not path.is_file() or not ids_path.is_file():
            raise FileNotFoundError(f"Incomplete FAISS artifacts: {path}")
        index = faiss.read_index(str(path))
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise ValueError("Invalid FAISS ID mapping.")
        if index.ntotal != len(ids) or len(set(ids)) != len(ids):
            raise ValueError("FAISS index and ID mapping do not match.")
        self._index = index
        self._ids = ids
