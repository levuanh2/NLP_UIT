"""Bounded vector-index writer contracts and local FAISS sharding."""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from app.indexing.vector_store.faiss_store import FAISSVectorStore


class VectorIndexWriter(ABC):
    @abstractmethod
    def add_batch(self, ids: list[str], vectors: np.ndarray) -> None:
        """Add one bounded embedding batch."""

    @abstractmethod
    def finalize(self) -> None:
        """Flush the final incomplete shard."""

    @abstractmethod
    def save(self, output_dir: Path) -> None:
        """Persist index artifacts."""


class FAISSShardWriter(VectorIndexWriter):
    def __init__(
        self,
        output_dir: Path,
        dimension: int,
        index_type: str = "auto",
        metric: str = "cosine",
        normalize_embeddings: bool = True,
        shard_size: int = 50_000,
    ) -> None:
        self.output_dir = output_dir
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.normalize_embeddings = normalize_embeddings
        self.shard_size = shard_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._known_ids = self._load_existing_ids()
        self._shard_number = len(list(self.output_dir.glob("shard_*.index")))
        self._store = self._new_store()

    @property
    def count(self) -> int:
        return len(self._known_ids)

    @property
    def resolved_index_type(self) -> str:
        return self._store.resolved_index_type

    def add_batch(self, ids: list[str], vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape != (len(ids), self.dimension):
            raise ValueError("Embedding batch shape does not match IDs/dimension")
        keep = [
            index
            for index, child_id in enumerate(ids)
            if child_id not in self._known_ids
        ]
        cursor = 0
        while cursor < len(keep):
            capacity = self.shard_size - self._store.count
            indexes = keep[cursor : cursor + capacity]
            batch_ids = [ids[index] for index in indexes]
            self._store.add(matrix[indexes], batch_ids)
            self._known_ids.update(batch_ids)
            cursor += len(indexes)
            if self._store.count == self.shard_size:
                self._flush_shard()

    def finalize(self) -> None:
        if self._store.count:
            self._flush_shard()
        state = self.output_dir / "shards.json"
        state.write_text(
            json.dumps(
                {
                    "count": self.count,
                    "dimension": self.dimension,
                    "dtype": "float32",
                    "estimated_vector_bytes": self.count * self.dimension * 4,
                    "index_type": self.index_type,
                    "resolved_index_type": self.resolved_index_type,
                    "shards": self._shard_number,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def save(self, output_dir: Path) -> None:
        if output_dir.resolve() != self.output_dir.resolve():
            raise ValueError(
                "FAISSShardWriter persists directly to its version directory"
            )
        self.finalize()

    def _new_store(self) -> FAISSVectorStore:
        store = FAISSVectorStore(
            index_type=self.index_type,
            metric=self.metric,
            normalize_embeddings=self.normalize_embeddings,
        )
        store.create(self.dimension)
        return store

    def _flush_shard(self) -> None:
        path = self.output_dir / f"shard_{self._shard_number:04d}.index"
        self._store.save(path)
        self._shard_number += 1
        self._store = self._new_store()

    def _load_existing_ids(self) -> set[str]:
        ids: set[str] = set()
        for path in sorted(self.output_dir.glob("shard_*.index.ids.json")):
            ids.update(json.loads(path.read_text(encoding="utf-8")))
        return ids
