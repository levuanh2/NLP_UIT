"""BM25 index skeleton."""

from pathlib import Path
from typing import Any


class BM25Index:
    def __init__(self) -> None:
        self._index: Any | None = None
        self._ids: list[str] = []

    def build(self, texts: list[str], ids: list[str]) -> None:
        # TODO(phase-implementation):
        # Tokenize Vietnamese legal text and build a rank-bm25 index.
        raise NotImplementedError

    def search(
        self, query: str, top_n: int, allowed_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        # TODO(phase-implementation):
        # Score query tokens and apply the optional allowed-ID set.
        raise NotImplementedError

    def save(self, path: Path) -> None:
        # TODO(phase-implementation):
        # Persist corpus tokens and stable IDs safely.
        raise NotImplementedError

    def load(self, path: Path) -> None:
        # TODO(phase-implementation):
        # Restore and validate the local BM25 index.
        raise NotImplementedError
