"""Disk-backed Unicode-aware SQLite FTS5 BM25 index."""

import re
import sqlite3
import tempfile
from pathlib import Path

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


class BM25Index:
    def __init__(self) -> None:
        self._connection: sqlite3.Connection | None = None
        self._path: Path | None = None

    def create(self, path: Path | None = None) -> None:
        self.close()
        if path is None:
            self._connection = sqlite3.connect(":memory:")
            self._path = None
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(path)
            self._path = path
        self._connection.execute("DROP TABLE IF EXISTS chunks")
        self._connection.execute(
            "CREATE VIRTUAL TABLE chunks USING fts5("
            "child_id UNINDEXED, text, tokenize='unicode61 remove_diacritics 2')"
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.commit()

    def add(self, texts: list[str], ids: list[str]) -> None:
        if self._connection is None:
            raise RuntimeError("BM25 index has not been created.")
        if len(texts) != len(ids):
            raise ValueError("BM25 texts and IDs are not aligned.")
        with self._connection:
            self._connection.executemany(
                "INSERT INTO chunks(child_id, text) VALUES(?, ?)",
                zip(ids, texts, strict=True),
            )

    def build(self, texts: list[str], ids: list[str]) -> None:
        if len(set(ids)) != len(ids):
            raise ValueError("BM25 IDs must be unique.")
        self.create()
        self.add(texts, ids)

    def optimize(self) -> None:
        if self._connection is None:
            raise RuntimeError("BM25 index has not been created.")
        self._connection.execute("INSERT INTO chunks(chunks) VALUES('optimize')")
        self._connection.commit()

    def search(
        self, query: str, top_n: int, allowed_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        if self._connection is None:
            raise RuntimeError("BM25 index has not been built or loaded.")
        terms = list(dict.fromkeys(_TOKEN.findall(query.casefold())))
        if top_n <= 0 or not terms:
            return []
        match = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
        search_limit = max(top_n, top_n * 10)
        while True:
            rows = self._connection.execute(
                "SELECT child_id, -bm25(chunks, 0.0, 1.0) AS score "
                "FROM chunks WHERE chunks MATCH ? ORDER BY score DESC LIMIT ?",
                (match, search_limit),
            ).fetchall()
            filtered = [
                (str(identifier), float(score))
                for identifier, score in rows
                if allowed_ids is None or str(identifier) in allowed_ids
            ]
            if len(filtered) >= top_n or len(rows) < search_limit:
                return filtered[:top_n]
            search_limit *= 2

    def save(self, path: Path) -> None:
        if self._connection is None:
            raise RuntimeError("BM25 index has not been built.")
        self.optimize()
        if self._path is not None and self._path.resolve() == path.resolve():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".db", delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            target = sqlite3.connect(temporary)
            with target:
                self._connection.backup(target)
            target.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"BM25 artifact does not exist: {path}")
        self.close()
        self._connection = sqlite3.connect(path)
        self._path = path
        row = self._connection.execute(
            "SELECT name FROM sqlite_master WHERE name='chunks'"
        ).fetchone()
        if row is None:
            self.close()
            raise ValueError("Invalid BM25 SQLite artifact.")

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
        self._connection = None
        self._path = None
