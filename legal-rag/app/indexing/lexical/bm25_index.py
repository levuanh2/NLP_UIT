"""Disk-backed SQLite FTS5 lexical index and incremental writer."""

import shutil
import sqlite3
from pathlib import Path

from app.domain.chunks import ChildChunk


class BM25IndexWriter:
    """Incrementally persist lexical documents without retaining the corpus."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.output_dir / "bm25.sqlite"
        self._connection = sqlite3.connect(self.database_path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5("
            "child_id UNINDEXED, text, tokenize='unicode61 remove_diacritics 2')"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS chunk_ids("
            "child_id TEXT PRIMARY KEY, fts_rowid INTEGER UNIQUE NOT NULL)"
        )

    @property
    def count(self) -> int:
        row = self._connection.execute("SELECT count(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0

    def add_batch(self, chunks: list[ChildChunk]) -> None:
        """Add one bounded batch and commit it immediately."""
        with self._connection:
            for chunk in chunks:
                existing = self._connection.execute(
                    "SELECT fts_rowid FROM chunk_ids WHERE child_id = ?",
                    (chunk.child_id,),
                ).fetchone()
                if existing:
                    row_id = int(existing[0])
                    self._connection.execute(
                        "DELETE FROM chunks WHERE rowid = ?", (row_id,)
                    )
                    self._connection.execute(
                        "INSERT INTO chunks(rowid, child_id, text) VALUES (?, ?, ?)",
                        (row_id, chunk.child_id, chunk.text),
                    )
                else:
                    cursor = self._connection.execute(
                        "INSERT INTO chunks(child_id, text) VALUES (?, ?)",
                        (chunk.child_id, chunk.text),
                    )
                    self._connection.execute(
                        "INSERT INTO chunk_ids(child_id, fts_rowid) VALUES (?, ?)",
                        (chunk.child_id, int(cursor.lastrowid)),
                    )

    def finalize(self) -> None:
        self._connection.commit()
        self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def save(self, output_dir: Path) -> None:
        self.finalize()
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / self.database_path.name
        if target.resolve() != self.database_path.resolve():
            shutil.copy2(self.database_path, target)

    def close(self) -> None:
        self.finalize()
        self._connection.close()


class BM25Index:
    """Search a disk-backed FTS5 index using SQLite's BM25 ranking."""

    def __init__(self) -> None:
        self._connection: sqlite3.Connection | None = None
        self._path: Path | None = None

    def build(self, texts: list[str], ids: list[str]) -> None:
        raise RuntimeError(
            "Use BM25IndexWriter for bounded ingestion; in-memory build is disabled"
        )

    def search(
        self, query: str, top_n: int, allowed_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        connection = self._require_connection()
        terms = [token.replace('"', "") for token in query.split() if token.strip()]
        if not terms or top_n <= 0:
            return []
        match_query = " OR ".join(f'"{term}"' for term in terms[:64])
        if allowed_ids is not None:
            if not allowed_ids:
                return []
            connection.execute(
                "CREATE TEMP TABLE IF NOT EXISTS retrieval_allowed_ids("
                "child_id TEXT PRIMARY KEY) WITHOUT ROWID"
            )
            connection.execute("DELETE FROM retrieval_allowed_ids")
            connection.executemany(
                "INSERT INTO retrieval_allowed_ids(child_id) VALUES (?)",
                ((child_id,) for child_id in allowed_ids),
            )
            rows = connection.execute(
                "SELECT chunks.child_id, bm25(chunks) AS score FROM chunks "
                "JOIN retrieval_allowed_ids AS allowed "
                "ON allowed.child_id = chunks.child_id "
                "WHERE chunks MATCH ? ORDER BY score, chunks.child_id LIMIT ?",
                (match_query, top_n),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT child_id, bm25(chunks) AS score FROM chunks "
                "WHERE chunks MATCH ? ORDER BY score, child_id LIMIT ?",
                (match_query, top_n),
            ).fetchall()
        return [(str(child_id), float(-score)) for child_id, score in rows]

    def save(self, path: Path) -> None:
        if self._path is None:
            raise RuntimeError("BM25 index has not been loaded")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != self._path.resolve():
            shutil.copy2(self._path, path)

    def load(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        if self._connection is not None:
            self._connection.close()
        self._path = path
        self._connection = sqlite3.connect(path)

    @property
    def count(self) -> int:
        connection = self._require_connection()
        row = connection.execute("SELECT count(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("BM25 index has not been loaded")
        return self._connection
