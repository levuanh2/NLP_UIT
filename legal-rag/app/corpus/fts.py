"""SQLite FTS5 index for the organizer-provided legal corpus ZIP."""

import hashlib
import json
import re
import sqlite3
import tempfile
import unicodedata
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;:])\s+|\n+")
_STOPWORDS = frozenset(
    {
        "ai",
        "bao",
        "bị",
        "các",
        "cần",
        "cho",
        "có",
        "của",
        "được",
        "gì",
        "hay",
        "khi",
        "không",
        "là",
        "làm",
        "một",
        "nào",
        "như",
        "những",
        "phải",
        "quy",
        "sao",
        "sẽ",
        "theo",
        "thế",
        "thì",
        "trong",
        "trường",
        "tại",
        "và",
        "về",
        "với",
    }
)


@dataclass(frozen=True, slots=True)
class CorpusEvidence:
    """One retrieved legal-corpus chunk."""

    document_id: int
    document_name: str
    source_link: str
    chunk_index: int
    text: str
    score: float
    chunk_id: str = ""
    article: str | None = None
    semantic_score: float | None = None
    hybrid_score: float | None = None


class LegalCorpusIndex:
    """Build and query a compact FTS5 index without loading the corpus in RAM."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def build_from_zip(
        self,
        zip_path: Path,
        target_words: int = 350,
        overlap_words: int = 60,
    ) -> tuple[int, int]:
        """Create the index from every context JSON in the archive."""
        if not zip_path.is_file():
            raise FileNotFoundError(f"Corpus ZIP does not exist: {zip_path}")
        if target_words <= 0 or not 0 <= overlap_words < target_words:
            raise ValueError("Invalid corpus chunk size or overlap.")
        with zipfile.ZipFile(zip_path) as archive:
            entries = [
                entry
                for entry in sorted(archive.infolist(), key=lambda item: item.filename)
                if entry.filename.lower().endswith(".json")
            ]

            def records() -> Iterator[Any]:
                for entry in entries:
                    with archive.open(entry) as stream:
                        yield json.load(stream)

            return self._build(records(), target_words, overlap_words, str(zip_path))

    def build_from_directory(
        self,
        source_directory: Path,
        target_words: int = 350,
        overlap_words: int = 60,
    ) -> tuple[int, int]:
        """Build from all context JSON files without reading question data."""
        if not source_directory.is_dir():
            raise FileNotFoundError(
                f"Corpus directory does not exist: {source_directory}"
            )
        paths = sorted(source_directory.glob("context_*.json"))
        if not paths:
            raise ValueError(f"No context_*.json files found in {source_directory}")

        def records() -> Iterator[Any]:
            for path in paths:
                with path.open(encoding="utf-8") as stream:
                    yield json.load(stream)

        return self._build(
            records(), target_words, overlap_words, str(source_directory)
        )

    def _build(
        self,
        records: Iterable[Any],
        target_words: int,
        overlap_words: int,
        source: str,
    ) -> tuple[int, int]:
        if target_words <= 0 or not 0 <= overlap_words < target_words:
            raise ValueError("Invalid corpus chunk size or overlap.")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.NamedTemporaryFile(
            prefix=f".{self.database_path.name}.",
            suffix=".building",
            dir=self.database_path.parent,
            delete=False,
        )
        temporary_path = Path(temporary.name)
        temporary.close()
        connection = sqlite3.connect(temporary_path)
        try:
            self._initialize(connection)
            document_count = 0
            chunk_count = 0
            with connection:
                for record in records:
                    validated = _validate_context_record(record)
                    if validated is None:
                        continue
                    document_id, document_name, source_link, passage = validated
                    document_count += 1
                    for chunk_index, (article, text) in enumerate(
                        structure_aware_chunks(passage, target_words, overlap_words)
                    ):
                        chunk_id = stable_chunk_id(document_id, chunk_index, text)
                        connection.execute(
                            "INSERT INTO chunks(document_id, document_name, "
                            "source_link, chunk_id, chunk_index, article, text) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (
                                document_id,
                                document_name,
                                source_link,
                                chunk_id,
                                chunk_index,
                                article,
                                text,
                            ),
                        )
                        chunk_count += 1
                connection.execute("INSERT INTO chunks(chunks) VALUES('optimize')")
                connection.execute(
                    "INSERT INTO manifest(schema_version, source, target_words, "
                    "overlap_words, document_count, chunk_count) "
                    "VALUES(?, ?, ?, ?, ?, ?)",
                    (
                        2,
                        source,
                        target_words,
                        overlap_words,
                        document_count,
                        chunk_count,
                    ),
                )
        except Exception:
            connection.close()
            temporary_path.unlink(missing_ok=True)
            raise
        connection.close()
        temporary_path.replace(self.database_path)
        return document_count, chunk_count

    def search(self, query: str, limit: int = 5) -> list[CorpusEvidence]:
        """Retrieve the best legal chunks using Unicode-aware BM25."""
        if limit <= 0:
            return []
        terms = query_terms(query)
        if not terms:
            return []
        connection = sqlite3.connect(self.database_path)
        try:
            ranked_terms = self._rank_terms_by_document_frequency(connection, terms)
            rows: list[tuple[Any, ...]] = []
            for term_count in (4, 3, 2, 1):
                selected = ranked_terms[:term_count]
                if len(selected) < term_count:
                    continue
                match_query = " AND ".join(f'"{term}"' for term in selected)
                rows = self._search_rows(connection, match_query, limit)
                if rows:
                    break
            if not rows:
                match_query = " OR ".join(f'"{term}"' for term in ranked_terms[:4])
                rows = self._search_rows(connection, match_query, limit)
        finally:
            connection.close()
        return [
            CorpusEvidence(
                document_id=int(row[0]),
                document_name=str(row[1]),
                source_link=str(row[2]),
                chunk_id=str(row[3]),
                chunk_index=int(row[4]),
                article=str(row[5]) if row[5] else None,
                text=str(row[6]),
                score=float(-row[7]),
            )
            for row in rows
        ]

    @staticmethod
    def _search_rows(
        connection: sqlite3.Connection,
        match_query: str,
        limit: int,
    ) -> list[tuple[Any, ...]]:
        return connection.execute(
            "SELECT document_id, document_name, source_link, chunk_id, "
            "chunk_index, article, text, "
            "bm25(chunks, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0) AS rank "
            "FROM chunks WHERE chunks MATCH ? ORDER BY rank LIMIT ?",
            (match_query, limit),
        ).fetchall()

    @staticmethod
    def _rank_terms_by_document_frequency(
        connection: sqlite3.Connection,
        terms: list[str],
    ) -> list[str]:
        folded_to_original = {fold_accents(term): term for term in terms}
        folded_terms = list(folded_to_original)
        placeholders = ",".join("?" for _ in folded_terms)
        rows = connection.execute(
            f"SELECT term, doc FROM chunks_vocab WHERE term IN ({placeholders})",
            folded_terms,
        ).fetchall()
        frequencies = {str(term): int(count) for term, count in rows}
        return sorted(
            terms,
            key=lambda term: (
                frequencies.get(fold_accents(term), 10**12),
                -len(term),
            ),
        )

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        connection.execute("DROP TABLE IF EXISTS chunks_vocab")
        connection.execute("DROP TABLE IF EXISTS chunks")
        connection.execute("DROP TABLE IF EXISTS manifest")
        connection.execute(
            "CREATE VIRTUAL TABLE chunks USING fts5("
            "document_id UNINDEXED, document_name, source_link UNINDEXED, "
            "chunk_id UNINDEXED, chunk_index UNINDEXED, article, text, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        connection.execute(
            "CREATE VIRTUAL TABLE chunks_vocab USING fts5vocab(chunks, 'row')"
        )
        connection.execute(
            "CREATE TABLE manifest(schema_version INTEGER NOT NULL, "
            "source TEXT NOT NULL, "
            "target_words INTEGER NOT NULL, overlap_words INTEGER NOT NULL, "
            "document_count INTEGER NOT NULL, chunk_count INTEGER NOT NULL, "
            "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")


def query_terms(text: str) -> list[str]:
    """Extract stable Vietnamese search terms while retaining legal numbers."""
    terms: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_PATTERN.findall(text.casefold()):
        if token in _STOPWORDS or (len(token) < 2 and not token.isdigit()):
            continue
        if token not in seen:
            terms.append(token)
            seen.add(token)
    terms.sort(
        key=lambda value: (
            not any(char.isdigit() for char in value),
            -len(value),
        )
    )
    return terms


def fold_accents(text: str) -> str:
    """Match the FTS unicode61 remove_diacritics=2 vocabulary form."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    return without_marks.replace("đ", "d")


def chunk_text(
    text: str,
    target_words: int = 350,
    overlap_words: int = 60,
) -> Iterator[str]:
    """Yield overlapping word windows after conservative whitespace cleanup."""
    words = text.replace("\u00a0", " ").split()
    if not words:
        return
    step = max(1, target_words - overlap_words)
    for start in range(0, len(words), step):
        chunk = words[start : start + target_words]
        if not chunk:
            break
        yield " ".join(chunk)
        if start + target_words >= len(words):
            break


_ARTICLE_BOUNDARY = re.compile(r"(?im)(?=^\s*(?:điều|article)\s+\d+[a-zđ]?(?:[.\s:]))")
_ARTICLE_LABEL = re.compile(r"(?i)^\s*((?:điều|article)\s+\d+[a-zđ]?)")


def structure_aware_chunks(
    text: str,
    target_words: int = 350,
    overlap_words: int = 60,
) -> Iterator[tuple[str | None, str]]:
    """Keep legal articles together when possible and window oversized articles."""
    cleaned = clean_legal_text(text)
    sections = [
        part.strip() for part in _ARTICLE_BOUNDARY.split(cleaned) if part.strip()
    ]
    for section in sections:
        match = _ARTICLE_LABEL.match(section)
        article = match.group(1) if match else None
        for chunk in chunk_text(section, target_words, overlap_words):
            yield article, chunk


def clean_legal_text(text: str) -> str:
    """Conservatively normalize whitespace while preserving legal line markers."""
    normalized = unicodedata.normalize("NFC", text).replace("\u00a0", " ")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def stable_chunk_id(document_id: int, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"doc-{document_id}:chunk-{chunk_index}:{digest}"


def _validate_context_record(
    record: Any,
) -> tuple[int, str, str, str] | None:
    if not isinstance(record, dict):
        return None
    document_id = record.get("id")
    passage = record.get("passage")
    if not isinstance(document_id, int) or not isinstance(passage, str):
        return None
    if not passage.strip():
        return None
    name = record.get("name")
    link = record.get("link")
    return (
        document_id,
        name if isinstance(name, str) else "",
        link if isinstance(link, str) else "",
        passage,
    )


def extract_answer_span(
    question: str,
    evidences: list[CorpusEvidence],
    max_words: int = 360,
) -> str:
    """Extract the evidence span with the strongest query-term coverage."""
    if not evidences:
        return ""
    terms = set(query_terms(question))
    best_text = evidences[0].text
    best_score = -1.0
    for evidence in evidences:
        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_BOUNDARY.split(evidence.text)
            if sentence.strip()
        ]
        if not sentences:
            sentences = [evidence.text]
        for start in range(len(sentences)):
            words: list[str] = []
            for end in range(start, min(len(sentences), start + 8)):
                words.extend(sentences[end].split())
                if len(words) > max_words:
                    break
                candidate = " ".join(words)
                candidate_terms = set(query_terms(candidate))
                coverage = len(terms & candidate_terms) / max(1, len(terms))
                compactness = len(terms & candidate_terms) / max(30, len(words))
                score = coverage + 0.35 * compactness
                if score > best_score and len(words) >= 35:
                    best_score = score
                    best_text = candidate
    return " ".join(best_text.split()[:max_words]).strip()
