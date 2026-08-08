"""Build and persist the legal corpus chunk index from organizer JSON files."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.core.constants import CONTEXT_FILE_GLOB
from app.corpus.fts import LegalCorpusIndex


@dataclass(frozen=True, slots=True)
class CorpusIngestionReport:
    source_directory: Path
    database_path: Path
    context_files: int
    documents_indexed: int
    chunks_indexed: int
    target_words: int
    overlap_words: int


class CorpusIngestionService:
    """Chunk every ``context_*.json`` file and persist a searchable FTS index."""

    def __init__(
        self,
        database_path: Path,
        manifest_path: Path,
        target_words: int = 350,
        overlap_words: int = 60,
    ) -> None:
        self.database_path = database_path
        self.manifest_path = manifest_path
        self.target_words = target_words
        self.overlap_words = overlap_words

    def ingest(self, source_directory: Path) -> CorpusIngestionReport:
        if not source_directory.is_dir():
            raise ValueError(f"Context source must be a directory: {source_directory}")
        context_files = sorted(source_directory.rglob(CONTEXT_FILE_GLOB))
        if not context_files:
            raise ValueError(
                f"No {CONTEXT_FILE_GLOB} files found under {source_directory}"
            )
        documents, chunks = LegalCorpusIndex(self.database_path).build_from_directory(
            source_directory,
            target_words=self.target_words,
            overlap_words=self.overlap_words,
        )
        report = CorpusIngestionReport(
            source_directory=source_directory,
            database_path=self.database_path,
            context_files=len(context_files),
            documents_indexed=documents,
            chunks_indexed=chunks,
            target_words=self.target_words,
            overlap_words=self.overlap_words,
        )
        self._write_manifest(report)
        return report

    def _write_manifest(self, report: CorpusIngestionReport) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(report)
        payload["source_directory"] = str(report.source_directory)
        payload["database_path"] = str(report.database_path)
        self.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
