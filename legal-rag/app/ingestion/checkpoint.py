"""Atomic filesystem checkpoints for resumable ingestion jobs."""

import os
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class IngestionCheckpoint(BaseModel):
    job_id: str
    index_version: str
    last_processed_document_id: int | None = None
    last_processed_path: str | None = None
    documents_processed: int = 0
    documents_failed: int = 0
    chunks_created: int = 0
    children_created: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: str = "running"
    completed_documents: dict[str, str] = Field(default_factory=dict)
    failed_documents: dict[str, str] = Field(default_factory=dict)


class IngestionCheckpointManager:
    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        self._checkpoint: IngestionCheckpoint | None = None

    def load(self, job_id: str) -> IngestionCheckpoint | None:
        """Load the latest checkpoint for one job."""
        path = self._path(job_id)
        if not path.is_file():
            return None
        self._checkpoint = IngestionCheckpoint.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        return self._checkpoint

    def save(self, checkpoint: IngestionCheckpoint) -> None:
        """Persist a checkpoint atomically in the same filesystem."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint.timestamp = datetime.now(UTC)
        path = self._path(checkpoint.job_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, path)
        self._checkpoint = checkpoint

    def mark_document_completed(self, document_id: int, checksum: str) -> None:
        """Record a completed document in the active job checkpoint."""
        checkpoint = self._require_checkpoint()
        checkpoint.completed_documents[str(document_id)] = checksum
        self.save(checkpoint)

    def is_document_completed(self, document_id: int, checksum: str) -> bool:
        """Return whether this exact document content was completed."""
        checkpoint = self._checkpoint
        return bool(
            checkpoint
            and checkpoint.completed_documents.get(str(document_id)) == checksum
        )

    def mark_document_failed(self, document_id: int, checksum: str) -> None:
        checkpoint = self._require_checkpoint()
        checkpoint.failed_documents[str(document_id)] = checksum
        self.save(checkpoint)

    def is_document_failed(self, document_id: int, checksum: str) -> bool:
        checkpoint = self._checkpoint
        return bool(
            checkpoint and checkpoint.failed_documents.get(str(document_id)) == checksum
        )

    def _path(self, job_id: str) -> Path:
        safe_job_id = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in job_id
        )
        return self.checkpoint_dir / f"{safe_job_id}.json"

    def _require_checkpoint(self) -> IngestionCheckpoint:
        if self._checkpoint is None:
            raise RuntimeError("No active ingestion checkpoint")
        return self._checkpoint
