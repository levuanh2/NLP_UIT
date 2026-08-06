"""Ingestion use-case service skeleton."""

from pathlib import Path

from app.ingestion.pipeline import IngestionPipeline, IngestionResult


class IngestionService:
    def __init__(self, pipeline: IngestionPipeline) -> None:
        self.pipeline = pipeline

    def ingest(self, source_directory: Path) -> list[IngestionResult]:
        # TODO(phase-implementation):
        # Validate the context directory and invoke its ingestion pipeline.
        raise NotImplementedError
