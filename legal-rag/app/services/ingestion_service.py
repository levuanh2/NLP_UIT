"""Corpus ingestion use-case service."""

from pathlib import Path

from app.ingestion.pipeline import IngestionPipeline, IngestionResult


class IngestionService:
    def __init__(self, pipeline: IngestionPipeline) -> None:
        self.pipeline = pipeline

    def ingest(self, source_directory: Path) -> list[IngestionResult]:
        if not source_directory.is_dir():
            raise ValueError(f"Context source must be a directory: {source_directory}")
        return self.pipeline.run(source_directory)
