"""Ingestion pipeline composition root."""

from pathlib import Path

from pydantic import BaseModel

from app.core.constants import CONTEXT_FILE_GLOB
from app.domain.chunks import ChildChunk, ParentChunk
from app.domain.documents import LegalDocument
from app.ingestion.chunking.base import BaseChunker
from app.ingestion.cleaners.legal_text_cleaner import LegalTextCleaner
from app.ingestion.enrichment.metadata_enricher import MetadataEnricher
from app.ingestion.parsers.factory import DocumentParserFactory
from app.ingestion.structure.extractor import LegalStructureExtractor


class IngestionResult(BaseModel):
    document: LegalDocument
    parent_chunks: list[ParentChunk]
    child_chunks: list[ChildChunk]


class IngestionPipeline:
    def __init__(
        self,
        parser_factory: DocumentParserFactory,
        cleaner: LegalTextCleaner,
        structure_extractor: LegalStructureExtractor,
        chunker: BaseChunker,
        metadata_enricher: MetadataEnricher,
    ) -> None:
        self.parser_factory = parser_factory
        self.cleaner = cleaner
        self.structure_extractor = structure_extractor
        self.chunker = chunker
        self.metadata_enricher = metadata_enricher

    def discover_context_files(self, source_directory: Path) -> list[Path]:
        """Return competition context files in deterministic filename order."""
        if not source_directory.is_dir():
            raise ValueError(f"Context source must be a directory: {source_directory}")
        return sorted(source_directory.glob(CONTEXT_FILE_GLOB))

    def run(self, source_directory: Path) -> list[IngestionResult]:
        """Ingest all ``context_*.json`` files from one folder."""
        # TODO(phase-implementation):
        # For each discovered JSON file: parse one LegalDocument, clean its
        # passage, extract structure, create parent-child chunks, and enrich
        # metadata before returning results for indexing.
        raise NotImplementedError
