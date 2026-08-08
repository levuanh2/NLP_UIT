"""Ingestion pipeline composition root."""

from collections.abc import Iterator
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
        return list(self.iter_run(source_directory))

    def iter_run(self, source_directory: Path) -> Iterator[IngestionResult]:
        """Stream ingestion results to avoid retaining the complete corpus."""
        files = self.discover_context_files(source_directory)
        if not files:
            raise ValueError(
                f"No {CONTEXT_FILE_GLOB} files found in {source_directory}"
            )
        for path in files:
            parser = self.parser_factory.get_parser(path)
            document = parser.parse(path)
            if not document.raw_text.strip():
                continue
            document = document.model_copy(
                update={"cleaned_text": self.cleaner.clean(document.raw_text)}
            )
            document = self.structure_extractor.extract(document)
            parents, children = self.chunker.chunk(document)
            children = self.metadata_enricher.enrich(document, children)
            yield IngestionResult(
                document=document,
                parent_chunks=parents,
                child_chunks=children,
            )
