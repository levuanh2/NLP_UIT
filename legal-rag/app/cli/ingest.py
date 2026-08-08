"""Ingestion CLI command."""

from pathlib import Path

import typer

from app.core.config import get_settings
from app.corpus.fts import LegalCorpusIndex
from app.ingestion.cache import write_chunk_cache
from app.ingestion.chunking.parent_child import ParentChildChunker
from app.ingestion.cleaners.legal_text_cleaner import LegalTextCleaner
from app.ingestion.enrichment.metadata_enricher import MetadataEnricher
from app.ingestion.parsers.factory import DocumentParserFactory
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.structure.extractor import LegalStructureExtractor


def ingest(
    source: Path | None = typer.Option(
        None,
        "--source",
        "-s",
        help="Context directory; defaults to CORPUS_DATA_DIR.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="SQLite chunk index; defaults to SQLITE_DATABASE_PATH.",
    ),
    target_words: int = typer.Option(350, min=100),
    overlap_words: int = typer.Option(60, min=0),
    chunk_cache: Path | None = typer.Option(
        None, "--chunk-cache", help="Parent-child JSONL cache directory."
    ),
    build_fts: bool = typer.Option(True, "--build-fts/--no-build-fts"),
) -> None:
    """Clean, chunk, and persist every corpus context in one reusable index."""
    settings = get_settings()
    source_directory = source or settings.corpus_data_dir
    database_path = output or settings.sqlite_database_path
    cache_directory = chunk_cache or settings.cache_data_dir
    if not source_directory.is_dir():
        raise typer.BadParameter(
            f"Context directory does not exist: {source_directory}"
        )
    if build_fts:
        try:
            documents, chunks = LegalCorpusIndex(database_path).build_from_directory(
                source_directory,
                target_words=target_words,
                overlap_words=overlap_words,
            )
        except (OSError, ValueError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(
            f"Ingested {documents} corpus documents into {chunks} stable chunks: "
            f"{database_path}"
        )
    pipeline = IngestionPipeline(
        parser_factory=DocumentParserFactory(),
        cleaner=LegalTextCleaner(),
        structure_extractor=LegalStructureExtractor(),
        chunker=ParentChildChunker(
            settings.parent_target_tokens,
            settings.parent_max_tokens,
            settings.child_target_tokens,
            settings.child_max_tokens,
        ),
        metadata_enricher=MetadataEnricher(),
    )
    domain_counts = write_chunk_cache(
        pipeline.iter_run(source_directory), cache_directory
    )
    typer.echo(
        f"Cached {domain_counts[0]} documents, {domain_counts[1]} parents and "
        f"{domain_counts[2]} children: {cache_directory}"
    )
