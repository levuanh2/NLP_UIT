"""Ingestion CLI command."""

from pathlib import Path

import typer

from app.core.config import get_settings


def ingest(
    source: Path | None = typer.Option(
        None,
        "--source",
        "-s",
        help="Context directory; defaults to CORPUS_DATA_DIR.",
    ),
) -> None:
    """Scan the corpus directory and ingest every context_*.json file."""
    source_directory = source or get_settings().corpus_data_dir
    if not source_directory.is_dir():
        raise typer.BadParameter(
            f"Context directory does not exist: {source_directory}"
        )
    # TODO(phase-implementation):
    # Wire IngestionService to scan the directory, process every context file,
    # and pass resulting chunks to IndexingService.
    typer.echo(f"JSON context ingestion scaffold ready for {source_directory}.")
