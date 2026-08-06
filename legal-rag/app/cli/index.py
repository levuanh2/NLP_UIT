"""Index-building CLI command."""

from pathlib import Path

import typer

from app.core.config import get_settings


def index(
    cache: Path | None = typer.Option(
        None, "--cache", help="Chunk cache; defaults to CACHE_DATA_DIR."
    ),
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild all indexes."),
) -> None:
    """Build local FAISS, SQLite, and BM25 indexes."""
    cache_directory = cache or get_settings().cache_data_dir
    if not cache_directory.is_dir():
        raise typer.BadParameter(f"Cache directory does not exist: {cache_directory}")
    # TODO(phase-implementation):
    # Build dependencies and wire IndexingService.
    typer.echo(
        f"Indexing scaffold ready for {cache_directory} (rebuild={rebuild})."
    )
