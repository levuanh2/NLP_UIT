"""Legal corpus FTS index CLI."""

from pathlib import Path

import typer

from app.corpus.fts import LegalCorpusIndex


def build_corpus_index(
    corpus_zip: Path = typer.Option(..., "--corpus-zip"),
    output: Path = typer.Option(
        Path("storage/sqlite/legal_corpus_fts.db"), "--output", "-o"
    ),
    target_words: int = typer.Option(350, min=100),
    overlap_words: int = typer.Option(60, min=0),
) -> None:
    """Build a Unicode-aware legal corpus search index."""
    try:
        documents, chunks = LegalCorpusIndex(output).build_from_zip(
            corpus_zip,
            target_words=target_words,
            overlap_words=overlap_words,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Indexed {documents} documents into {chunks} chunks: {output}")
