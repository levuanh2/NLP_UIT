"""Evaluation CLI command."""

from pathlib import Path

import typer


def evaluate(
    dataset: Path = typer.Argument(..., help="Path to local evaluation dataset."),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Optional evaluation report path."
    ),
) -> None:
    """Evaluate retrieval, generation, and citation behavior offline."""
    if not dataset.is_file():
        raise typer.BadParameter(f"Dataset does not exist: {dataset}")
    # TODO(phase-implementation):
    # Build dependencies and wire EvaluationRunner.
    typer.echo(f"Evaluation scaffold ready for {dataset} (output={output}).")
