"""Single-question CLI command."""

import typer


def ask(
    question: str = typer.Argument(..., help="Vietnamese legal question."),
    question_id: str = typer.Option(
        "cli-question", "--question-id", help="Stable question identifier."
    ),
) -> None:
    """Answer one legal question using local indexes and models."""
    if not question.strip():
        raise typer.BadParameter("Question must not be empty.")
    if not question_id.strip():
        raise typer.BadParameter("Question ID must not be empty.")
    # TODO(phase-implementation):
    # Build dependencies and wire LegalRAGService.
    typer.echo(f"Ask scaffold ready for question_id={question_id}.")
