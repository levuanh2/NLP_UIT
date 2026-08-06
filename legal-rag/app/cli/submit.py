"""Submission generation and validation CLI commands."""

import json
from pathlib import Path
from typing import Any

import typer

from app.core.config import get_settings
from app.submission.validator import SubmissionValidator


def submit(
    questions: Path = typer.Option(
        ...,
        "--questions",
        "-q",
        help="Competition question JSON path.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path; defaults to OUTPUT_DIR/SUBMISSION_FILENAME.",
    ),
) -> None:
    """Generate a Subtask 2 submission file."""
    if not questions.is_file():
        raise typer.BadParameter(f"Question file does not exist: {questions}")
    settings = get_settings()
    output_path = output or settings.output_dir / settings.submission_filename
    # TODO(phase-implementation):
    # Load questions, build dependencies, and wire SubmissionService.
    typer.echo(f"Submission scaffold ready: {questions} -> {output_path}.")


def validate_submission(
    submission: Path = typer.Argument(..., help="Submission JSON to validate."),
    questions: Path = typer.Option(
        ..., "--questions", "-q", help="Question JSON used to derive expected IDs."
    ),
) -> None:
    """Validate a submission against expected question IDs."""
    if not submission.is_file():
        raise typer.BadParameter(f"Submission does not exist: {submission}")
    if not questions.is_file():
        raise typer.BadParameter(f"Question file does not exist: {questions}")
    with submission.open(encoding="utf-8") as stream:
        payload: Any = json.load(stream)
    with questions.open(encoding="utf-8") as stream:
        question_payload: Any = json.load(stream)
    expected_ids = _extract_question_ids(question_payload)
    result = SubmissionValidator().validate(payload, expected_ids)
    if not result.valid:
        for error in result.errors:
            typer.echo(error, err=True)
        raise typer.Exit(code=1)
    typer.echo("Submission is valid.")


def _extract_question_ids(payload: Any) -> set[str]:
    """Extract IDs from common object or list question-file shapes."""
    if isinstance(payload, dict):
        return {str(value) for value in payload}
    if isinstance(payload, list):
        ids = {
            str(item["question_id"])
            for item in payload
            if isinstance(item, dict) and "question_id" in item
        }
        if len(ids) != len(payload):
            raise typer.BadParameter(
                "Every list item in the question file needs question_id."
            )
        return ids
    raise typer.BadParameter("Question JSON root must be an object or list.")
