"""Submission generation and validation CLI commands."""

from pathlib import Path
from typing import Any

import typer

from app.core.exceptions import SubmissionValidationError
from app.submission.formatter import SubmissionFormatter
from app.submission.json_io import load_json_strict
from app.submission.validator import SubmissionValidator
from app.submission.writer import SubmissionWriter


def submit(
    questions: Path = typer.Option(
        ...,
        "--questions",
        "-q",
        help="Competition question JSON path.",
    ),
    answers: Path = typer.Option(
        ...,
        "--answers",
        "-a",
        help="Internal RAG results JSON containing question_id and answer.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path; defaults to data/outputs/submission.json.",
    ),
) -> None:
    """Generate a Subtask 2 submission file."""
    if not questions.is_file():
        raise typer.BadParameter(f"Question file does not exist: {questions}")
    if not answers.is_file():
        raise typer.BadParameter(f"Internal answer file does not exist: {answers}")
    output_path = output or Path("data/outputs/submission.json")
    if output_path.name != "submission.json":
        raise typer.BadParameter("Output file must be named submission.json.")
    try:
        question_payload = load_json_strict(questions)
        internal_results = load_json_strict(answers)
        expected_ids = _extract_question_ids(question_payload)
        submission = SubmissionFormatter().format_internal_results(internal_results)
        raw_submission = {
            question_id: answer.model_dump()
            for question_id, answer in submission.items()
        }
        result = SubmissionValidator().validate(raw_submission, expected_ids)
        if not result.valid:
            for error in result.errors:
                typer.echo(error, err=True)
            raise typer.Exit(code=1)

        SubmissionWriter().write(submission, output_path)
        persisted = load_json_strict(output_path)
        persisted_result = SubmissionValidator().validate(persisted, expected_ids)
        if not persisted_result.valid:
            raise SubmissionValidationError(
                "Written submission failed validation: "
                + "; ".join(persisted_result.errors)
            )
    except (SubmissionValidationError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Valid submission written: {output_path}")


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
    if submission.name != "submission.json":
        raise typer.BadParameter("Submission file must be named submission.json.")
    try:
        payload = load_json_strict(submission)
        question_payload = load_json_strict(questions)
        expected_ids = _extract_question_ids(question_payload)
    except SubmissionValidationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    result = SubmissionValidator().validate(payload, expected_ids)
    if not result.valid:
        for error in result.errors:
            typer.echo(error, err=True)
        raise typer.Exit(code=1)
    typer.echo("Submission is valid.")


def _extract_question_ids(payload: Any) -> set[str]:
    """Extract IDs from common object or list question-file shapes."""
    if isinstance(payload, dict):
        ids = set(payload)
        if not all(isinstance(value, str) and value.strip() for value in ids):
            raise typer.BadParameter("Every question ID must be a nonempty string.")
        return ids
    if isinstance(payload, list):
        raw_ids: list[str] = []
        for item in payload:
            value = item.get("question_id") if isinstance(item, dict) else None
            if not isinstance(value, str) or not value.strip():
                raise typer.BadParameter(
                    "Every list item needs a nonempty string question_id."
                )
            raw_ids.append(value)
        ids = set(raw_ids)
        if len(ids) != len(raw_ids):
            raise typer.BadParameter("Question input contains duplicate IDs.")
        return ids
    raise typer.BadParameter("Question JSON root must be an object or list.")
