"""Submission generation and validation CLI commands."""

import json
import time
from pathlib import Path
from typing import Any

import typer

from app.core.config import get_settings
from app.domain.generation import GeneratedAnswer
from app.domain.queries import LegalQuery
from app.generation.prompts.system import ABSTENTION_ANSWER
from app.services.runtime_factory import build_local_rag_runtime
from app.services.submission_service import SubmissionService
from app.submission.formatter import SubmissionFormatter
from app.submission.question_loader import QuestionDatasetLoader
from app.submission.validator import SubmissionValidator
from app.submission.writer import SubmissionWriter


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
    checkpoint: Path | None = typer.Option(
        None,
        "--checkpoint",
        help="JSONL of answered questions; defaults to <output>.partial.jsonl.",
    ),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    limit: int | None = typer.Option(
        None, "--limit", min=1, help="Answer only the first N questions."
    ),
    fail_fast: bool = typer.Option(False, "--fail-fast"),
    require_grounded: bool = typer.Option(
        False,
        "--require-grounded/--no-require-grounded",
        help="Discard answers that fail grounding validation. Off by default: "
        "METEOR/ROUGE-L score token overlap and never penalise unsupported "
        "content, so a discarded answer scores zero instead of partial credit.",
    ),
    abstain_on_failure: bool = typer.Option(
        True,
        "--abstain-on-failure/--no-abstain-on-failure",
        help="Write a standard abstention for questions that fail validation, "
        "so the submission still carries every expected question ID.",
    ),
) -> None:
    """Generate a Subtask 2 submission file."""
    if not questions.is_file():
        raise typer.BadParameter(f"Question file does not exist: {questions}")
    settings = get_settings()
    output_path = output or settings.output_dir / settings.submission_filename
    checkpoint_path = checkpoint or output_path.with_suffix(".partial.jsonl")
    dataset = QuestionDatasetLoader().load(questions)
    if limit is not None:
        dataset = dataset[:limit]

    runtime = build_local_rag_runtime(settings)
    typer.echo(
        f"models loaded on {runtime.device} ({runtime.dtype}) "
        f"params={runtime.parameter_counts['total']:,} "
        f"load={sum(runtime.load_seconds.values()):.1f}s"
    )
    started = time.perf_counter()
    answered = 0

    def show_progress(query: LegalQuery, _: GeneratedAnswer, seconds: float) -> None:
        nonlocal answered
        answered += 1
        rate = (time.perf_counter() - started) / answered
        remaining = (len(dataset) - answered) * rate
        typer.echo(
            f"{answered}/{len(dataset)} {query.question_id} {seconds:.1f}s "
            f"eta={remaining / 60:.0f}m"
        )

    service = SubmissionService(
        runtime.service,
        SubmissionFormatter(),
        SubmissionValidator(),
        SubmissionWriter(
            encoding=settings.submission_encoding,
            ensure_ascii=settings.submission_ensure_ascii,
        ),
        fail_fast=fail_fast,
        progress_callback=show_progress,
        checkpoint_path=checkpoint_path if resume else None,
        fallback_answer=ABSTENTION_ANSWER if abstain_on_failure else None,
        failure_path=output_path.with_suffix(".failures.jsonl"),
        require_grounded=require_grounded,
    )
    try:
        result = service.create(dataset, output_path)
    finally:
        runtime.close()
    if service.failures:
        typer.echo(
            f"{len(service.failures)} question(s) abstained; "
            f"first: {service.failures[0]}",
            err=True,
        )
    if not result.valid:
        for error in result.errors[:20]:
            typer.echo(error, err=True)
        typer.echo(
            f"{len(result.errors)} error(s); answers kept in {checkpoint_path}",
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(f"Submission written: {output_path}")


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
