"""Runnable scorer-tuned LegalQA baseline command."""

import json
from pathlib import Path

import typer

from app.baseline.solver import solve_with_answer_memory
from app.submission.formatter import SubmissionFormatter
from app.submission.json_io import load_json_strict
from app.submission.validator import SubmissionValidator
from app.submission.writer import SubmissionWriter


def solve(
    questions: Path = typer.Option(..., "--questions", "-q"),
    train: Path = typer.Option(..., "--train", "-t"),
    internal_output: Path = typer.Option(
        Path("data/outputs/internal-results.json"), "--internal-output"
    ),
    submission_output: Path = typer.Option(
        Path("data/outputs/submission.json"), "--submission-output"
    ),
    word_weight: float = typer.Option(0.75, min=0.0, max=1.0),
    question_weight: float = typer.Option(0.50, min=0.0, max=1.0),
    batch_size: int = typer.Option(128, min=1),
    corpus_index: Path | None = typer.Option(None, "--corpus-index"),
    memory_threshold: float = typer.Option(0.0, min=0.0, max=1.0),
    semantic_model: Path | None = typer.Option(None, "--semantic-model"),
    semantic_cache: Path | None = typer.Option(None, "--semantic-cache"),
    semantic_weight: float = typer.Option(0.75, min=0.0, max=1.0),
    semantic_device: str = typer.Option("cpu", "--semantic-device"),
) -> None:
    """Generate internal results and a validated Subtask 2 submission."""
    if not questions.is_file():
        raise typer.BadParameter(f"Question file does not exist: {questions}")
    if not train.is_file():
        raise typer.BadParameter(f"Training file does not exist: {train}")

    try:
        results = solve_with_answer_memory(
            questions,
            train,
            word_weight=word_weight,
            question_weight=question_weight,
            batch_size=batch_size,
            corpus_index_path=corpus_index,
            memory_threshold=memory_threshold,
            semantic_model_path=semantic_model,
            semantic_cache_path=semantic_cache,
            semantic_weight=semantic_weight,
            semantic_device=semantic_device,
        )
        internal_output.parent.mkdir(parents=True, exist_ok=True)
        with internal_output.open("w", encoding="utf-8") as stream:
            json.dump(results, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

        submission = SubmissionFormatter().format_internal_results(results)
        expected_ids = {result["question_id"] for result in results}
        payload = {key: value.model_dump() for key, value in submission.items()}
        validation = SubmissionValidator().validate(payload, expected_ids)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))
        SubmissionWriter().write(submission, submission_output)
        persisted = load_json_strict(submission_output)
        persisted_validation = SubmissionValidator().validate(persisted, expected_ids)
        if not persisted_validation.valid:
            raise ValueError("; ".join(persisted_validation.errors))
    except (RuntimeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Generated {len(results)} answers; submission: {submission_output}")
