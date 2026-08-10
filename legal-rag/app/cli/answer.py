"""Bounded local question answering and submission generation command."""

import sys
from pathlib import Path

import typer

from app.core.config import get_settings
from app.domain.generation import GeneratedAnswer
from app.domain.queries import LegalQuery
from app.services.runtime_factory import build_local_rag_runtime
from app.services.submission_service import SubmissionService
from app.submission.formatter import SubmissionFormatter
from app.submission.question_loader import QuestionDatasetLoader
from app.submission.validator import SubmissionValidator
from app.submission.writer import SubmissionWriter


def answer(
    questions: Path = typer.Option(..., "--questions", "-q"),
    output: Path = typer.Option(..., "--output", "-o"),
    limit: int | None = typer.Option(None, min=1),
    start: int = typer.Option(0, min=0),
    end: int | None = typer.Option(None, min=1),
    trace: bool = typer.Option(False),
    fail_fast: bool = typer.Option(False),
) -> None:
    """Answer a bounded question slice using only approved local models."""
    # Windows may expose a legacy CP1252 console even though submission data is
    # UTF-8.  Progress output must never turn a valid Vietnamese answer into a
    # failed question.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    loaded = QuestionDatasetLoader().load(questions)
    if end is not None and end < start:
        raise typer.BadParameter("--end must be greater than or equal to --start")
    selected = loaded[start:end]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise typer.BadParameter("Selected question range is empty")
    settings = get_settings()
    try:
        runtime = build_local_rag_runtime(settings, trace=trace)
    except Exception as exc:
        typer.echo(f"Failed to initialize local Legal RAG runtime: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Loaded local model: {settings.llm_model_name}")
    typer.echo(
        "runtime "
        f"device={runtime.device} dtype={runtime.dtype} "
        f"embedding_load={runtime.load_seconds['embedding']:.3f}s "
        f"reranker_load={runtime.load_seconds['reranker']:.3f}s "
        f"llm_load={runtime.load_seconds['llm']:.3f}s"
    )

    def report(
        query: LegalQuery, generated: GeneratedAnswer, elapsed_seconds: float
    ) -> None:
        attempts = " ".join(
            (
                f"attempt_{item.attempt}_latency={item.latency_seconds:.3f}s "
                f"input_tokens={item.metrics.input_tokens} "
                f"generated_tokens={item.metrics.generated_tokens} "
                f"tokenize_seconds={item.metrics.tokenize_seconds:.3f}s "
                f"generation_seconds={item.metrics.generation_seconds:.3f}s "
                f"decode_seconds={item.metrics.decode_seconds:.3f}s "
                f"validation_seconds={item.validation_seconds:.3f}s "
                f"tokens_per_second={item.metrics.tokens_per_second:.3f}"
            )
            for item in generated.attempts
            if item.latency_seconds is not None
            and item.metrics is not None
            and item.validation_seconds is not None
        )
        typer.echo(
            f"question_id={query.question_id} latency={elapsed_seconds:.3f}s "
            f"grounded={generated.grounded} citations={len(generated.citations)} "
            f"context_tokens={generated.context_tokens} "
            f"evidences={generated.evidence_count} parents={generated.parent_count} "
            f"prompt_build={generated.prompt_build_seconds:.3f}s "
            f"attempts={len(generated.attempts)} {attempts}".rstrip()
        )
        typer.echo(generated.answer)

    try:
        service = SubmissionService(
            runtime.service,
            SubmissionFormatter(),
            SubmissionValidator(),
            SubmissionWriter(),
            fail_fast=fail_fast,
            progress_callback=report,
        )
        result = service.create(selected, output)
        if not result.valid:
            for error in result.errors:
                typer.echo(error, err=True)
            raise typer.Exit(code=1)
        typer.echo(f"Submission validation passed: {output}")
    finally:
        runtime.close()
