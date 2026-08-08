"""Corpus-only LegalQA batch solver for public and unseen private questions."""

import json
from dataclasses import asdict
from pathlib import Path

import typer

from app.baseline.data import load_question_records
from app.corpus.fts import LegalCorpusIndex, extract_answer_span
from app.corpus.semantic import SemanticCorpusReranker
from app.submission.formatter import SubmissionFormatter
from app.submission.validator import SubmissionValidator
from app.submission.writer import SubmissionWriter


def solve_corpus(
    questions: Path = typer.Option(..., "--questions", "-q"),
    corpus_index: Path = typer.Option(..., "--corpus-index", "-i"),
    internal_output: Path = typer.Option(
        Path("data/outputs/internal-results.json"), "--internal-output"
    ),
    submission_output: Path = typer.Option(
        Path("data/outputs/submission.json"), "--submission-output"
    ),
    top_k: int = typer.Option(5, "--top-k", min=1),
    max_answer_words: int = typer.Option(360, "--max-answer-words", min=30),
    semantic_model: Path | None = typer.Option(None, "--semantic-model"),
    semantic_device: str = typer.Option("cpu", "--semantic-device"),
    semantic_weight: float = typer.Option(0.75, min=0.0, max=1.0),
    semantic_top_k: int = typer.Option(2, min=1),
) -> None:
    """Retrieve from stored corpus chunks and write the exact scorer payload."""
    if not questions.is_file():
        raise typer.BadParameter(f"Question file does not exist: {questions}")
    if not corpus_index.is_file():
        raise typer.BadParameter(f"Corpus index does not exist: {corpus_index}")
    if submission_output.name != "submission.json":
        raise typer.BadParameter(
            "Submission output file must be named submission.json."
        )

    question_records = load_question_records(questions, require_answers=False)
    index = LegalCorpusIndex(corpus_index)
    reranker = (
        SemanticCorpusReranker(
            semantic_model,
            device=semantic_device,
            semantic_weight=semantic_weight,
        )
        if semantic_model is not None
        else None
    )
    results: list[dict[str, object]] = []
    missing: list[str] = []
    for record in question_records:
        evidences = index.search(record.question, limit=top_k)
        if reranker is not None:
            evidences = reranker.rerank(
                record.question,
                evidences,
                limit=min(semantic_top_k, len(evidences)),
            )
        answer = extract_answer_span(
            record.question, evidences, max_words=max_answer_words
        )
        if not answer:
            missing.append(record.question_id)
            continue
        results.append(
            {
                "question_id": record.question_id,
                "question": record.question,
                "answer": answer,
                "method": (
                    "persisted_corpus_fts_semantic_extractive"
                    if reranker is not None
                    else "persisted_corpus_fts_extractive"
                ),
                "corpus_evidence": [asdict(item) for item in evidences],
            }
        )
    if missing:
        preview = ", ".join(missing[:10])
        raise typer.BadParameter(
            f"No grounded evidence for {len(missing)} questions: {preview}"
        )

    internal_output.parent.mkdir(parents=True, exist_ok=True)
    with internal_output.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    submission = SubmissionFormatter().format_internal_results(results)
    expected_ids = {record.question_id for record in question_records}
    payload = {key: value.model_dump() for key, value in submission.items()}
    validation = SubmissionValidator().validate(payload, expected_ids)
    if not validation.valid:
        raise typer.BadParameter("; ".join(validation.errors))
    SubmissionWriter().write(submission, submission_output)
    typer.echo(
        f"Answered {len(results)} questions from stored corpus chunks; "
        f"submission: {submission_output}"
    )
