"""Full persisted hybrid RAG batch solver."""

import json
import tempfile
from pathlib import Path

import typer

from app.baseline.data import load_question_records
from app.core.config import get_settings
from app.domain.generation import GeneratedAnswer
from app.domain.queries import LegalQuery
from app.runtime import build_local_rag_service
from app.submission.formatter import SubmissionFormatter
from app.submission.validator import SubmissionValidator
from app.submission.writer import SubmissionWriter


def solve_rag(
    questions: Path = typer.Option(..., "--questions", "-q"),
    faiss_index: Path = typer.Option(Path("storage/faiss/legal.index")),
    bm25_index: Path = typer.Option(Path("storage/bm25/legal.db")),
    metadata_db: Path = typer.Option(Path("storage/sqlite/legal.db")),
    embedding_model: Path = typer.Option(Path("models/vietnamese-legal-embedding")),
    reranker_model: Path = typer.Option(Path("models/Vietnamese_Reranker")),
    llm_model: Path = typer.Option(Path("models/Vi-Qwen2-1.5B-RAG")),
    internal_output: Path = typer.Option(Path("data/outputs/internal-results.json")),
    submission_output: Path = typer.Option(Path("data/outputs/submission.json")),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
) -> None:
    """Answer unseen public/private questions with full local hybrid RAG."""
    if not questions.is_file():
        raise typer.BadParameter(f"Question file does not exist: {questions}")
    if submission_output.name != "submission.json":
        raise typer.BadParameter(
            "Submission output file must be named submission.json."
        )
    settings = get_settings()
    try:
        service = build_local_rag_service(
            settings,
            faiss_index,
            bm25_index,
            metadata_db,
            embedding_model,
            reranker_model,
            llm_model,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    records = load_question_records(questions, require_answers=False)
    answers: list[GeneratedAnswer] = []
    if resume and internal_output.is_file():
        payload = json.loads(internal_output.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise typer.BadParameter("Resume file must contain a JSON list.")
        answers = [GeneratedAnswer.model_validate(item) for item in payload]
    completed = {item.question_id for item in answers}
    if len(completed) != len(answers):
        raise typer.BadParameter("Resume file contains duplicate question IDs.")
    expected_ids = {item.question_id for item in records}
    unexpected = completed - expected_ids
    if unexpected:
        raise typer.BadParameter(
            f"Resume file contains unexpected IDs: {', '.join(sorted(unexpected))}"
        )
    pending = [item for item in records if item.question_id not in completed]
    pending_queries = [
        LegalQuery(question_id=item.question_id, question=item.question)
        for item in pending
    ]
    for answer in service.answer_many(pending_queries):
        answers.append(answer)
        _write_internal(answers, internal_output)
    answer_map = {item.question_id: item for item in answers}
    answers = [answer_map[item.question_id] for item in records]
    submission = SubmissionFormatter().format(answers)
    raw = {key: value.model_dump() for key, value in submission.items()}
    validation = SubmissionValidator().validate(raw, expected_ids)
    if not validation.valid:
        raise typer.BadParameter("; ".join(validation.errors))
    SubmissionWriter().write(submission, submission_output)
    typer.echo(f"Generated {len(answers)} grounded answers: {submission_output}")


def _write_internal(answers: list[GeneratedAnswer], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        suffix=".json",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(
            [item.model_dump() for item in answers],
            stream,
            ensure_ascii=False,
            indent=2,
        )
        stream.write("\n")
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
