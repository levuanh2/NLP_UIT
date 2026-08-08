"""Single-question full local hybrid RAG command."""

import json
from pathlib import Path

import typer

from app.core.config import get_settings
from app.domain.queries import LegalQuery
from app.runtime import build_local_rag_service


def ask_rag(
    question: str = typer.Argument(...),
    question_id: str = typer.Option("cli-question", "--question-id"),
    faiss_index: Path = typer.Option(Path("storage/faiss/legal.index")),
    bm25_index: Path = typer.Option(Path("storage/bm25/legal.db")),
    metadata_db: Path = typer.Option(Path("storage/sqlite/legal.db")),
    embedding_model: Path = typer.Option(Path("models/vietnamese-legal-embedding")),
    reranker_model: Path = typer.Option(Path("models/Vietnamese_Reranker")),
    llm_model: Path = typer.Option(Path("models/Vi-Qwen2-1.5B-RAG")),
) -> None:
    """Answer one unseen question using persisted hybrid indexes and local models."""
    if not question.strip() or not question_id.strip():
        raise typer.BadParameter("Question and question ID must not be empty.")
    try:
        service = build_local_rag_service(
            get_settings(),
            faiss_index,
            bm25_index,
            metadata_db,
            embedding_model,
            reranker_model,
            llm_model,
        )
        answer = service.answer(LegalQuery(question_id=question_id, question=question))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(answer.model_dump(), ensure_ascii=False, indent=2))
