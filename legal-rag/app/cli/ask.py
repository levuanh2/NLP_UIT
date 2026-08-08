"""Single-question corpus retrieval command."""

import json
from dataclasses import asdict
from pathlib import Path

import typer

from app.core.config import get_settings
from app.corpus.fts import LegalCorpusIndex, extract_answer_span
from app.corpus.semantic import SemanticCorpusReranker


def ask(
    question: str = typer.Argument(..., help="Vietnamese legal question."),
    question_id: str = typer.Option(
        "cli-question", "--question-id", help="Stable question identifier."
    ),
    corpus_index: Path | None = typer.Option(
        None, "--corpus-index", help="Persisted corpus index path."
    ),
    top_k: int = typer.Option(5, "--top-k", min=1),
    semantic_model: Path | None = typer.Option(None, "--semantic-model"),
    semantic_device: str = typer.Option("cpu", "--semantic-device"),
    semantic_weight: float = typer.Option(0.75, min=0.0, max=1.0),
    semantic_top_k: int = typer.Option(2, min=1),
) -> None:
    """Answer one legal question exclusively from the persisted corpus index."""
    if not question.strip():
        raise typer.BadParameter("Question must not be empty.")
    if not question_id.strip():
        raise typer.BadParameter("Question ID must not be empty.")
    database_path = corpus_index or get_settings().sqlite_database_path
    if not database_path.is_file():
        raise typer.BadParameter(f"Corpus index does not exist: {database_path}")
    evidences = LegalCorpusIndex(database_path).search(question, limit=top_k)
    if semantic_model is not None:
        evidences = SemanticCorpusReranker(
            semantic_model,
            device=semantic_device,
            semantic_weight=semantic_weight,
        ).rerank(question, evidences, limit=min(semantic_top_k, len(evidences)))
    answer = extract_answer_span(question, evidences)
    if not answer:
        typer.echo("No grounded corpus evidence found.", err=True)
        raise typer.Exit(code=2)
    typer.echo(
        json.dumps(
            {
                "question_id": question_id,
                "answer": answer,
                "evidence": [asdict(item) for item in evidences],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
