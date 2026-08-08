"""Full local RAG evaluation command."""

import json
from pathlib import Path

import typer

from app.core.config import get_settings
from app.evaluation.dataset import EvaluationDatasetLoader
from app.evaluation.runner import EvaluationRunner
from app.runtime import build_local_rag_service


def evaluate(
    dataset: Path = typer.Argument(..., help="Path to local evaluation dataset."),
    output: Path | None = typer.Option(None, "--output", "-o"),
    faiss_index: Path = typer.Option(Path("storage/faiss/legal.index")),
    bm25_index: Path = typer.Option(Path("storage/bm25/legal.db")),
    metadata_db: Path = typer.Option(Path("storage/sqlite/legal.db")),
    embedding_model: Path = typer.Option(Path("models/vietnamese-legal-embedding")),
    reranker_model: Path = typer.Option(Path("models/Vietnamese_Reranker")),
    llm_model: Path = typer.Option(Path("models/Vi-Qwen2-1.5B-RAG")),
) -> None:
    """Evaluate retrieval, generation, grounding, and citations offline."""
    if not dataset.is_file():
        raise typer.BadParameter(f"Dataset does not exist: {dataset}")
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
        samples = EvaluationDatasetLoader().load(dataset)
        report = EvaluationRunner(service).run(samples)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    rendered = json.dumps(report.model_dump(), ensure_ascii=False, indent=2) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    typer.echo(rendered)
