"""Streaming ingestion CLI command."""

from pathlib import Path

import typer

from app.core.config import get_settings
from app.indexing.embeddings.factory import EmbeddingModelFactory
from app.indexing.versioning import IndexVersionManager
from app.ingestion.checkpoint import IngestionCheckpointManager
from app.ingestion.chunking.parent_child import ParentChildChunker
from app.ingestion.job import IngestionJob, IngestionJobResult


def ingest(
    corpus_dir: Path | None = typer.Option(
        None,
        "--corpus-dir",
        "--source",
        "-s",
        help="Directory containing context_*.json files.",
    ),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    job_id: str | None = typer.Option(None, "--job-id"),
    batch_size: int | None = typer.Option(None, "--batch-size", min=1),
    embedding_batch_size: int | None = typer.Option(
        None, "--embedding-batch-size", min=1, max=128
    ),
    index_version: str | None = typer.Option(None, "--index-version"),
) -> None:
    """Build a checkpointed versioned index document by document."""
    settings = get_settings()
    source = corpus_dir or settings.corpus_data_dir
    if not source.is_dir():
        raise typer.BadParameter(f"Context directory does not exist: {source}")
    version_manager = IndexVersionManager(settings.index_root_dir)
    checkpoint_manager = IngestionCheckpointManager(settings.checkpoint_dir)
    if index_version is None and resume and job_id:
        checkpoint = checkpoint_manager.load(job_id)
        index_version = checkpoint.index_version if checkpoint else None
    version = index_version or version_manager.create_version()
    model = EmbeddingModelFactory.create(
        provider="sentence_transformers",
        model_name=settings.embedding_model_name,
        device=settings.embedding_device,
        local_files_only=settings.model_local_files_only,
    )

    def show_progress(result: IngestionJobResult, total: int) -> None:
        handled = result.documents_processed + result.documents_failed
        if handled % 100 == 0 or handled == total or result.errors:
            typer.echo(
                f"documents={handled}/{total} failed={result.documents_failed} "
                f"parents={result.parents_created} children={result.children_created} "
                f"version={result.index_version} status={result.status}"
            )

    job = IngestionJob(
        version_manager=version_manager,
        checkpoint_manager=checkpoint_manager,
        embedding_model=model,
        chunker=ParentChildChunker(
            parent_target_tokens=settings.parent_target_tokens,
            parent_max_tokens=min(settings.parent_max_tokens, 1200),
            child_target_tokens=settings.child_target_tokens,
            child_max_tokens=min(settings.child_max_tokens, 320),
        ),
        chunking_version=settings.chunking_version,
        chunk_batch_size=batch_size or settings.ingestion_chunk_batch_size,
        embedding_batch_size=(
            embedding_batch_size or settings.ingestion_embedding_batch_size
        ),
        faiss_index_type=settings.faiss_index_type,
        continue_on_document_error=settings.ingestion_continue_on_document_error,
        report_path=settings.output_dir / "ingestion_report.json",
        progress_callback=show_progress,
    )
    typer.echo(
        f"Legal RAG Ingestion: job={job_id or f'ingest-{version}'} version={version}"
    )
    try:
        result = job.run(source, version, job_id=job_id, resume=resume)
    except (RuntimeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    current_version = version_manager.get_current_version()
    typer.echo(
        f"status={result.status} processed={result.documents_processed} "
        f"failed={result.documents_failed} parents={result.parents_created} "
        f"children={result.children_created} current={current_version}"
    )
    if result.status != "completed":
        raise typer.Exit(code=1)
