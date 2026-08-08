"""Build persisted dense, BM25, and metadata indexes from cached chunks."""

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

import typer

from app.core.config import get_settings
from app.indexing.embeddings.legal_embedding import VietnameseLegalEmbeddingModel
from app.indexing.lexical.bm25_index import BM25Index
from app.indexing.metadata_store.database import Database
from app.indexing.metadata_store.repository import LegalRepository
from app.indexing.pipeline import IndexingPipeline
from app.indexing.vector_store.faiss_store import FAISSVectorStore
from app.ingestion.cache import (
    iter_child_cache,
    iter_parent_cache,
    load_chunk_cache,
)

T = TypeVar("T")


def index(
    cache: Path | None = typer.Option(
        None, "--cache", help="Chunk cache; defaults to CACHE_DATA_DIR."
    ),
    embedding_model: Path = typer.Option(
        Path("models/vietnamese-legal-embedding"), "--embedding-model"
    ),
    faiss_output: Path = typer.Option(
        Path("storage/faiss/legal.index"), "--faiss-output"
    ),
    bm25_output: Path = typer.Option(Path("storage/bm25/legal.db"), "--bm25-output"),
    metadata_output: Path | None = typer.Option(None, "--metadata-output"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Replace all indexes."),
    max_children: int | None = typer.Option(
        None,
        "--max-children",
        min=1,
        help="Diagnostic subset limit; omit for the required full-corpus build.",
    ),
    stream_batch_size: int = typer.Option(256, min=1),
) -> None:
    """Build full-corpus local indexes from the deterministic chunk cache."""
    settings = get_settings()
    cache_directory = cache or settings.cache_data_dir
    database_path = metadata_output or settings.sqlite_database_path
    if not cache_directory.is_dir():
        raise typer.BadParameter(f"Cache directory does not exist: {cache_directory}")
    if not embedding_model.is_dir():
        raise typer.BadParameter(f"Embedding model does not exist: {embedding_model}")
    if not rebuild and any(
        path.exists() for path in (faiss_output, bm25_output, database_path)
    ):
        raise typer.BadParameter(
            "Index artifacts exist; pass --rebuild to replace them."
        )
    database = Database(database_path)
    database.initialize()
    vector_store = FAISSVectorStore()
    bm25 = BM25Index()
    embedding = VietnameseLegalEmbeddingModel(
        model_name=str(embedding_model),
        device=settings.embedding_device,
        local_files_only=True,
        query_prefix="query: ",
        passage_prefix="passage: ",
    )
    repository = LegalRepository(database)
    if max_children is not None:
        parents, children = load_chunk_cache(cache_directory, max_children=max_children)
        result = IndexingPipeline(embedding, vector_store, bm25, repository).build(
            children, parents
        )
        vector_store.save(faiss_output)
        bm25.save(bm25_output)
        typer.echo(
            f"Indexed {result.child_count} children and {result.parent_count} "
            f"parents at dimension {result.embedding_dimension}."
        )
        return

    repository.reset()
    parent_count = 0
    for batch in _batches(iter_parent_cache(cache_directory), stream_batch_size):
        repository.add_parents(batch)
        parent_count += len(batch)
    embedding.load()
    dimension = embedding.dimension()
    vector_store.create(dimension)
    bm25.create(bm25_output)
    child_count = 0
    for batch in _batches(iter_child_cache(cache_directory), stream_batch_size):
        ids = [item.child_id for item in batch]
        texts = [item.original_text for item in batch]
        vectors = embedding.embed_documents(
            [item.embedding_text or item.original_text for item in batch]
        )
        vector_store.add(vectors, ids)
        bm25.add(texts, ids)
        repository.add_children(batch)
        child_count += len(batch)
        if child_count % 10_000 < len(batch):
            typer.echo(f"Indexed {child_count} child chunks...")
    vector_store.save(faiss_output)
    bm25.save(bm25_output)
    typer.echo(
        f"Indexed {child_count} children and {parent_count} parents "
        f"at dimension {dimension}."
    )


def _batches(values: Iterable[T], size: int) -> Iterator[list[T]]:
    batch: list[T] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
