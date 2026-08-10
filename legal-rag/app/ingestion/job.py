"""Long-running streaming ingestion with checkpoint and atomic index publish."""

import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.checksum import calculate_file_checksum
from app.domain.chunks import ChildChunk, ParentChunk
from app.domain.documents import LegalDocument
from app.indexing.embeddings.base import BaseEmbeddingModel
from app.indexing.lexical.bm25_index import BM25IndexWriter
from app.indexing.manifest import IndexManifest
from app.indexing.metadata_store.database import Database
from app.indexing.metadata_store.models import IngestionJobRecord
from app.indexing.metadata_store.repository import LegalRepository
from app.indexing.vector_store.writer import FAISSShardWriter, VectorIndexWriter
from app.indexing.versioning import IndexVersionManager
from app.ingestion.checkpoint import (
    IngestionCheckpoint,
    IngestionCheckpointManager,
)
from app.ingestion.chunking.parent_child import ParentChildChunker
from app.ingestion.cleaners.legal_text_cleaner import LegalTextCleaner
from app.ingestion.parsers.factory import DocumentParserFactory
from app.ingestion.structure.extractor import LegalStructureExtractor


class IngestionError(BaseModel):
    document_id: int | None = None
    source_path: str
    error_type: str
    error_message: str


class IngestionJobResult(BaseModel):
    job_id: str
    index_version: str
    documents_processed: int = 0
    documents_failed: int = 0
    parents_created: int = 0
    children_created: int = 0
    embeddings_processed: int = 0
    status: str
    errors: list[IngestionError] = Field(default_factory=list)


ProgressCallback = Callable[[IngestionJobResult, int], None]


class IngestionJob:
    """Process one document and bounded chunk/embedding batches at a time."""

    def __init__(
        self,
        version_manager: IndexVersionManager,
        checkpoint_manager: IngestionCheckpointManager,
        embedding_model: BaseEmbeddingModel,
        parser_factory: DocumentParserFactory | None = None,
        cleaner: LegalTextCleaner | None = None,
        structure_extractor: LegalStructureExtractor | None = None,
        chunker: ParentChildChunker | None = None,
        chunking_version: str = "v2",
        chunk_batch_size: int = 1000,
        embedding_batch_size: int = 64,
        faiss_index_type: str = "auto",
        continue_on_document_error: bool = True,
        report_path: Path = Path("data/outputs/ingestion_report.json"),
        vector_writer_factory: (Callable[[Path, int], VectorIndexWriter] | None) = None,
        bm25_writer_factory: Callable[[Path], BM25IndexWriter] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        if not 1 <= embedding_batch_size <= 128:
            raise ValueError("embedding batch size must be between 1 and 128")
        self.version_manager = version_manager
        self.checkpoint_manager = checkpoint_manager
        self.embedding_model = embedding_model
        self.parser_factory = parser_factory or DocumentParserFactory()
        self.cleaner = cleaner or LegalTextCleaner()
        self.structure_extractor = structure_extractor or LegalStructureExtractor()
        self.chunker = chunker or ParentChildChunker()
        self.chunking_version = chunking_version
        self.chunk_batch_size = chunk_batch_size
        self.embedding_batch_size = embedding_batch_size
        self.faiss_index_type = faiss_index_type
        self.continue_on_document_error = continue_on_document_error
        self.report_path = report_path
        self.vector_writer_factory = vector_writer_factory
        self.bm25_writer_factory = bm25_writer_factory or BM25IndexWriter
        self.progress_callback = progress_callback

    def run(
        self,
        corpus_dir: Path,
        index_version: str,
        *,
        job_id: str | None = None,
        resume: bool = True,
    ) -> IngestionJobResult:
        """Stream context files without materializing corpus-wide objects."""
        if not corpus_dir.is_dir():
            raise ValueError(f"Corpus directory does not exist: {corpus_dir}")
        version_dir = self.version_manager.ensure_version(index_version)
        job_id = job_id or f"ingest-{index_version}"
        paths = sorted(corpus_dir.glob("context_*.json"))
        total_documents = len(paths)

        database = Database(version_dir / "metadata" / "legal.sqlite")
        database.initialize()
        repository = LegalRepository(database)
        checkpoint = self.checkpoint_manager.load(job_id) if resume else None
        if checkpoint and checkpoint.index_version != index_version:
            raise ValueError(
                "Checkpoint index version does not match requested version"
            )
        if checkpoint is None:
            checkpoint = IngestionCheckpoint(job_id=job_id, index_version=index_version)
            self.checkpoint_manager.save(checkpoint)

        result = IngestionJobResult(
            job_id=job_id,
            index_version=index_version,
            documents_processed=checkpoint.documents_processed,
            documents_failed=checkpoint.documents_failed,
            parents_created=checkpoint.chunks_created,
            children_created=checkpoint.children_created,
            status="running",
        )
        repository.save_job(
            IngestionJobRecord(
                job_id=job_id,
                index_version=index_version,
                status="running",
                documents_processed=result.documents_processed,
                documents_failed=result.documents_failed,
                chunks_created=result.parents_created,
                children_created=result.children_created,
            )
        )

        self.embedding_model.load()
        dimension = self.embedding_model.dimension()
        vector_writer = self._vector_writer(version_dir / "faiss", dimension)
        bm25_writer = self.bm25_writer_factory(version_dir / "bm25")
        manifest = IndexManifest(
            index_version=index_version,
            corpus_path=str(corpus_dir.resolve()),
            document_count=total_documents,
            valid_document_count=result.documents_processed,
            failed_document_count=result.documents_failed,
            parent_count=result.parents_created,
            child_count=result.children_created,
            embedding_model=getattr(self.embedding_model, "model_name", "injected"),
            embedding_dimension=dimension,
            chunking_version=self.chunking_version,
            faiss_index_type=self.faiss_index_type,
            status="building",
        )
        self.version_manager.write_manifest(manifest)

        fatal_index_error = False
        try:
            for source_path in paths:
                document: LegalDocument | None = None
                checksum = calculate_file_checksum(source_path)
                path_document_id = self._id_from_path(source_path)
                key = str(path_document_id)
                was_completed = key in checkpoint.completed_documents
                was_failed = key in checkpoint.failed_documents
                if (
                    self.checkpoint_manager.is_document_completed(
                        path_document_id, checksum
                    )
                    and repository.is_document_current(
                        path_document_id,
                        checksum,
                        self.chunking_version,
                        index_version,
                    )
                ) or self.checkpoint_manager.is_document_failed(
                    path_document_id, checksum
                ):
                    continue
                try:
                    document = self.parser_factory.get_parser(source_path).parse(
                        source_path
                    )
                    if self.checkpoint_manager.is_document_completed(
                        document.document_id, checksum
                    ) and repository.is_document_current(
                        document.document_id,
                        checksum,
                        self.chunking_version,
                        index_version,
                    ):
                        continue
                    if not document.raw_text.strip():
                        repository.save_document(
                            document,
                            checksum,
                            self.chunking_version,
                            index_version,
                            "failed",
                        )
                        raise ValueError("Document passage is empty")

                    repository.delete_document_chunks(document.document_id)
                    repository.save_document(
                        document,
                        checksum,
                        self.chunking_version,
                        index_version,
                        "processing",
                        getattr(self.embedding_model, "model_name", "injected"),
                        dimension,
                    )
                    cleaned = self.cleaner.clean(document.raw_text)
                    document = document.model_copy(update={"cleaned_text": cleaned})
                    document = self.structure_extractor.extract(document)
                    parent_count, child_count = self._process_document(
                        document,
                        repository,
                        vector_writer,
                        bm25_writer,
                    )
                    repository.save_document(
                        document,
                        checksum,
                        self.chunking_version,
                        index_version,
                        "completed",
                        getattr(self.embedding_model, "model_name", "injected"),
                        dimension,
                    )
                    if was_failed:
                        result.documents_failed -= 1
                        checkpoint.failed_documents.pop(key, None)
                    if not was_completed:
                        result.documents_processed += 1
                    result.parents_created += parent_count
                    result.children_created += child_count
                    result.embeddings_processed += child_count
                    checkpoint.documents_processed = result.documents_processed
                    checkpoint.chunks_created = result.parents_created
                    checkpoint.children_created = result.children_created
                    checkpoint.last_processed_document_id = document.document_id
                    checkpoint.last_processed_path = str(source_path)
                    self.checkpoint_manager.save(checkpoint)
                    self.checkpoint_manager.mark_document_completed(
                        document.document_id, checksum
                    )
                except Exception as exc:
                    parsed_id = (
                        document.document_id
                        if document is not None
                        else self._id_from_path(source_path)
                    )
                    error = IngestionError(
                        document_id=parsed_id,
                        source_path=str(source_path),
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                    result.errors.append(error)
                    if was_completed:
                        result.documents_processed -= 1
                        checkpoint.completed_documents.pop(key, None)
                    if not was_failed:
                        result.documents_failed += 1
                    checkpoint.documents_failed = result.documents_failed
                    checkpoint.last_processed_document_id = parsed_id
                    checkpoint.last_processed_path = str(source_path)
                    repository.record_error(
                        job_id,
                        str(source_path),
                        error.error_type,
                        error.error_message,
                        parsed_id,
                    )
                    self.checkpoint_manager.save(checkpoint)
                    self.checkpoint_manager.mark_document_failed(parsed_id, checksum)
                    if error.error_type not in {"DocumentParseError", "ValueError"}:
                        fatal_index_error = True
                    if not self.continue_on_document_error:
                        raise
                self._emit_progress(result, total_documents)

            vector_writer.finalize()
            bm25_writer.finalize()
            issues = repository.validate_chunk_integrity()
            _, stored_parents, stored_children = repository.counts()
            if stored_parents != result.parents_created:
                issues.append("SQLite parent count differs from checkpoint")
            if stored_children != result.children_created:
                issues.append("SQLite child count differs from checkpoint")
            if vector_writer.count != stored_children:
                issues.append("FAISS vector count differs from SQLite child count")
            if bm25_writer.count != stored_children:
                issues.append("BM25 count differs from SQLite child count")
            if fatal_index_error:
                issues.append("one or more indexing-stage document errors occurred")

            manifest.valid_document_count = result.documents_processed
            manifest.failed_document_count = result.documents_failed
            manifest.parent_count = stored_parents
            manifest.child_count = stored_children
            manifest.faiss_index_type = getattr(
                vector_writer, "resolved_index_type", self.faiss_index_type
            )
            manifest.status = "ready" if not issues else "failed"
            self.version_manager.write_manifest(manifest)
            if issues:
                result.errors.extend(
                    IngestionError(
                        source_path=str(version_dir),
                        error_type="IndexValidationError",
                        error_message=issue,
                    )
                    for issue in issues
                )
                result.status = "failed"
            else:
                self.version_manager.publish(index_version)
                result.status = "completed"
        finally:
            self.embedding_model.unload()
            bm25_writer.close()

        checkpoint.status = result.status
        self.checkpoint_manager.save(checkpoint)
        repository.save_job(
            IngestionJobRecord(
                job_id=job_id,
                index_version=index_version,
                status=result.status,
                started_at=datetime.now(UTC).replace(tzinfo=None),
                completed_at=datetime.now(UTC).replace(tzinfo=None),
                documents_processed=result.documents_processed,
                documents_failed=result.documents_failed,
                chunks_created=result.parents_created,
                children_created=result.children_created,
            )
        )
        self._write_report(result)
        return result

    def _process_document(
        self,
        document: LegalDocument,
        repository: LegalRepository,
        vector_writer: VectorIndexWriter,
        bm25_writer: BM25IndexWriter,
    ) -> tuple[int, int]:
        parent_buffer: list[ParentChunk] = []
        child_buffer: list[ChildChunk] = []
        parent_count = 0
        child_count = 0
        for parent, children in self.chunker.iter_chunk_groups(document):
            parent_buffer.append(parent)
            child_buffer.extend(children)
            parent_count += 1
            child_count += len(children)
            if len(child_buffer) >= self.chunk_batch_size:
                self._flush(
                    parent_buffer,
                    child_buffer,
                    repository,
                    vector_writer,
                    bm25_writer,
                )
                parent_buffer.clear()
                child_buffer.clear()
        if parent_buffer or child_buffer:
            self._flush(
                parent_buffer,
                child_buffer,
                repository,
                vector_writer,
                bm25_writer,
            )
        return parent_count, child_count

    def _flush(
        self,
        parents: list[ParentChunk],
        children: list[ChildChunk],
        repository: LegalRepository,
        vector_writer: VectorIndexWriter,
        bm25_writer: BM25IndexWriter,
    ) -> None:
        repository.save_chunks(parents, children)
        bm25_writer.add_batch(children)
        for start in range(0, len(children), self.embedding_batch_size):
            batch = children[start : start + self.embedding_batch_size]
            vectors = self.embedding_model.embed_documents(
                [child.embedding_text for child in batch]
            )
            vector_writer.add_batch([child.child_id for child in batch], vectors)

    def _vector_writer(self, output_dir: Path, dimension: int) -> VectorIndexWriter:
        if self.vector_writer_factory:
            return self.vector_writer_factory(output_dir, dimension)
        return FAISSShardWriter(
            output_dir,
            dimension,
            index_type=self.faiss_index_type,
        )

    def _write_report(self, result: IngestionJobResult) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.report_path.with_suffix(".json.tmp")
        temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, self.report_path)

    def _emit_progress(self, result: IngestionJobResult, total: int) -> None:
        if self.progress_callback:
            self.progress_callback(result, total)

    @staticmethod
    def _id_from_path(path: Path) -> int:
        try:
            return int(path.stem.removeprefix("context_"))
        except ValueError:
            return -1
