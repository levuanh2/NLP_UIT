"""Checksum, checkpoint, SQLite and index-version tests."""

from pathlib import Path

from app.core.checksum import calculate_file_checksum
from app.domain.chunks import ChildChunk, ParentChunk
from app.domain.documents import LegalDocument
from app.indexing.manifest import IndexManifest
from app.indexing.metadata_store.database import Database
from app.indexing.metadata_store.repository import LegalRepository
from app.indexing.versioning import IndexVersionManager
from app.ingestion.checkpoint import (
    IngestionCheckpoint,
    IngestionCheckpointManager,
)


def test_document_checksum_is_streaming_and_stable(tmp_path: Path) -> None:
    source = tmp_path / "large.json"
    source.write_bytes(b"legal-data" * 200_000)

    first = calculate_file_checksum(source, block_size=1024)
    second = calculate_file_checksum(source, block_size=4096)

    assert first == second
    assert len(first) == 64


def test_checkpoint_save_resume_and_checksum_change(tmp_path: Path) -> None:
    manager = IngestionCheckpointManager(tmp_path)
    checkpoint = IngestionCheckpoint(job_id="job-1", index_version="v1")
    manager.save(checkpoint)
    manager.mark_document_completed(10, "old")

    resumed = IngestionCheckpointManager(tmp_path)
    loaded = resumed.load("job-1")

    assert loaded is not None
    assert resumed.is_document_completed(10, "old")
    assert not resumed.is_document_completed(10, "changed")


def test_sqlite_micro_batch_and_document_idempotency(tmp_path: Path) -> None:
    database = Database(tmp_path / "metadata.sqlite")
    database.initialize()
    repository = LegalRepository(database)
    document = LegalDocument(
        document_id=1,
        document_name=None,
        source_link="https://example.test",
        raw_text="Điều 1. Nội dung",
    )
    repository.save_document(document, "abc", "v2", "v1", "completed")
    parent = ParentChunk(
        parent_id="doc:1:parent:0",
        document_id=1,
        chapter=None,
        section=None,
        article="Điều 1",
        text="Điều 1. Nội dung",
        token_count=3,
    )
    child = ChildChunk(
        child_id="doc:1:parent:0:child:0",
        parent_id=parent.parent_id,
        document_id=1,
        chapter=None,
        section=None,
        article="Điều 1",
        clause=None,
        point=None,
        text=parent.text,
        embedding_text=parent.text,
        token_count=3,
    )

    repository.save_chunks([parent], [child])
    repository.save_chunks([parent], [child])

    assert repository.counts() == (1, 1, 1)
    assert repository.is_document_current(1, "abc", "v2", "v1")
    assert repository.validate_chunk_integrity() == []


def ready_manifest(version: str) -> IndexManifest:
    return IndexManifest(
        index_version=version,
        corpus_path="data/corpus",
        document_count=1,
        valid_document_count=1,
        failed_document_count=0,
        parent_count=1,
        child_count=1,
        embedding_model="test",
        embedding_dimension=4,
        chunking_version="v2",
        faiss_index_type="flat",
        status="ready",
    )


def test_index_version_creation_publish_and_rollback(tmp_path: Path) -> None:
    manager = IndexVersionManager(tmp_path)
    first = manager.create_version()
    manager.write_manifest(ready_manifest(first))
    manager.publish(first)
    second = manager.create_version()
    manager.write_manifest(ready_manifest(second))
    manager.publish(second)
    manager.rollback(first)

    assert first == "v1"
    assert second == "v2"
    assert manager.get_current_version() == "v1"


def test_failed_manifest_is_not_published(tmp_path: Path) -> None:
    manager = IndexVersionManager(tmp_path)
    version = manager.create_version()
    manifest = ready_manifest(version).model_copy(update={"status": "failed"})
    manager.write_manifest(manifest)

    try:
        manager.publish(version)
    except ValueError:
        pass
    else:
        raise AssertionError("failed index was published")

    assert manager.get_current_version() is None


def test_manifest_validation_rejects_inconsistent_counts() -> None:
    manifest = ready_manifest("v1").model_copy(
        update={"document_count": 3, "valid_document_count": 1}
    )

    assert manifest.validation_issues()
