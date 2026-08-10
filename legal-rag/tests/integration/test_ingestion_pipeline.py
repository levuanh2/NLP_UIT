"""Small end-to-end streaming ingestion fixture."""

import json
from pathlib import Path

import numpy as np

from app.indexing.embeddings.base import BaseEmbeddingModel
from app.indexing.vector_store.writer import VectorIndexWriter
from app.indexing.versioning import IndexVersionManager
from app.ingestion.checkpoint import IngestionCheckpointManager
from app.ingestion.chunking.parent_child import ParentChildChunker
from app.ingestion.job import IngestionJob


class FixtureEmbedding(BaseEmbeddingModel):
    model_name = "fixture-embedding"

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        assert self.loaded
        self.batch_sizes.append(len(texts))
        return np.ones((len(texts), 4), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return np.ones(4, dtype=np.float32)

    def dimension(self) -> int:
        assert self.loaded
        return 4

    def unload(self) -> None:
        self.loaded = False


class FixtureVectorWriter(VectorIndexWriter):
    def __init__(self) -> None:
        self.ids: set[str] = set()

    @property
    def count(self) -> int:
        return len(self.ids)

    def add_batch(self, ids: list[str], vectors: np.ndarray) -> None:
        assert vectors.shape == (len(ids), 4)
        self.ids.update(ids)

    def finalize(self) -> None:
        return None

    def save(self, output_dir: Path) -> None:
        return None


def write_context(path: Path, identifier: int, passage: str, name: str | None) -> None:
    path.write_text(
        json.dumps(
            {
                "id": identifier,
                "name": name,
                "link": f"https://example.test/{identifier}",
                "passage": passage,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_streaming_job_checkpoint_resume_and_atomic_publish(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for identifier in range(1, 5):
        write_context(
            corpus / f"context_{identifier:06d}.json",
            identifier,
            f"Điều {identifier}. Nội dung\n1. Khoản một\n2. Khoản hai",
            None if identifier == 2 else f"Văn bản {identifier}",
        )
    write_context(corpus / "context_000005.json", 5, "", "Văn bản rỗng")
    versions = IndexVersionManager(tmp_path / "indexes")
    checkpoints = IngestionCheckpointManager(tmp_path / "checkpoints")
    embedding = FixtureEmbedding()
    vectors = FixtureVectorWriter()
    job = IngestionJob(
        version_manager=versions,
        checkpoint_manager=checkpoints,
        embedding_model=embedding,
        chunker=ParentChildChunker(
            parent_target_tokens=20,
            parent_max_tokens=30,
            child_target_tokens=8,
            child_max_tokens=12,
            child_overlap_tokens=2,
        ),
        chunk_batch_size=3,
        embedding_batch_size=2,
        report_path=tmp_path / "ingestion_report.json",
        vector_writer_factory=lambda _path, _dimension: vectors,
    )

    first = job.run(corpus, "v1", job_id="fixture-job")
    resumed = job.run(corpus, "v1", job_id="fixture-job", resume=True)

    assert first.status == "completed"
    assert first.documents_processed == 4
    assert first.documents_failed == 1
    assert first.children_created == vectors.count
    assert max(embedding.batch_sizes) <= 2
    assert versions.get_current_version() == "v1"
    assert resumed.documents_processed == first.documents_processed
    assert resumed.children_created == first.children_created
    assert (tmp_path / "ingestion_report.json").is_file()

    write_context(
        corpus / "context_000001.json",
        1,
        "Điều 1. Nội dung đã đổi\n1. Khoản mới",
        "Văn bản 1",
    )
    second_vectors = FixtureVectorWriter()
    changed_job = IngestionJob(
        version_manager=versions,
        checkpoint_manager=checkpoints,
        embedding_model=FixtureEmbedding(),
        chunker=ParentChildChunker(),
        report_path=tmp_path / "ingestion_report_v2.json",
        vector_writer_factory=lambda _path, _dimension: second_vectors,
    )

    changed = changed_job.run(corpus, "v2", job_id="fixture-job-v2")

    assert changed.status == "completed"
    assert changed.documents_processed == 4
    assert versions.get_current_version() == "v2"
