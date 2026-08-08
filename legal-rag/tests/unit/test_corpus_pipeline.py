"""Tests for the private-test-safe persisted corpus pipeline."""

import json
import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from app.cli.main import app
from app.corpus.fts import LegalCorpusIndex, stable_chunk_id, structure_aware_chunks


def test_structure_aware_chunks_have_stable_ids() -> None:
    passage = (
        "Điều 1. Phạm vi điều chỉnh\nNội dung áp dụng chung.\n"
        "Điều 2. Mức phạt\nPhạt tiền từ 1.000.000 đồng đến 2.000.000 đồng."
    )

    chunks = list(structure_aware_chunks(passage, target_words=100, overlap_words=10))

    assert [article for article, _ in chunks] == ["Điều 1", "Điều 2"]
    first_id = stable_chunk_id(42, 0, chunks[0][1])
    assert first_id == stable_chunk_id(42, 0, chunks[0][1])
    assert first_id.startswith("doc-42:chunk-0:")


def test_directory_ingestion_persists_manifest_and_searches(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "context_42.json").write_text(
        json.dumps(
            {
                "id": 42,
                "name": "Nghị định thử nghiệm",
                "link": "https://example.test/42",
                "passage": (
                    "Điều 7. Xử phạt. Hành vi vi phạm bị phạt tiền "
                    "5.000.000 đồng và phải khắc phục hậu quả."
                ),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    database = tmp_path / "legal.db"

    counts = LegalCorpusIndex(database).build_from_directory(
        corpus, target_words=100, overlap_words=10
    )
    evidence = LegalCorpusIndex(database).search("phạt tiền 5.000.000 đồng", limit=1)

    assert counts == (1, 1)
    assert evidence[0].document_id == 42
    assert evidence[0].article == "Điều 7"
    with sqlite3.connect(database) as connection:
        manifest = connection.execute(
            "SELECT schema_version, document_count, chunk_count FROM manifest"
        ).fetchone()
    assert manifest == (2, 1, 1)


def test_solve_corpus_writes_exact_submission_schema(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "context_1.json").write_text(
        json.dumps(
            {
                "id": 1,
                "name": "Luật mẫu",
                "link": "https://example.test/1",
                "passage": "Điều 1. Người lao động được nghỉ phép mười hai ngày.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    database = tmp_path / "legal.db"
    LegalCorpusIndex(database).build_from_directory(
        corpus, target_words=100, overlap_words=10
    )
    questions = tmp_path / "private.json"
    questions.write_text(
        json.dumps({"unseen-1": {"question": "Nghỉ phép bao nhiêu ngày?"}}),
        encoding="utf-8",
    )
    internal = tmp_path / "internal.json"
    submission = tmp_path / "submission.json"

    result = CliRunner().invoke(
        app,
        [
            "solve-corpus",
            "--questions",
            str(questions),
            "--corpus-index",
            str(database),
            "--internal-output",
            str(internal),
            "--submission-output",
            str(submission),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(submission.read_text(encoding="utf-8")) == {
        "unseen-1": {"answer": "Điều 1. Người lao động được nghỉ phép mười hai ngày."}
    }
