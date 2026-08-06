"""Legal corpus indexing and extraction tests."""

import json
import zipfile
from pathlib import Path

from app.corpus.fts import (
    LegalCorpusIndex,
    chunk_text,
    extract_answer_span,
    query_terms,
)


def test_query_terms_preserve_vietnamese_and_legal_numbers() -> None:
    terms = query_terms("Mức phạt theo Điều 37 Nghị định 153/2020/NĐ-CP là gì?")

    assert "phạt" in terms
    assert "37" in terms
    assert "153" in terms
    assert "là" not in terms


def test_chunk_text_has_configured_overlap() -> None:
    words = [f"w{index}" for index in range(20)]
    chunks = list(chunk_text(" ".join(words), target_words=10, overlap_words=2))

    assert chunks[0].split()[-2:] == chunks[1].split()[:2]


def test_index_searches_and_extracts_relevant_span(tmp_path: Path) -> None:
    archive_path = tmp_path / "contexts.zip"
    database_path = tmp_path / "corpus.db"
    record = {
        "id": 10,
        "name": "Nghị định xử phạt lao động",
        "link": "https://example.test/10",
        "passage": (
            "Điều 1. Quy định chung về lao động. "
            "Người sử dụng lao động vi phạm nghĩa vụ trả lương bị phạt tiền. "
            "Mức phạt là từ 10.000.000 đồng đến 20.000.000 đồng. "
        )
        * 20,
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "selected-contexts/context_10.json",
            json.dumps(record, ensure_ascii=False),
        )

    documents, chunks = LegalCorpusIndex(database_path).build_from_zip(
        archive_path,
        target_words=100,
        overlap_words=20,
    )
    evidences = LegalCorpusIndex(database_path).search(
        "Mức phạt vi phạm nghĩa vụ trả lương?"
    )
    answer = extract_answer_span(
        "Mức phạt vi phạm nghĩa vụ trả lương?", evidences
    )

    assert documents == 1
    assert chunks >= 1
    assert evidences[0].document_id == 10
    assert "10.000.000 đồng" in answer
