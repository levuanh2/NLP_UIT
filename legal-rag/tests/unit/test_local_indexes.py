"""Persistence tests for the local BM25 and FAISS indexes."""

from pathlib import Path

import numpy as np
import pytest

from app.indexing.lexical.bm25_index import BM25Index
from app.indexing.vector_store.faiss_store import FAISSVectorStore


def test_bm25_round_trip_and_allowed_ids(tmp_path: Path) -> None:
    path = tmp_path / "bm25.db"
    index = BM25Index()
    index.build(
        ["phạt tiền năm triệu đồng", "người lao động được nghỉ phép"],
        ["fine", "leave"],
    )
    index.save(path)
    restored = BM25Index()
    restored.load(path)

    assert restored.search("phạt tiền", 1)[0][0] == "fine"
    assert restored.search("phạt tiền nghỉ phép", 1, {"leave"})[0][0] == "leave"


def test_faiss_round_trip_uses_stable_external_ids(tmp_path: Path) -> None:
    path = tmp_path / "vectors.index"
    store = FAISSVectorStore()
    store.create(2)
    store.add(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ["a", "b"],
    )
    store.save(path)
    restored = FAISSVectorStore()
    restored.load(path)

    result = restored.search(np.asarray([0.9, 0.1]), 1)
    assert result[0][0] == "a"
    assert result[0][1] == pytest.approx(0.9)
    assert restored.search(np.asarray([0.9, 0.1]), 1, {"b"})[0][0] == "b"
    batched = restored.search_many(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        1,
        [None, {"b"}],
    )
    assert [items[0][0] for items in batched] == ["a", "b"]
