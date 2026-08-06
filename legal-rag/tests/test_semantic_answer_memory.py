from pathlib import Path

import numpy as np
import pytest

from app.baseline.data import QuestionRecord
from app.baseline.semantic import (
    SemanticHybridAnswerMemoryRetriever,
    row_minmax,
)


def test_row_minmax_normalizes_each_query() -> None:
    scores = np.asarray([[2.0, 4.0, 3.0], [7.0, 7.0, 7.0]], dtype=np.float32)

    normalized = row_minmax(scores)

    assert normalized.tolist() == [[0.0, 1.0, 0.5], [0.0, 0.0, 0.0]]


def test_semantic_cache_ids_must_match_training(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    cache_path = tmp_path / "cache.npz"
    np.savez(
        cache_path,
        ids=np.asarray(["wrong-id"]),
        question_embeddings=np.ones((1, 2), dtype=np.float32),
    )
    retriever = SemanticHybridAnswerMemoryRetriever(model_path, cache_path)
    records = [QuestionRecord("expected-id", "Câu hỏi?", "Câu trả lời")]

    with pytest.raises(ValueError, match="cache IDs"):
        retriever.fit(records)
