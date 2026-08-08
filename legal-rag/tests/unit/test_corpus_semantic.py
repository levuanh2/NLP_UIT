"""Semantic corpus reranking tests without loading a neural model."""

from pathlib import Path

import numpy as np

from app.corpus.fts import CorpusEvidence
from app.corpus.semantic import SemanticCorpusReranker


class _FakeModel:
    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        if texts[0].startswith("query:"):
            return np.asarray([[1.0, 0.0]], dtype=np.float32)
        return np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)


def test_semantic_reranker_uses_only_query_and_corpus_passages(tmp_path: Path) -> None:
    reranker = SemanticCorpusReranker(tmp_path, semantic_weight=1.0)
    reranker._model = _FakeModel()
    evidences = [
        CorpusEvidence(1, "A", "", 0, "lexical first", 10.0),
        CorpusEvidence(2, "B", "", 0, "semantic first", 1.0),
    ]

    ranked = reranker.rerank("unseen question", evidences, limit=2)

    assert [item.document_id for item in ranked] == [2, 1]
    assert ranked[0].semantic_score == 1.0
