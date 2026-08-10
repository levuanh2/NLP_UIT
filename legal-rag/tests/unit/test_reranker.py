"""Local reranker adapter contract tests."""

import pytest

from app.domain.retrieval import RetrievalCandidate
from app.retrieval.reranker import Reranker


def test_reranker_interface() -> None:
    reranker = Reranker(
        "local/missing-model",
        "cpu",
        local_files_only=True,
        trust_remote_code=False,
    )
    candidate = RetrievalCandidate(child_id="c1", source="rrf", rank=1)

    with pytest.raises(RuntimeError, match="not loaded"):
        reranker.rerank("query", [candidate])
