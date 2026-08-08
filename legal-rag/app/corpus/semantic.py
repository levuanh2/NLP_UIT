"""Local semantic reranking over corpus chunks retrieved from persisted FTS."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from app.corpus.fts import CorpusEvidence


class SemanticCorpusReranker:
    """Rerank stored corpus evidence without using training questions or answers."""

    def __init__(
        self,
        model_path: Path,
        device: str = "cpu",
        semantic_weight: float = 0.75,
    ) -> None:
        if not model_path.is_dir():
            raise ValueError(f"Semantic model does not exist: {model_path}")
        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("semantic_weight must be between 0 and 1.")
        self.model_path = model_path
        self.device = device
        self.semantic_weight = semantic_weight
        self._model: Any | None = None

    def rerank(
        self,
        question: str,
        evidences: list[CorpusEvidence],
        limit: int,
    ) -> list[CorpusEvidence]:
        if not evidences or limit <= 0:
            return []
        model = self._load_model()
        query = model.encode(
            [f"query: {question}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        passages = model.encode(
            [f"passage: {item.text}" for item in evidences],
            batch_size=min(32, len(evidences)),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        semantic = np.asarray(passages @ query, dtype=np.float32)
        lexical = np.asarray([item.score for item in evidences], dtype=np.float32)
        semantic_normalized = _minmax(semantic)
        lexical_normalized = _minmax(lexical)
        hybrid = (
            self.semantic_weight * semantic_normalized
            + (1.0 - self.semantic_weight) * lexical_normalized
        )
        ranked = sorted(
            (
                replace(
                    item,
                    semantic_score=float(semantic[index]),
                    hybrid_score=float(hybrid[index]),
                )
                for index, item in enumerate(evidences)
            ),
            key=lambda item: item.hybrid_score or 0.0,
            reverse=True,
        )
        return ranked[:limit]

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for semantic corpus reranking."
                ) from exc
            self._model = SentenceTransformer(
                str(self.model_path),
                device=self.device,
                local_files_only=True,
            )
        return self._model


def _minmax(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min())
    span = float(values.max()) - minimum
    if span <= 1e-12:
        return np.ones_like(values)
    return (values - minimum) / span
