"""Semantic ensemble for expert LegalQA answer memory."""

from pathlib import Path
from typing import Any

import numpy as np

from app.baseline.data import QuestionRecord
from app.baseline.retriever import AnswerMemoryRetriever, RetrievalMatch


def row_minmax(scores: np.ndarray) -> np.ndarray:
    """Normalize every query row for stable cross-retriever score fusion."""
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D matrix.")
    if scores.shape[1] == 0:
        return scores.astype(np.float32, copy=True)
    minimum = scores.min(axis=1, keepdims=True)
    scale = scores.max(axis=1, keepdims=True) - minimum
    return ((scores - minimum) / np.maximum(scale, 1e-6)).astype(np.float32)


class SemanticHybridAnswerMemoryRetriever:
    """Fuse TF-IDF and Vietnamese legal embeddings over training questions."""

    def __init__(
        self,
        model_path: Path,
        cache_path: Path,
        semantic_weight: float = 0.75,
        word_weight: float = 0.75,
        question_weight: float = 0.50,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight must be between 0 and 1.")
        self.model_path = model_path
        self.cache_path = cache_path
        self.semantic_weight = semantic_weight
        self.batch_size = batch_size
        self.device = device
        self._lexical = AnswerMemoryRetriever(
            word_weight=word_weight,
            question_weight=question_weight,
            batch_size=batch_size,
        )
        self._records: list[QuestionRecord] = []
        self._passages: np.ndarray | None = None
        self._model: Any = None

    def fit(self, records: list[QuestionRecord]) -> None:
        """Validate the prebuilt cache and initialize both retrieval branches."""
        if not self.model_path.is_dir():
            raise ValueError(
                f"Semantic model directory does not exist: {self.model_path}"
            )
        if not self.cache_path.is_file():
            raise ValueError(f"Semantic cache does not exist: {self.cache_path}")
        with np.load(self.cache_path) as cache:
            if "ids" not in cache or "question_embeddings" not in cache:
                raise ValueError("Semantic cache is missing required arrays.")
            cache_ids = cache["ids"].tolist()
            passages = cache["question_embeddings"].astype(np.float32)
        expected_ids = [record.question_id for record in records]
        if cache_ids != expected_ids:
            raise ValueError("Semantic cache IDs do not match the training dataset.")
        if passages.ndim != 2 or passages.shape[0] != len(records):
            raise ValueError("Semantic cache has an invalid embedding shape.")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency error
            raise RuntimeError(
                "sentence-transformers is required for semantic mode."
            ) from exc
        self._records = list(records)
        self._passages = passages
        self._lexical.fit(records)
        self._model = SentenceTransformer(
            str(self.model_path),
            device=self.device,
            local_files_only=True,
            trust_remote_code=True,
        )

    def retrieve(self, questions: list[str]) -> list[RetrievalMatch]:
        """Return the best copied expert answer using tuned score fusion."""
        if self._model is None or self._passages is None:
            raise RuntimeError("Semantic retriever fit must be called first.")
        if not questions:
            return []
        matches: list[RetrievalMatch] = []
        for start in range(0, len(questions), self.batch_size):
            batch = questions[start : start + self.batch_size]
            query_embeddings = self._model.encode(
                [f"query: {question}" for question in batch],
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
            lexical = row_minmax(self._lexical.score_questions(batch))
            semantic = row_minmax(query_embeddings @ self._passages.T)
            scores = (
                1 - self.semantic_weight
            ) * lexical + self.semantic_weight * semantic
            for row in scores:
                top_count = min(2, len(row))
                top = np.argpartition(row, -top_count)[-top_count:]
                ordered = top[np.argsort(row[top])[::-1]]
                best_index = int(ordered[0])
                second_score = float(row[ordered[1]]) if len(ordered) > 1 else 0.0
                best_score = float(row[best_index])
                source = self._records[best_index]
                matches.append(
                    RetrievalMatch(
                        source_question_id=source.question_id,
                        source_question=source.question,
                        answer=source.answer or "",
                        score=best_score,
                        score_margin=max(0.0, best_score - second_score),
                    )
                )
        return matches
