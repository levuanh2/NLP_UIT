"""Hybrid TF-IDF retrieval over expert LegalQA training answers."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.baseline.data import QuestionRecord


@dataclass(frozen=True, slots=True)
class RetrievalMatch:
    """Nearest expert question and its answer."""

    source_question_id: str
    source_question: str
    answer: str
    score: float
    score_margin: float


class AnswerMemoryRetriever:
    """Retrieve expert answers with a scorer-tuned word/character ensemble."""

    def __init__(
        self,
        word_weight: float = 0.75,
        question_weight: float = 0.50,
        batch_size: int = 128,
    ) -> None:
        if not 0 <= word_weight <= 1:
            raise ValueError("word_weight must be between 0 and 1.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not 0 <= question_weight <= 1:
            raise ValueError("question_weight must be between 0 and 1.")
        self.word_weight = word_weight
        self.question_weight = question_weight
        self.batch_size = batch_size
        self._records: list[QuestionRecord] = []
        self._word_vectorizer: Any = None
        self._char_vectorizer: Any = None
        self._answer_vectorizer: Any = None
        self._word_matrix: Any = None
        self._char_matrix: Any = None
        self._answer_matrix: Any = None
        self._exact_matches: dict[str, int] = {}

    @staticmethod
    def normalize_question(text: str) -> str:
        """Normalize spacing and case without destroying Vietnamese accents."""
        return " ".join(text.casefold().split())

    def fit(self, records: list[QuestionRecord]) -> None:
        """Build word and character TF-IDF indexes over training questions."""
        if not records:
            raise ValueError("At least one training question is required.")
        if any(not record.answer for record in records):
            raise ValueError("Every training record must have an answer.")

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:  # pragma: no cover - dependency error
            raise RuntimeError(
                "scikit-learn is required for the LegalQA baseline."
            ) from exc

        self._records = list(records)
        questions = [record.question for record in records]
        self._word_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        self._char_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(4, 6),
            min_df=2 if len(records) >= 3 else 1,
            max_features=200_000,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        self._answer_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2 if len(records) >= 3 else 1,
            max_features=250_000,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        self._word_matrix = self._word_vectorizer.fit_transform(questions)
        self._char_matrix = self._char_vectorizer.fit_transform(questions)
        self._answer_matrix = self._answer_vectorizer.fit_transform(
            [record.answer or "" for record in records]
        )
        self._exact_matches = {}
        for index, question in enumerate(questions):
            self._exact_matches.setdefault(self.normalize_question(question), index)

    def retrieve(self, questions: list[str]) -> list[RetrievalMatch]:
        """Return one deterministic best expert answer for every question."""
        if not questions:
            return []

        matches: list[RetrievalMatch] = []
        for start in range(0, len(questions), self.batch_size):
            batch = questions[start : start + self.batch_size]
            scores = self.score_questions(batch)

            for row_index, question in enumerate(batch):
                exact_index = self._exact_matches.get(self.normalize_question(question))
                row = scores[row_index]
                if exact_index is not None:
                    best_index = exact_index
                    best_score = 1.0
                    second_score = (
                        float(np.partition(row, -2)[-2]) if len(row) > 1 else 0.0
                    )
                else:
                    top_count = min(2, len(row))
                    top_indices = np.argpartition(row, -top_count)[-top_count:]
                    ordered = top_indices[np.argsort(row[top_indices])[::-1]]
                    best_index = int(ordered[0])
                    best_score = float(row[best_index])
                    second_score = float(row[ordered[1]]) if len(ordered) > 1 else 0.0
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

    def score_questions(self, questions: list[str]) -> np.ndarray:
        """Return dense candidate scores for ensembling with other retrievers."""
        if self._word_matrix is None or self._char_matrix is None:
            raise RuntimeError("AnswerMemoryRetriever.fit must be called first.")
        if not questions:
            return np.empty((0, len(self._records)), dtype=np.float32)
        word_queries = self._word_vectorizer.transform(questions)
        char_queries = self._char_vectorizer.transform(questions)
        answer_queries = self._answer_vectorizer.transform(questions)
        word_scores = word_queries @ self._word_matrix.T
        char_scores = char_queries @ self._char_matrix.T
        question_scores = word_scores.multiply(self.word_weight) + char_scores.multiply(
            1 - self.word_weight
        )
        answer_scores = answer_queries @ self._answer_matrix.T
        return (
            question_scores.multiply(self.question_weight)
            + answer_scores.multiply(1 - self.question_weight)
        ).toarray()
