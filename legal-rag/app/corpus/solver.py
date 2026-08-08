"""Corpus retrieval solver: query stored legal chunks and format answers."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.baseline.data import load_question_records
from app.corpus.fts import LegalCorpusIndex, extract_answer_span


def solve_with_corpus_retrieval(
    questions_path: Path,
    corpus_index_path: Path,
    top_k: int = 5,
    max_answer_words: int = 360,
) -> list[dict[str, Any]]:
    """Answer questions by retrieving evidence from a persisted corpus FTS index."""
    if not corpus_index_path.is_file():
        raise ValueError(f"Corpus index does not exist: {corpus_index_path}")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    questions = load_question_records(questions_path, require_answers=False)
    corpus_index = LegalCorpusIndex(corpus_index_path)
    results: list[dict[str, Any]] = []

    for question in questions:
        evidences = corpus_index.search(question.question, limit=top_k)
        answer = extract_answer_span(
            question.question,
            evidences,
            max_words=max_answer_words,
        )
        if not answer.strip() and evidences:
            answer = " ".join(evidences[0].text.split()[:max_answer_words]).strip()
        if not answer.strip():
            answer = question.question.strip()

        top_score = evidences[0].score if evidences else 0.0
        second_score = evidences[1].score if len(evidences) > 1 else 0.0
        results.append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "answer": answer,
                "retrieval": {
                    "source_question_id": None,
                    "source_question": None,
                    "answer": answer,
                    "score": top_score,
                    "score_margin": top_score - second_score,
                },
                "corpus_evidence": [asdict(evidence) for evidence in evidences],
                "method": "corpus_fts_extractive",
                "confidence": _confidence_label(top_score, top_score - second_score),
            }
        )
    return results


def _confidence_label(score: float, margin: float) -> str:
    if score >= 45.0 and margin >= 3.0:
        return "high"
    if score >= 30.0 and margin >= 1.0:
        return "medium"
    return "low"
