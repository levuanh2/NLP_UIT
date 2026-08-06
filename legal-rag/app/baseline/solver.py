"""End-to-end answer-memory LegalQA solver."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.baseline.data import load_question_records
from app.baseline.retriever import AnswerMemoryRetriever
from app.baseline.semantic import SemanticHybridAnswerMemoryRetriever
from app.corpus.fts import LegalCorpusIndex, extract_answer_span


def solve_with_answer_memory(
    questions_path: Path,
    train_path: Path,
    word_weight: float = 0.75,
    question_weight: float = 0.50,
    batch_size: int = 128,
    corpus_index_path: Path | None = None,
    memory_threshold: float = 0.0,
    semantic_model_path: Path | None = None,
    semantic_cache_path: Path | None = None,
    semantic_weight: float = 0.75,
    semantic_device: str = "cpu",
) -> list[dict[str, Any]]:
    """Answer a competition dataset using nearest expert training answers."""
    if corpus_index_path is not None and not corpus_index_path.is_file():
        raise ValueError(f"Corpus index does not exist: {corpus_index_path}")
    training = load_question_records(train_path, require_answers=True)
    questions = load_question_records(questions_path, require_answers=False)
    if (semantic_model_path is None) != (semantic_cache_path is None):
        raise ValueError("Semantic model and cache must be provided together.")
    if semantic_model_path is not None and semantic_cache_path is not None:
        retriever = SemanticHybridAnswerMemoryRetriever(
            model_path=semantic_model_path,
            cache_path=semantic_cache_path,
            semantic_weight=semantic_weight,
            word_weight=word_weight,
            question_weight=question_weight,
            batch_size=batch_size,
            device=semantic_device,
        )
        method = "hybrid_tfidf_semantic_expert_memory"
    else:
        retriever = AnswerMemoryRetriever(
            word_weight=word_weight,
            question_weight=question_weight,
            batch_size=batch_size,
        )
        method = "hybrid_tfidf_expert_memory"
    retriever.fit(training)
    matches = retriever.retrieve([record.question for record in questions])
    corpus_index = (
        LegalCorpusIndex(corpus_index_path) if corpus_index_path is not None else None
    )

    results: list[dict[str, Any]] = []
    for question, match in zip(questions, matches, strict=True):
        evidences = []
        extractive_answer = ""
        if corpus_index is not None and match.score < memory_threshold:
            evidences = corpus_index.search(question.question, limit=5)
            extractive_answer = extract_answer_span(question.question, evidences)
        use_corpus = bool(extractive_answer)
        answer = extractive_answer if use_corpus else match.answer
        result = {
            "question_id": question.question_id,
            "question": question.question,
            "answer": answer,
            "retrieval": asdict(match),
            "corpus_evidence": [asdict(evidence) for evidence in evidences],
            "method": (
                "corpus_fts_extractive"
                if use_corpus
                else method
            ),
            "confidence": _confidence_label(match.score, match.score_margin),
        }
        results.append(result)
    return results


def _confidence_label(score: float, margin: float) -> str:
    """Return a transparent retrieval confidence bucket."""
    if score >= 0.45 and margin >= 0.03:
        return "high"
    if score >= 0.30 and margin >= 0.01:
        return "medium"
    return "low"
