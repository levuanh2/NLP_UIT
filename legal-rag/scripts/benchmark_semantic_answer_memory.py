"""Benchmark semantic and lexical-semantic LegalQA answer memory."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from app.baseline.data import load_question_records
from app.baseline.retriever import AnswerMemoryRetriever
from app.evaluation.scorer_compatible import score_answer_pairs


def _row_minmax(scores: np.ndarray) -> np.ndarray:
    minimum = scores.min(axis=1, keepdims=True)
    scale = scores.max(axis=1, keepdims=True) - minimum
    return (scores - minimum) / np.maximum(scale, 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--holdout-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    records = load_question_records(args.train, require_answers=True)
    with np.load(args.cache) as cache:
        ids = cache["ids"].tolist()
        passages = cache["question_embeddings"]
    expected_ids = [record.question_id for record in records]
    if ids != expected_ids:
        raise ValueError("Embedding cache IDs do not match the training dataset.")

    if args.query_cache.is_file():
        queries = np.load(args.query_cache)
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            str(args.model),
            device="cpu",
            local_files_only=True,
            trust_remote_code=True,
        )
        queries = model.encode(
            [f"query: {record.question}" for record in records],
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        args.query_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.query_cache, queries)
    if queries.shape != passages.shape:
        raise ValueError("Query and passage embedding cache shapes differ.")

    fit_indices, holdout_indices = train_test_split(
        np.arange(len(records)),
        test_size=args.holdout_size,
        random_state=args.seed,
    )
    fit_records = [records[int(index)] for index in fit_indices]
    holdout = [records[int(index)] for index in holdout_indices]
    references = [record.answer or "" for record in holdout]

    lexical = AnswerMemoryRetriever()
    lexical.fit(fit_records)
    lexical_scores = lexical.score_questions(
        [record.question for record in holdout]
    ).astype(np.float32)
    semantic_scores = queries[holdout_indices] @ passages[fit_indices].T
    lexical_scaled = _row_minmax(lexical_scores)
    semantic_scaled = _row_minmax(semantic_scores)

    for semantic_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        scores = (
            (1 - semantic_weight) * lexical_scaled
            + semantic_weight * semantic_scaled
        )
        best = np.argmax(scores, axis=1)
        predictions = [fit_records[int(index)].answer or "" for index in best]
        report = score_answer_pairs(references, predictions)
        print(
            f"semantic_weight={semantic_weight:.2f} "
            f"meteor={report.meteor:.6f} rouge_l={report.rouge_l:.6f}"
        )


if __name__ == "__main__":
    main()
