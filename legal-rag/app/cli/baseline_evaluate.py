"""Holdout evaluation for the scorer-tuned baseline."""

from pathlib import Path

import typer

from app.baseline.data import load_question_records
from app.baseline.retriever import AnswerMemoryRetriever
from app.evaluation.scorer_compatible import score_answer_pairs


def evaluate_baseline(
    train: Path = typer.Option(..., "--train", "-t"),
    holdout_size: int = typer.Option(300, min=10),
    seed: int = typer.Option(2026),
    word_weight: float = typer.Option(0.75, min=0.0, max=1.0),
    question_weight: float = typer.Option(0.50, min=0.0, max=1.0),
) -> None:
    """Evaluate without leaking holdout examples into answer memory."""
    try:
        import numpy as np
        from sklearn.model_selection import train_test_split
    except ImportError as exc:  # pragma: no cover - dependency error
        raise RuntimeError("numpy and scikit-learn are required.") from exc

    records = load_question_records(train, require_answers=True)
    if holdout_size >= len(records):
        raise typer.BadParameter("holdout-size must be smaller than the dataset.")
    train_indices, holdout_indices = train_test_split(
        np.arange(len(records)),
        test_size=holdout_size,
        random_state=seed,
    )
    fit_records = [records[int(index)] for index in train_indices]
    holdout = [records[int(index)] for index in holdout_indices]
    retriever = AnswerMemoryRetriever(
        word_weight=word_weight,
        question_weight=question_weight,
    )
    retriever.fit(fit_records)
    matches = retriever.retrieve([record.question for record in holdout])
    report = score_answer_pairs(
        [record.answer or "" for record in holdout],
        [match.answer for match in matches],
    )
    typer.echo(
        f"samples={report.sample_count} meteor={report.meteor:.6f} "
        f"rouge_l={report.rouge_l:.6f}"
    )
