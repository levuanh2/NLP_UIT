"""Grounding evaluation sampling and gold-dependent metric tests."""

import pytest

from app.domain.queries import LegalQuery
from app.evaluation.grounding_evaluation import calculate_metrics, deterministic_sample


def _questions(count: int) -> list[LegalQuery]:
    return [
        LegalQuery(question_id=str(index), question=f"Question {index}")
        for index in range(count)
    ]


def test_grounding_sample_is_deterministic_and_forces_regressions() -> None:
    questions = _questions(100)
    first = deterministic_sample(
        questions, sample_size=50, seed=20260810, forced_ids=("1", "2")
    )
    second = deterministic_sample(
        questions, sample_size=50, seed=20260810, forced_ids=("1", "2")
    )

    assert [item.question_id for item in first] == [
        item.question_id for item in second
    ]
    assert [item.question_id for item in first[:2]] == ["1", "2"]
    assert len({item.question_id for item in first}) == 50


def test_grounding_sample_fails_when_dataset_is_too_small() -> None:
    with pytest.raises(ValueError, match="50 required"):
        deterministic_sample(
            _questions(49), sample_size=50, seed=20260810, forced_ids=("1",)
        )


def test_grounding_metrics_ignore_unannotated_cases() -> None:
    metrics = calculate_metrics(
        [
            {"question_id": "1", "validator_grounded": True, "error": None},
            {"question_id": "2", "validator_grounded": False, "error": None},
        ],
        [
            {"question_id": "1", "gold_grounded": None},
            {"question_id": "2", "gold_grounded": True},
        ],
    )

    assert metrics["certain_cases"] == 1
    assert metrics["uncertain_cases"] == 1
    assert metrics["confusion_matrix"] == {"tp": 0, "tn": 0, "fp": 0, "fn": 1}
    assert metrics["precision"] is None
    assert metrics["false_negative_rate"] == 1.0


def test_grounding_metrics_include_manual_confusion_and_breakdown() -> None:
    metrics = calculate_metrics(
        [
            {
                "question_id": "tp",
                "validator_grounded": True,
                "citation_valid": True,
                "abstained": False,
                "error": None,
            },
            {
                "question_id": "fp",
                "validator_grounded": True,
                "citation_valid": True,
                "abstained": False,
                "error": None,
            },
            {
                "question_id": "fn",
                "validator_grounded": False,
                "citation_valid": True,
                "abstained": False,
                "error": None,
            },
            {
                "question_id": "tn",
                "validator_grounded": False,
                "citation_valid": False,
                "abstained": False,
                "error": None,
            },
        ],
        [
            {
                "question_id": "tp",
                "gold_grounded": True,
                "review_status": "manual_reviewed",
                "failure_types": [],
            },
            {
                "question_id": "fp",
                "gold_grounded": False,
                "review_status": "manual_reviewed",
                "failure_types": ["validator_false_positive"],
            },
            {
                "question_id": "fn",
                "gold_grounded": True,
                "review_status": "manual_reviewed",
                "failure_types": ["validator_false_negative"],
            },
            {
                "question_id": "tn",
                "gold_grounded": False,
                "review_status": "manual_reviewed",
                "failure_types": ["missing_citation"],
            },
        ],
    )

    assert metrics["confusion_matrix"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    assert metrics["manual_reviewed"] == 4
    assert metrics["uncertain"] == 0
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["false_negative_rate"] == 0.5
    assert metrics["accuracy"] == 0.5
    assert metrics["failure_breakdown"]["missing_citation"] == 1
    assert metrics["failure_breakdown"]["validator_false_positive"] == 1
    assert metrics["failure_breakdown"]["validator_false_negative"] == 1
