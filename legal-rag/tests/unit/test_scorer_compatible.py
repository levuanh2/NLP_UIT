"""Organizer-compatible local metric tests."""

import pytest

from app.evaluation.scorer_compatible import (
    organizer_rouge_tokens,
    rouge_l_fmeasure,
    score_answer_pairs,
)


def test_identical_answers_receive_expected_near_perfect_scores() -> None:
    answer = "Theo Điều 37, người lao động có quyền."

    report = score_answer_pairs([answer], [answer])

    assert report.meteor == pytest.approx(0.9990234375)
    assert report.rouge_l == 1.0


def test_rouge_tokenizer_replicates_ascii_fragmentation() -> None:
    tokens = organizer_rouge_tokens("Người lao động")

    assert tokens == ["ng", "i", "lao", "ng"]


def test_rouge_l_is_zero_without_overlap() -> None:
    assert rouge_l_fmeasure("abc", "xyz") == 0.0
