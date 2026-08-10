"""Configured model identity and strict aggregate parameter budget tests."""

import pytest

from app.services.runtime_factory import (
    APPROVED_MODELS,
    PARAMETER_BUDGET_LIMIT,
    verify_parameter_budget,
)


def test_approved_generation_model_is_vi_qwen_3b() -> None:
    assert APPROVED_MODELS["llm"] == "AITeamVN/Vi-Qwen2-3B-RAG"


def test_parameter_budget_accepts_total_strictly_below_4b() -> None:
    verify_parameter_budget(
        {"embedding": 100, "reranker": 200, "llm": PARAMETER_BUDGET_LIMIT - 301}
    )


def test_parameter_budget_rejects_total_equal_to_4b() -> None:
    with pytest.raises(RuntimeError, match="strict <4B"):
        verify_parameter_budget(
            {"embedding": 100, "reranker": 200, "llm": PARAMETER_BUDGET_LIMIT - 300}
        )


def test_parameter_budget_requires_every_model_count() -> None:
    with pytest.raises(RuntimeError, match="missing counts: reranker"):
        verify_parameter_budget({"embedding": 100, "llm": 200})
