"""Submission orchestration integration tests."""

import json
from pathlib import Path

import pytest

from app.domain.generation import GeneratedAnswer
from app.domain.queries import LegalQuery
from app.generation.prompts.system import ABSTENTION_ANSWER
from app.generation.validation.citation_validator import CitationValidator
from app.services.submission_service import SubmissionService
from app.submission.formatter import SubmissionFormatter
from app.submission.validator import SubmissionValidator
from app.submission.writer import SubmissionWriter


class FakeRAGService:
    def __init__(self, crash_on: str | None = None) -> None:
        self.crash_on = crash_on
        self.answered: list[str] = []

    def answer(self, query: LegalQuery) -> GeneratedAnswer:
        if query.question_id == self.crash_on:
            raise RuntimeError("simulated kill")
        self.answered.append(query.question_id)
        return GeneratedAnswer(
            question_id=query.question_id,
            answer=f"Trả lời có căn cứ cho {query.question_id} [1].",
            grounded=True,
        )


def test_submission_pipeline_writes_one_answer_per_question(tmp_path: Path) -> None:
    questions = [
        LegalQuery(question_id="1", question="Câu hỏi một"),
        LegalQuery(question_id="2", question="Câu hỏi hai"),
    ]
    output = tmp_path / "submission.json"
    service = SubmissionService(
        FakeRAGService(),  # type: ignore[arg-type]
        SubmissionFormatter(),
        SubmissionValidator(),
        SubmissionWriter(),
    )

    result = service.create(questions, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.valid is True
    assert set(payload) == {"1", "2"}
    assert all(set(value) == {"answer"} for value in payload.values())


def test_submission_resumes_from_checkpoint_without_reanswering(
    tmp_path: Path,
) -> None:
    questions = [
        LegalQuery(question_id=str(index), question=f"Câu hỏi {index}")
        for index in range(1, 4)
    ]
    output = tmp_path / "submission.json"
    checkpoint = tmp_path / "submission.partial.jsonl"

    def build(rag: FakeRAGService) -> SubmissionService:
        return SubmissionService(
            rag,  # type: ignore[arg-type]
            SubmissionFormatter(),
            SubmissionValidator(),
            SubmissionWriter(),
            fail_fast=True,
            checkpoint_path=checkpoint,
        )

    crashing = FakeRAGService(crash_on="3")
    with pytest.raises(RuntimeError):
        build(crashing).create(questions, output)
    assert crashing.answered == ["1", "2"]
    assert not output.exists()

    resuming = FakeRAGService()
    result = build(resuming).create(questions, output)

    assert resuming.answered == ["3"], "checkpointed answers must not be recomputed"
    assert result.valid is True
    assert set(json.loads(output.read_text(encoding="utf-8"))) == {"1", "2", "3"}


def test_failed_question_abstains_so_every_id_is_present(tmp_path: Path) -> None:
    questions = [
        LegalQuery(question_id="1", question="Câu hỏi một"),
        LegalQuery(question_id="2", question="Câu hỏi hai"),
    ]
    output = tmp_path / "submission.json"
    checkpoint = tmp_path / "submission.partial.jsonl"
    failures = tmp_path / "submission.failures.jsonl"
    service = SubmissionService(
        FakeRAGService(crash_on="2"),  # type: ignore[arg-type]
        SubmissionFormatter(),
        SubmissionValidator(),
        SubmissionWriter(),
        checkpoint_path=checkpoint,
        fallback_answer=ABSTENTION_ANSWER,
        failure_path=failures,
    )

    result = service.create(questions, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.valid is True
    assert set(payload) == {"1", "2"}
    assert payload["2"]["answer"] == ABSTENTION_ANSWER
    assert len(service.failures) == 1
    # An abstention is not checkpointed, so a rerun retries that question.
    checkpointed = [
        json.loads(line)["question_id"]
        for line in checkpoint.read_text(encoding="utf-8").splitlines()
    ]
    assert checkpointed == ["1"]
    # Progress is pass + fail, so the abstention must be visible somewhere.
    assert [json.loads(line)["question_id"] for line in failures.read_text(
        encoding="utf-8"
    ).splitlines()] == ["2"]


def test_abstention_answer_is_accepted_by_the_citation_validator() -> None:
    assert CitationValidator.is_safe_abstention(ABSTENTION_ANSWER)
