"""Submission formatting and writing tests."""

import json
from pathlib import Path

from app.domain.generation import GeneratedAnswer
from app.submission.formatter import SubmissionFormatter
from app.submission.writer import SubmissionWriter


def test_submission_formatter_uses_question_id_as_key() -> None:
    answer = GeneratedAnswer(
        question_id="147194", answer="Câu trả lời.", grounded=False
    )

    submission = SubmissionFormatter().format([answer])

    assert set(submission) == {"147194"}


def test_submission_formatter_contains_only_answer() -> None:
    answer = GeneratedAnswer(
        question_id="147194",
        answer="Theo Điều 1...",
        evidence_ids=["e-1"],
        grounded=True,
    )

    payload = SubmissionFormatter().format([answer])["147194"].model_dump()

    assert payload == {"answer": "Theo Điều 1..."}


def test_formatter_strips_scaffolding_the_reference_answers_never_contain() -> None:
    answer = GeneratedAnswer(
        question_id="1",
        answer=(
            "Theo [1], người lao động có quyền.\n\n"
            "1. **Điều kiện**: Theo [1] và [2], phải đủ 18 tuổi.\n"
            "- Thời hạn là 30 ngày [3]."
        ),
        grounded=False,
    )

    text = SubmissionFormatter().format([answer])["1"].answer

    assert "[" not in text and "*" not in text
    assert "\n" not in text
    assert text == (
        "Theo, người lao động có quyền. "
        "Điều kiện: Theo và, phải đủ 18 tuổi. "
        "Thời hạn là 30 ngày."
    )


def test_formatter_strips_prompt_layout_leaks_and_corpus_filenames() -> None:
    answer = GeneratedAnswer(
        question_id="1",
        answer=(
            "Dựa trên ngữ cảnh cung cấp, mẫu thông báo nằm ở các văn bản sau:\n"
            "1. **Văn bản 1 (Ngữ cảnh 1)**:\n"
            "   - **Điều 50** của **Luật Doanh nghiệp 2020** "
            "(Nghi-dinh-01-2021-ND-CP-dang-ky-doanh-nghiep-283247).\n"
            "Theo quy định tại Điều 43 (Ngữ cảnh 2), hồ sơ gồm "
            "(Nghi-dinh-78-2015-ND-CP-dang-ky-do"
        ),
        grounded=False,
    )

    text = SubmissionFormatter().format([answer])["1"].answer

    assert text == (
        "Mẫu thông báo nằm ở các văn bản sau: "
        "Điều 50 của Luật Doanh nghiệp 2020. "
        "Theo quy định tại Điều 43, hồ sơ gồm"
    )


def test_formatter_trims_filename_tails_and_self_grading_closers() -> None:
    answer = GeneratedAnswer(
        question_id="1",
        answer=(
            "Trong ngữ cảnh cung cấp, theo Quyết định "
            "02-2023-QD-KTNN-lap-tham-dinh-ban-hanh-ke-hoach-kiem-toan-nam-554186, "
            "kiểm toán viên phải lập kế hoạch. "
            "Vì vậy, câu trả lời chính xác và đầy đủ dựa trên thông tin được cung cấp."
        ),
        grounded=False,
    )

    text = SubmissionFormatter().format([answer])["1"].answer

    assert text == (
        "Theo Quyết định 02-2023-QD-KTNN, kiểm toán viên phải lập kế hoạch."
    )


def test_formatter_keeps_document_codes_that_have_no_filename_tail() -> None:
    answer = GeneratedAnswer(
        question_id="1",
        answer="Theo Nghị định 88-2022-ND-CP và Nghị định 74/2015/NĐ-CP, mức phạt là.",
        grounded=False,
    )

    text = SubmissionFormatter().format([answer])["1"].answer

    assert "88-2022-ND-CP" in text and "74/2015/NĐ-CP" in text


def test_submission_writer_uses_utf8(tmp_path: Path) -> None:
    submission = SubmissionFormatter().format(
        [GeneratedAnswer(question_id="1", answer="Quyền và nghĩa vụ", grounded=True)]
    )
    output = tmp_path / "submission.json"

    SubmissionWriter().write(submission, output)

    assert output.read_bytes().decode("utf-8")


def test_submission_writer_preserves_vietnamese_characters(tmp_path: Path) -> None:
    expected = "Người lao động có quyền."
    submission = SubmissionFormatter().format(
        [GeneratedAnswer(question_id="147195", answer=expected, grounded=True)]
    )
    output = tmp_path / "submission.json"

    SubmissionWriter().write(submission, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["147195"]["answer"] == expected
    assert "Người lao động" in output.read_text(encoding="utf-8")


def test_submission_writer(tmp_path: Path) -> None:
    output = tmp_path / "submission.json"
    answers = {
        "147194": GeneratedAnswer(
            question_id="147194",
            answer="Câu trả lời có căn cứ [1].",
            grounded=True,
        )
    }

    SubmissionWriter().write(answers, output, expected_question_ids={"147194"})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "147194": {"answer": "Câu trả lời có căn cứ [1]."}
    }
