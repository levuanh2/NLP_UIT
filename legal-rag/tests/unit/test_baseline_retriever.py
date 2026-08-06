"""Answer-memory retrieval tests."""

from app.baseline.data import QuestionRecord
from app.baseline.retriever import AnswerMemoryRetriever


def test_exact_question_retrieves_its_expert_answer() -> None:
    records = [
        QuestionRecord("1", "Người lao động có quyền gì?", "Đáp án lao động."),
        QuestionRecord("2", "Mức phạt giao thông là bao nhiêu?", "Đáp án phạt."),
    ]
    retriever = AnswerMemoryRetriever()
    retriever.fit(records)

    match = retriever.retrieve(["  NGƯỜI LAO ĐỘNG có quyền gì?  "])[0]

    assert match.source_question_id == "1"
    assert match.answer == "Đáp án lao động."
    assert match.score == 1.0


def test_related_question_prefers_relevant_training_question() -> None:
    records = [
        QuestionRecord(
            "1",
            "Điều kiện hưởng bảo hiểm thất nghiệp là gì?",
            "Đáp án bảo hiểm thất nghiệp.",
        ),
        QuestionRecord(
            "2",
            "Mức xử phạt vi phạm giao thông là bao nhiêu?",
            "Đáp án giao thông.",
        ),
    ]
    retriever = AnswerMemoryRetriever()
    retriever.fit(records)

    match = retriever.retrieve(["Điều kiện nhận trợ cấp thất nghiệp?"])[0]

    assert match.source_question_id == "1"
