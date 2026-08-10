"""Safe insufficient-evidence policy."""

from app.domain.retrieval import LegalContext, RetrievalResult


class AbstentionValidator:
    def __init__(
        self,
        abstention_message: str = (
            "Không tìm thấy đủ căn cứ trong tài liệu được cung cấp để trả lời "
            "câu hỏi này."
        ),
    ) -> None:
        self.abstention_message = abstention_message

    def should_abstain(self, context: LegalContext | RetrievalResult) -> bool:
        return not any(evidence.text.strip() for evidence in context.evidences)
