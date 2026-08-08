"""Bounded grounded Vietnamese legal answer prompt."""

from app.domain.retrieval import LegalContext
from app.generation.prompts.system import LEGAL_SYSTEM_PROMPT


class LegalPromptBuilder:
    def build(self, question: str, context: LegalContext) -> str:
        """Build system instruction, legal evidence, and user question."""
        return (
            f"{LEGAL_SYSTEM_PROMPT}\n\n"
            f"LEGAL EVIDENCE\n{context.formatted_context}\n\n"
            f"USER QUESTION\n{question.strip()}\n\n"
            "Hãy trả lời bằng văn xuôi tiếng Việt, nêu căn cứ pháp lý chính xác. "
            "Gắn nhãn [E1], [E2] tương ứng sau nội dung được sử dụng. "
            "Không thêm thông tin không có trong LEGAL EVIDENCE."
        )
