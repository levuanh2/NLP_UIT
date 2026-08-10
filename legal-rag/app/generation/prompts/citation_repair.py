"""Prompt construction for a bounded, model-generated citation repair."""

from app.domain.retrieval import LegalEvidence


class CitationRepairPromptBuilder:
    """Render the immutable evidence-to-citation mapping for one repair call."""

    def build(
        self,
        *,
        question: str,
        answer: str,
        evidences: list[LegalEvidence],
    ) -> str:
        grounds = "\n\n".join(
            self._render_evidence(index, evidence)
            for index, evidence in enumerate(evidences, start=1)
        )
        return f"""SYSTEM:
Bạn sửa citation cho câu trả lời pháp luật.

QUY TẮC:
- Giữ nguyên ý được căn cứ hỗ trợ; không thêm thông tin hoặc kiến thức ngoài.
- Mỗi nhận định phải có citation [n] thực sự hỗ trợ và chỉ dùng [n] bên dưới.
- Không tạo Điều, Khoản, Điểm, citation hoặc nội dung pháp luật mới.
- Nếu nội dung không được hỗ trợ đầy đủ, hoặc câu hỏi xin mẫu nhưng căn cứ
  không chứa mẫu hoàn chỉnh, chỉ trả lời:
  "Không đủ căn cứ trong tài liệu được cung cấp."
- Chỉ trả về câu trả lời cuối; không chép quy tắc hay danh sách căn cứ.

### Câu hỏi
{question.strip()}

### Câu trả lời hiện tại
{answer.strip()}

### Các căn cứ được phép trích dẫn
{grounds}

### Câu trả lời đã sửa
"""

    @staticmethod
    def _render_evidence(index: int, evidence: LegalEvidence) -> str:
        fields = (
            ("Tên văn bản", evidence.document_name),
            ("Chương", evidence.chapter),
            ("Điều", evidence.article),
            ("Khoản", evidence.clause),
            ("Điểm", evidence.point),
        )
        metadata = "\n".join(f"{label}: {value}" for label, value in fields if value)
        return f"[{index}]\n{metadata}\nNội dung:\n{evidence.text}\n[/{index}]"
