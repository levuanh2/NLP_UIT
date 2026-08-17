"""Grounded prompt, citation, grounding, and generation tests."""

import pytest

from app.domain.generation import GenerationRequest
from app.domain.queries import QueryMetadata
from app.domain.retrieval import LegalEvidence, RetrievalResult
from app.generation.citation_repair import CitationRepair
from app.generation.citation_validator import CitationValidator
from app.generation.grounding import GroundingValidator
from app.generation.llm.base import BaseLLMGenerator
from app.generation.llm.qwen_generator import QwenGenerator
from app.generation.pipeline import GenerationPipeline
from app.generation.prompt import LegalPromptBuilder
from app.generation.prompts.citation_repair import CitationRepairPromptBuilder
from app.generation.validation.abstention_validator import AbstentionValidator


class MockLocalLLM(BaseLLMGenerator):
    model_name = "mock/local-llm"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    def load(self) -> None:
        pass

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self.calls += 1
        assert "CONTEXT:" in prompt
        return self.answer

    def unload(self) -> None:
        pass


class ScriptedLocalLLM(BaseLLMGenerator):
    model_name = "mock/scripted-local-llm"

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = responses
        self.calls = 0
        self.prompts: list[str] = []

    def load(self) -> None:
        pass

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self.prompts.append(prompt)
        response = self.responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response

    def unload(self) -> None:
        pass


def evidence() -> LegalEvidence:
    return LegalEvidence(
        evidence_id="parent-1",
        document_id=740,
        document_name="Luật Doanh nghiệp",
        source_link="https://example.test/740",
        chapter="Chương I",
        section=None,
        article="Điều 37",
        clause="Khoản 2",
        point="Điểm a",
        child_id="doc:740:article:37:clause:2",
        parent_id="parent-1",
        position=1,
        rank=1,
        text="Điều 37 Khoản 2 quy định doanh nghiệp đáp ứng điều kiện đăng ký.",
    )


def retrieval(*, with_evidence: bool = True) -> RetrievalResult:
    return RetrievalResult(
        query="Điều kiện đăng ký là gì?",
        query_metadata=QueryMetadata(raw_query="Điều kiện đăng ký là gì?"),
        candidates=[],
        evidences=[evidence()] if with_evidence else [],
        active_index_version="fixture-v9",
        dense_count=1,
        bm25_count=1,
        fused_count=1,
        reranked_count=1,
    )


def pipeline(llm: MockLocalLLM) -> GenerationPipeline:
    citations = CitationValidator()
    return GenerationPipeline(
        llm,
        LegalPromptBuilder(max_context_tokens=1000, reserved_generation_tokens=64),
        citations,
        GroundingValidator(citations),
        AbstentionValidator(),
        max_new_tokens=64,
    )


def pipeline_with_repair(
    llm: ScriptedLocalLLM, *, enabled: bool = True, max_retries: int = 1
) -> GenerationPipeline:
    citations = CitationValidator()
    return GenerationPipeline(
        llm,
        LegalPromptBuilder(max_context_tokens=1000, reserved_generation_tokens=64),
        citations,
        GroundingValidator(citations),
        AbstentionValidator(),
        CitationRepair(
            llm,
            CitationRepairPromptBuilder(),
            enabled=enabled,
            max_retries=max_retries,
            max_new_tokens=64,
            max_context_tokens=1000,
        ),
        max_new_tokens=64,
    )


def test_prompt_builder() -> None:
    prompt = LegalPromptBuilder(
        max_context_tokens=1000, reserved_generation_tokens=64
    ).build("Điều kiện là gì?", retrieval())
    assert "Dựa vào Ngữ cảnh" in prompt
    assert "[1]\nDOCUMENT:" in prompt
    assert "Document ID: 740" in prompt
    assert "Điều: Điều 37" in prompt
    assert "Child ID: doc:740:article:37:clause:2" in prompt
    assert "[/1]" in prompt
    assert "### Ngữ cảnh:" in prompt
    assert "### Câu hỏi:" in prompt
    assert "Cấm dùng dấu ngoặc vuông kiểu [1], [2] để trích dẫn." in prompt
    assert "dense_score" not in prompt and "rerank" not in prompt


def test_qwen_conversation_preserves_grounded_prompt_roles() -> None:
    prompt = LegalPromptBuilder(
        max_context_tokens=1000, reserved_generation_tokens=64
    ).build("Điều kiện là gì?", retrieval())
    conversation = QwenGenerator._conversation(prompt)
    assert [message["role"] for message in conversation] == ["system", "user"]
    assert "Dựa vào Ngữ cảnh" in conversation[0]["content"]
    assert conversation[1]["content"].startswith(
        "CONTEXT:\n### Ngữ cảnh:\n[1]\nDOCUMENT:"
    )
    assert "QUESTION:\n### Câu hỏi:\nĐiều kiện là gì?" in conversation[1]["content"]


def test_empty_context() -> None:
    llm = MockLocalLLM("must not be used")
    result = pipeline(llm).generate(
        GenerationRequest(
            question_id="q-empty",
            question="Câu hỏi",
            retrieval_result=retrieval(with_evidence=False),
        )
    )
    assert llm.calls == 0
    assert result.abstained is True
    assert result.grounded is True
    assert "Không tìm thấy đủ căn cứ" in result.answer


def test_citation_validator_valid() -> None:
    result = CitationValidator().validate("Nội dung được quy định [1].", retrieval())
    assert result.valid is True
    assert result.citations[0].document_id == 740


def test_citation_validator_invalid() -> None:
    result = CitationValidator().validate("Nội dung không có thật [99].", retrieval())
    assert result.valid is False
    assert "does not exist" in result.errors[0]


def test_grounding_validator() -> None:
    validator = GroundingValidator()
    valid = validator.validate(
        "Theo Điều 37, doanh nghiệp phải đăng ký [1].", retrieval()
    )
    invalid = validator.validate(
        "Theo Điều 100, doanh nghiệp phải đăng ký [1].", retrieval()
    )
    assert valid.grounded is True
    assert invalid.grounded is False
    assert any("Unsupported legal metadata" in error for error in invalid.errors)


def _retrieval_with_metadata(
    *, article: str = "Điều 5", clause: str = "Khoản 2", point: str = "Điểm a"
) -> RetrievalResult:
    item = evidence().model_copy(
        update={
            "article": article,
            "clause": clause,
            "point": point,
            "text": f"{article} {clause} {point} quy định nội dung kiểm tra.",
        }
    )
    return retrieval().model_copy(update={"evidences": [item]})


def test_article_5_does_not_match_article_50() -> None:
    result = GroundingValidator().validate(
        "Theo Điều 5, nội dung được quy định như sau [1].",
        _retrieval_with_metadata(article="Điều 50"),
    )
    assert result.grounded is False
    assert "Unsupported legal metadata in answer: Điều 5" in result.errors


def test_article_5_matches_article_5() -> None:
    result = GroundingValidator().validate(
        "Theo Điều 5, nội dung được quy định như sau [1].",
        _retrieval_with_metadata(article="Điều 05"),
    )
    assert result.grounded is True


def test_clause_2_does_not_match_clause_20() -> None:
    result = GroundingValidator().validate(
        "Theo Khoản 2, nội dung được quy định như sau [1].",
        _retrieval_with_metadata(clause="Khoản 20"),
    )
    assert result.grounded is False
    assert "Unsupported legal metadata in answer: Khoản 2" in result.errors


def test_point_a_does_not_match_point_aa() -> None:
    result = GroundingValidator().validate(
        "Theo Điểm a, nội dung được quy định như sau [1].",
        _retrieval_with_metadata(point="Điểm aa"),
    )
    assert result.grounded is False
    assert "Unsupported legal metadata in answer: Điểm a" in result.errors


def _retrieval_with_text(text: str) -> RetrievalResult:
    item = evidence().model_copy(
        update={
            "article": None,
            "clause": None,
            "point": None,
            "text": text,
        }
    )
    return retrieval().model_copy(update={"evidences": [item]})


def test_claim_grounding_accepts_direct_support() -> None:
    result = GroundingValidator().validate(
        "Theo [1], doanh nghiệp phải thông báo thay đổi.",
        _retrieval_with_text("Doanh nghiệp phải thông báo thay đổi."),
    )

    assert result.grounded is True


def test_claim_grounding_accepts_modality_paraphrase() -> None:
    result = GroundingValidator().validate(
        "Theo [1], cơ quan phải hướng dẫn người dân.",
        _retrieval_with_text("Cơ quan có trách nhiệm hướng dẫn người dân."),
    )

    assert result.grounded is True


def test_claim_grounding_rejects_unsupported_deadline() -> None:
    result = GroundingValidator().validate(
        "Theo [1], doanh nghiệp phải thông báo thay đổi trong 30 ngày.",
        _retrieval_with_text("Doanh nghiệp phải thông báo thay đổi."),
    )

    assert result.grounded is False
    assert any("does not support this detail" in error for error in result.errors)


def test_claim_grounding_rejects_unsupported_condition() -> None:
    result = GroundingValidator().validate(
        "Theo [1], doanh nghiệp phải gửi hồ sơ trước khi quyết định có hiệu lực.",
        _retrieval_with_text("Doanh nghiệp phải gửi hồ sơ."),
    )

    assert result.grounded is False


def test_claim_grounding_rejects_unsupported_actor_expansion() -> None:
    result = GroundingValidator().validate(
        "Theo [1], cả người đại diện cũ và mới phải ký hồ sơ.",
        _retrieval_with_text("Người đại diện phải ký hồ sơ."),
    )

    assert result.grounded is False


def test_claim_grounding_accepts_multiple_supported_actions() -> None:
    result = GroundingValidator().validate(
        "Theo [1], doanh nghiệp phải thông báo và gửi hồ sơ.",
        _retrieval_with_text("Doanh nghiệp phải thông báo và gửi hồ sơ."),
    )

    assert result.grounded is True


def test_claim_grounding_accepts_safe_all_obligations_emphasis() -> None:
    result = GroundingValidator().validate(
        "Theo [1], hội viên phải hoàn thành tất cả các nghĩa vụ còn tồn tại "
        "trước khi quyết định có hiệu lực.",
        _retrieval_with_text(
            "Hội viên phải hoàn thành các nghĩa vụ còn tồn tại trước khi "
            "quyết định có hiệu lực."
        ),
    )

    assert result.grounded is True


def test_claim_grounding_rejects_all_emphasis_with_new_detail() -> None:
    result = GroundingValidator().validate(
        "Theo [1], hội viên phải hoàn thành tất cả các nghĩa vụ tài chính "
        "còn tồn tại trước khi quyết định có hiệu lực.",
        _retrieval_with_text(
            "Hội viên phải hoàn thành các nghĩa vụ còn tồn tại trước khi "
            "quyết định có hiệu lực."
        ),
    )

    assert result.grounded is False


def test_claim_grounding_rejects_all_emphasis_over_some_obligations() -> None:
    result = GroundingValidator().validate(
        "Theo [1], hội viên phải hoàn thành tất cả các nghĩa vụ.",
        _retrieval_with_text("Hội viên phải hoàn thành một số nghĩa vụ."),
    )

    assert result.grounded is False


def test_claim_grounding_rejects_permissive_to_mandatory_modality_change() -> None:
    result = GroundingValidator().validate(
        "Theo [1], hội viên phải thực hiện thủ tục.",
        _retrieval_with_text("Hội viên có thể thực hiện thủ tục."),
    )

    assert result.grounded is False


def test_claim_grounding_rejects_all_quantifier_actor_expansion() -> None:
    result = GroundingValidator().validate(
        "Theo [1], tất cả người lao động phải ký hồ sơ.",
        _retrieval_with_text("Người đại diện phải ký hồ sơ."),
    )

    assert result.grounded is False


def test_claim_grounding_accepts_responsibility_paraphrase() -> None:
    result = GroundingValidator().validate(
        "Theo [1], hội viên có trách nhiệm hoàn thành các nghĩa vụ.",
        _retrieval_with_text("Hội viên phải hoàn thành các nghĩa vụ."),
    )

    assert result.grounded is True


def test_claim_grounding_does_not_leak_citation_to_next_paragraph() -> None:
    result = GroundingValidator().validate(
        "Theo [1], doanh nghiệp phải thông báo thay đổi.\n\n"
        "Ngoài ra, quyết định chỉ có hiệu lực sau khi hoàn thành nghĩa vụ.",
        _retrieval_with_text("Doanh nghiệp phải thông báo thay đổi."),
    )

    assert result.grounded is False
    assert any("without citation scope" in error for error in result.errors)


def test_31221_unsupported_elaboration_is_not_grounded() -> None:
    source = _retrieval_with_text(
        "1. Hội viên tổ chức muốn ra khỏi Hội gửi đơn đề nghị đến Ban Chấp hành "
        "Hội. 2. Ban Chấp hành xem xét nghĩa vụ của hội viên và quyết định việc "
        "chấm dứt tư cách hội viên."
    )
    answer = (
        "Theo [1], thủ tục hội viên tổ chức ra khỏi Hội được quy định như sau:\n\n"
        "1. Gửi đơn đề nghị: Hội viên tổ chức muốn ra khỏi Hội phải gửi đơn đề "
        "nghị đến Ban Chấp hành Hội.\n\n"
        "2. Xem xét nghĩa vụ: Ban Chấp hành xem xét nghĩa vụ của hội viên và "
        "quyết định việc chấm dứt tư cách hội viên.\n\n"
        "3. Hoàn thành nghĩa vụ: Hội viên phải hoàn thành tất cả nghĩa vụ còn "
        "tồn tại trước khi quyết định có hiệu lực."
    )

    result = GroundingValidator().validate(answer, source)

    assert result.grounded is False
    assert any("Hoàn thành nghĩa vụ" in error for error in result.errors)


def test_generation_result() -> None:
    llm = MockLocalLLM("Doanh nghiệp đáp ứng điều kiện đăng ký [1].")
    result = pipeline(llm).generate(
        GenerationRequest(
            question_id="q-1",
            question="Điều kiện đăng ký là gì?",
            retrieval_result=retrieval(),
        )
    )
    assert result.question_id == "q-1"
    assert result.grounded is True
    assert result.citations[0].article == "Điều 37"
    assert result.validation_errors == []


def generation_request(*, with_evidence: bool = True) -> GenerationRequest:
    return GenerationRequest(
        question_id="q-repair",
        question="Điều kiện đăng ký là gì?",
        retrieval_result=retrieval(with_evidence=with_evidence),
    )


def test_no_repair_when_citation_valid() -> None:
    llm = ScriptedLocalLLM(["Doanh nghiệp đáp ứng điều kiện [1]."])
    result = pipeline_with_repair(llm).generate(generation_request())
    assert llm.calls == 1
    assert result.grounded is True
    assert [attempt.attempt for attempt in result.attempts] == [0]


def test_repair_when_citation_missing() -> None:
    llm = ScriptedLocalLLM(
        [
            "Doanh nghiệp đáp ứng điều kiện đăng ký.",
            "Doanh nghiệp đáp ứng điều kiện đăng ký [1].",
        ]
    )
    result = pipeline_with_repair(llm).generate(generation_request())
    assert llm.calls == 2
    assert result.grounded is True
    assert result.answer.endswith("[1].")


def test_repair_max_one_retry() -> None:
    llm = ScriptedLocalLLM(["Thiếu citation.", "Vẫn thiếu citation."])
    result = pipeline_with_repair(llm).generate(generation_request())
    assert llm.calls == 2
    assert result.grounded is False
    assert len(result.attempts) == 2


def test_repair_generation_is_token_bounded() -> None:
    llm = ScriptedLocalLLM(["Không dùng"])
    repair = CitationRepair(llm, max_new_tokens=512)
    assert repair.max_new_tokens == 192


def test_repair_preserves_evidence_mapping() -> None:
    llm = ScriptedLocalLLM(["Thiếu citation.", "Đã sửa [1]."])
    pipeline_with_repair(llm).generate(generation_request())
    repair_prompt = llm.prompts[1]
    assert "[1]\nTên văn bản: Luật Doanh nghiệp" in repair_prompt
    assert "Điều: Điều 37" in repair_prompt
    assert "[/1]" in repair_prompt
    assert "[2]" not in repair_prompt
    assert "Không đủ căn cứ trong tài liệu được cung cấp." in repair_prompt
    conversation = QwenGenerator._conversation(repair_prompt)
    assert [message["role"] for message in conversation] == ["system", "user"]
    assert conversation[1]["content"].startswith("### Câu hỏi\n")


def test_invalid_repaired_citation_is_rejected() -> None:
    llm = ScriptedLocalLLM(["Thiếu citation.", "Citation không tồn tại [99]."])
    result = pipeline_with_repair(llm).generate(generation_request())
    assert result.grounded is False
    assert any("does not exist" in error for error in result.validation_errors)


def test_repaired_answer_is_revalidated() -> None:
    llm = ScriptedLocalLLM(["Thiếu citation.", "Theo Điều 100, nội dung mới [1]."])
    result = pipeline_with_repair(llm).generate(generation_request())
    assert result.attempts[1].citations_valid is True
    assert result.attempts[1].grounded is False
    assert any(
        "Unsupported legal metadata" in error for error in result.validation_errors
    )


def test_no_repair_when_no_evidence() -> None:
    llm = ScriptedLocalLLM([])
    result = pipeline_with_repair(llm).generate(
        generation_request(with_evidence=False)
    )
    assert llm.calls == 0
    assert result.abstained is True


def test_model_failure_does_not_trigger_repair() -> None:
    llm = ScriptedLocalLLM([RuntimeError("model inference failed")])
    with pytest.raises(RuntimeError, match="model inference failed"):
        pipeline_with_repair(llm).generate(generation_request())
    assert llm.calls == 1


def test_unsupported_grounding_does_not_trigger_repair() -> None:
    llm = ScriptedLocalLLM(["Theo Điều 100, nội dung mới [1]."])
    result = pipeline_with_repair(llm).generate(generation_request())
    assert llm.calls == 1
    assert result.grounded is False
    assert any(
        "Unsupported legal metadata" in error for error in result.validation_errors
    )
