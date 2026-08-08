"""Grounded generation orchestration tests."""

from app.domain.queries import LegalQuery
from app.domain.retrieval import LegalContext
from app.generation.pipeline import GenerationPipeline
from app.generation.prompts.legal_answer import LegalPromptBuilder
from app.generation.validation.abstention_validator import AbstentionValidator
from app.generation.validation.citation_validator import CitationValidator
from app.generation.validation.grounding_validator import GroundingValidator


class _Generator:
    def load(self) -> None:
        pass

    def generate(self, prompt: str) -> str:
        return prompt

    def unload(self) -> None:
        pass


def test_generation_pipeline_abstains_on_insufficient_evidence() -> None:
    pipeline = GenerationPipeline(
        _Generator(),  # type: ignore[arg-type]
        LegalPromptBuilder(),
        CitationValidator(),
        GroundingValidator(),
        AbstentionValidator("Không đủ căn cứ."),
    )
    context = LegalContext(query="q", evidences=[], formatted_context="", token_count=0)
    answer = pipeline.generate(LegalQuery(question_id="1", question="q"), context)
    assert answer.abstained
    assert answer.answer == "Không đủ căn cứ."
