"""Generation pipeline composition root."""

from app.domain.generation import GeneratedAnswer
from app.domain.queries import LegalQuery
from app.domain.retrieval import LegalContext
from app.generation.llm.base import BaseLLMGenerator
from app.generation.prompts.legal_answer import LegalPromptBuilder
from app.generation.validation.abstention_validator import AbstentionValidator
from app.generation.validation.citation_validator import CitationValidator
from app.generation.validation.grounding_validator import GroundingValidator


class GenerationPipeline:
    def __init__(
        self,
        generator: BaseLLMGenerator,
        prompt_builder: LegalPromptBuilder,
        citation_validator: CitationValidator,
        grounding_validator: GroundingValidator,
        abstention_validator: AbstentionValidator,
    ) -> None:
        self.generator = generator
        self.prompt_builder = prompt_builder
        self.citation_validator = citation_validator
        self.grounding_validator = grounding_validator
        self.abstention_validator = abstention_validator

    def generate(
        self, query: LegalQuery, context: LegalContext
    ) -> GeneratedAnswer:
        """Build prompt, generate, validate citations/grounding, and abstain."""
        # TODO(phase-implementation):
        # Implement safe local generation orchestration.
        raise NotImplementedError
