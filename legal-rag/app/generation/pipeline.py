"""Grounded generation pipeline composition root."""

import re

from app.domain.generation import Citation, GeneratedAnswer
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
        require_citation: bool = True,
        grounded_only: bool = True,
    ) -> None:
        self.generator = generator
        self.prompt_builder = prompt_builder
        self.citation_validator = citation_validator
        self.grounding_validator = grounding_validator
        self.abstention_validator = abstention_validator
        self.require_citation = require_citation
        self.grounded_only = grounded_only

    def generate(self, query: LegalQuery, context: LegalContext) -> GeneratedAnswer:
        """Build prompt, generate, validate citations/grounding, and abstain."""
        if self.abstention_validator.should_abstain(context):
            return self._abstention(query.question_id)
        prompt = self.prompt_builder.build(query.question, context)
        text = self.generator.generate(prompt).strip()
        if not text:
            return self._abstention(query.question_id)
        citation_positions = {
            int(value)
            for value in re.findall(
                r"(?:\[|\()(?:Trích từ\s+)?E(\d+)(?:\]|\))", text, re.IGNORECASE
            )
        }
        cited_evidence = [
            context.evidences[index - 1]
            for index in sorted(citation_positions)
            if 1 <= index <= len(context.evidences)
        ]
        if not cited_evidence and context.evidences:
            cited_evidence = context.evidences[:1]
        clean_text = re.sub(
            r"\s*(?:\[|\()(?:Trích từ\s+)?E\d+(?:\]|\))",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        citations = [
            Citation(
                document_name=item.document_name,
                document_number=None,
                article=item.article,
                clause=item.clause,
                point=item.point,
                evidence_id=item.evidence_id,
            )
            for item in cited_evidence
        ]
        answer = GeneratedAnswer(
            question_id=query.question_id,
            answer=clean_text,
            citations=citations,
            evidence_ids=[item.evidence_id for item in cited_evidence],
            grounded=None,
            confidence=None,
        )
        citations_valid = self.citation_validator.validate(answer, context)
        if self.require_citation and not citations:
            citations_valid = False
        grounded = self.grounding_validator.validate(answer, context)
        confidence = (float(citations_valid) + float(grounded)) / 2.0
        answer = answer.model_copy(
            update={"grounded": grounded and citations_valid, "confidence": confidence}
        )
        if self.grounded_only and not answer.grounded:
            return self._abstention(query.question_id)
        return answer

    def _abstention(self, question_id: str) -> GeneratedAnswer:
        return GeneratedAnswer(
            question_id=question_id,
            answer=self.abstention_validator.abstention_message,
            grounded=False,
            confidence=0.0,
            abstained=True,
        )

    def unload(self) -> None:
        """Release the generator after a question batch."""
        self.generator.unload()
