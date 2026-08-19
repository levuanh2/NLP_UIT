"""Grounded local generation orchestration."""

import logging
import time

from app.domain.generation import (
    CitationValidationResult,
    GeneratedAnswer,
    GenerationAttempt,
    GenerationRequest,
    GroundingResult,
)
from app.domain.retrieval import LegalEvidence, RetrievalResult
from app.generation.citation_repair import CitationRepair
from app.generation.llm.base import BaseLLMGenerator
from app.generation.prompts.legal_answer import LegalPromptBuilder
from app.generation.validation.abstention_validator import AbstentionValidator
from app.generation.validation.citation_validator import CitationValidator
from app.generation.validation.grounding_validator import GroundingValidator

logger = logging.getLogger(__name__)


class GenerationPipeline:
    def __init__(
        self,
        generator: BaseLLMGenerator,
        prompt_builder: LegalPromptBuilder,
        citation_validator: CitationValidator,
        grounding_validator: GroundingValidator,
        abstention_validator: AbstentionValidator,
        citation_repair: CitationRepair | None = None,
        *,
        max_new_tokens: int = 192,
        temperature: float = 0.0,
        trace: bool = False,
    ) -> None:
        self.generator = generator
        self.prompt_builder = prompt_builder
        self.citation_validator = citation_validator
        self.grounding_validator = grounding_validator
        self.abstention_validator = abstention_validator
        self.citation_repair = citation_repair
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.trace = trace
        if trace and hasattr(generator, "debug_enabled"):
            generator.debug_enabled = True

    def _prepare(self, request: GenerationRequest):
        """Return (prompt, abstention, build_seconds); exactly one of the first two."""
        retrieval = request.retrieval_result
        if self.abstention_validator.should_abstain(retrieval):
            return None, self._abstention(request, "No usable retrieval evidence."), 0.0
        prompt_started = time.perf_counter()
        grounded_prompt = self.prompt_builder.prepare(
            request.question,
            retrieval,
            token_counter=self.generator.count_tokens,
        )
        prompt_build_seconds = time.perf_counter() - prompt_started
        if not grounded_prompt.evidences:
            return (
                None,
                self._abstention(
                    request,
                    "No complete evidence block fits the model context window.",
                ),
                prompt_build_seconds,
            )
        self._trace(
            "prompt",
            question=request.question,
            active_index_version=retrieval.active_index_version,
            evidence_ids=[item.evidence_id for item in grounded_prompt.evidences],
            prompt_token_count=grounded_prompt.token_count,
            context_token_count=grounded_prompt.context_token_count,
            evidence_count=len(grounded_prompt.evidences),
            parent_count=len({item.parent_id for item in grounded_prompt.evidences}),
            prompt_build_seconds=prompt_build_seconds,
            model_name=getattr(self.generator, "model_name", "injected-local-model"),
        )
        return grounded_prompt, None, prompt_build_seconds

    def generate(self, request: GenerationRequest) -> GeneratedAnswer:
        """Build bounded context, invoke the local model, and validate output."""
        grounded_prompt, abstention, prompt_build_seconds = self._prepare(request)
        if grounded_prompt is None:
            return abstention
        raw_answer, original_latency = self._invoke(grounded_prompt)
        return self._finish(
            request, grounded_prompt, raw_answer, original_latency,
            prompt_build_seconds,
        )

    def generate_batch(
        self, requests: list[GenerationRequest]
    ) -> list[GeneratedAnswer]:
        """Answer several requests, sharing one decode pass across them.

        Retrieval and validation stay per request; only the model call is
        pooled, because that is the part bound by reading weights out of VRAM.
        """
        prepared = [self._prepare(request) for request in requests]
        pending = [i for i, p in enumerate(prepared) if p[0] is not None]
        answers: list[GeneratedAnswer | None] = [p[1] for p in prepared]
        if not pending:
            return [a for a in answers if a is not None]

        started = time.perf_counter()
        raw = self.generator.generate_many(
            [prepared[i][0].prompt for i in pending],
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        # One wall-clock reading covers the whole batch, so each member is
        # charged its share rather than the total.
        share = (time.perf_counter() - started) / len(pending)
        for position, index in enumerate(pending):
            grounded_prompt, _, build_seconds = prepared[index]
            answers[index] = self._finish(
                requests[index], grounded_prompt, raw[position].strip(),
                share, build_seconds,
            )
        return [a for a in answers if a is not None]

    def _invoke(self, grounded_prompt) -> tuple[str, float]:
        started = time.perf_counter()
        raw_answer = self.generator.generate(
            grounded_prompt.prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        ).strip()
        self._trace_model_io(attempt=0)
        return raw_answer, time.perf_counter() - started

    def _finish(
        self,
        request: GenerationRequest,
        grounded_prompt,
        raw_answer: str,
        original_latency: float,
        prompt_build_seconds: float,
    ) -> GeneratedAnswer:
        retrieval = request.retrieval_result
        validation_started = time.perf_counter()
        citations, grounding, errors = self._validate(
            raw_answer, retrieval, grounded_prompt.evidences
        )
        validation_seconds = time.perf_counter() - validation_started
        attempts = [
            GenerationAttempt(
                answer=raw_answer,
                attempt=0,
                citations_valid=citations.valid,
                grounded=grounding.grounded,
                validation_errors=errors,
                latency_seconds=original_latency,
                validation_seconds=validation_seconds,
                metrics=self._last_generation_metrics(),
            )
        ]
        self._trace_attempt(attempts[-1])

        if self._should_repair(errors, grounded_prompt.evidences):
            repair_source = retrieval.model_copy(
                update={"evidences": grounded_prompt.evidences}
            )
            repair_started = time.perf_counter()
            raw_answer = self.citation_repair.repair(
                question=request.question,
                answer=raw_answer,
                retrieval_result=repair_source,
            ).strip()
            self._trace_model_io(attempt=1)
            repair_latency = time.perf_counter() - repair_started
            validation_started = time.perf_counter()
            citations, grounding, errors = self._validate(
                raw_answer, retrieval, grounded_prompt.evidences
            )
            validation_seconds = time.perf_counter() - validation_started
            attempts.append(
                GenerationAttempt(
                    answer=raw_answer,
                    attempt=1,
                    citations_valid=citations.valid,
                    grounded=grounding.grounded,
                    validation_errors=errors,
                    latency_seconds=repair_latency,
                    validation_seconds=validation_seconds,
                    metrics=self._last_generation_metrics(),
                )
            )
            self._trace_attempt(attempts[-1])

        result = GeneratedAnswer(
            question_id=request.question_id,
            answer=raw_answer,
            grounded=grounding.grounded and citations.valid,
            citations=citations.citations,
            confidence=None,
            validation_errors=errors,
            evidence_ids=[item.evidence_id for item in grounded_prompt.evidences],
            abstained=self.citation_validator.is_safe_abstention(raw_answer),
            attempts=attempts,
            prompt_build_seconds=prompt_build_seconds,
            context_tokens=grounded_prompt.context_token_count,
            evidence_count=len(grounded_prompt.evidences),
            parent_count=len({item.parent_id for item in grounded_prompt.evidences}),
        )
        self._trace(
            "validation",
            generation_token_count=self.generator.count_tokens(raw_answer),
            grounding_result=grounding.model_dump(),
            citation_result=citations.model_dump(),
        )
        return result

    def _last_generation_metrics(self):
        metrics = getattr(self.generator, "last_generation_metrics", None)
        return metrics.model_copy(deep=True) if metrics is not None else None

    def _validate(
        self,
        answer: str,
        retrieval: RetrievalResult,
        evidences: list[LegalEvidence],
    ) -> tuple[CitationValidationResult, GroundingResult, list[str]]:
        citations = self.citation_validator.validate(
            answer, retrieval, evidences=evidences
        )
        grounding = self.grounding_validator.validate(
            answer,
            retrieval,
            evidences=evidences,
            citation_result=citations,
        )
        errors = list(dict.fromkeys(grounding.errors + citations.errors))
        return citations, grounding, errors

    def _should_repair(
        self, errors: list[str], evidences: list[LegalEvidence]
    ) -> bool:
        if self.citation_repair is None or not evidences or not errors:
            return False
        if not self.citation_repair.enabled or self.citation_repair.max_retries == 0:
            return False
        repairable_prefixes = (
            "Answer contains no evidence citation.",
            "Citation [",
            "Answer has no cited supporting evidence.",
        )
        return all(error.startswith(repairable_prefixes) for error in errors)

    def _trace_attempt(self, attempt: GenerationAttempt) -> None:
        self._trace(
            "generation_attempt",
            attempt=attempt.attempt,
            citation_valid=attempt.citations_valid,
            grounded=attempt.grounded,
            latency_seconds=attempt.latency_seconds,
            validation_seconds=attempt.validation_seconds,
            generation_metrics=(
                attempt.metrics.model_dump() if attempt.metrics is not None else None
            ),
            validation_errors=attempt.validation_errors,
        )

    def _trace_model_io(self, *, attempt: int) -> None:
        """Expose exact local model I/O only in explicitly enabled trace mode."""
        if not self.trace:
            return
        debug = getattr(self.generator, "last_generation_debug", None)
        if debug is not None:
            self._trace("model_io", attempt=attempt, **debug)

    def _abstention(self, request: GenerationRequest, reason: str) -> GeneratedAnswer:
        self._trace(
            "abstention",
            question=request.question,
            active_index_version=request.retrieval_result.active_index_version,
            reason=reason,
        )
        return GeneratedAnswer(
            question_id=request.question_id,
            answer=self.abstention_validator.abstention_message,
            grounded=True,
            citations=[],
            confidence=None,
            validation_errors=[],
            evidence_ids=[],
            abstained=True,
        )

    def _trace(self, stage: str, **details: object) -> None:
        if self.trace:
            logger.info("generation_trace stage=%s details=%s", stage, details)
