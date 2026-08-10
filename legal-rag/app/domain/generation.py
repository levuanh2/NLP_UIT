"""Grounded answer-generation models."""

from pydantic import BaseModel, Field

from app.domain.retrieval import RetrievalResult


class Citation(BaseModel):
    document_id: int | str
    document_name: str | None = None
    source_link: str | None = None
    chapter: str | None = None
    article: str | None = None
    clause: str | None = None
    point: str | None = None
    child_id: str | None = None
    evidence_id: str | None = None


class GenerationRequest(BaseModel):
    question_id: str | None = None
    question: str
    retrieval_result: RetrievalResult


class CitationValidationResult(BaseModel):
    valid: bool
    citation_ids: list[int] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class GroundingResult(BaseModel):
    grounded: bool
    errors: list[str] = Field(default_factory=list)


class GenerationMetrics(BaseModel):
    input_tokens: int
    max_new_tokens: int
    generated_tokens: int
    tokenize_seconds: float
    generation_seconds: float
    decode_seconds: float
    tokens_per_second: float


class GenerationAttempt(BaseModel):
    answer: str
    attempt: int
    citations_valid: bool
    grounded: bool
    validation_errors: list[str] = Field(default_factory=list)
    latency_seconds: float | None = None
    validation_seconds: float | None = None
    metrics: GenerationMetrics | None = None


class GeneratedAnswer(BaseModel):
    question_id: str | None = None
    answer: str
    grounded: bool
    citations: list[Citation] = Field(default_factory=list)
    confidence: float | None = None
    validation_errors: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    abstained: bool = False
    attempts: list[GenerationAttempt] = Field(default_factory=list, exclude=True)
    prompt_build_seconds: float | None = Field(default=None, exclude=True)
    context_tokens: int = Field(default=0, exclude=True)
    evidence_count: int = Field(default=0, exclude=True)
    parent_count: int = Field(default=0, exclude=True)
