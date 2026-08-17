"""Environment and YAML configuration loading."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.exceptions import ConfigurationError
from app.core.paths import ProjectPaths


class Settings(BaseSettings):
    """Deployment settings read from environment variables or ``.env``."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "legal-rag"
    app_env: str = "development"
    app_debug: bool = False
    log_level: str = "INFO"

    project_root: Path = Path(".")
    data_dir: Path = Path("./data")
    corpus_data_dir: Path = Path("./data/corpus")
    question_data_dir: Path = Path("./data/questions")
    output_dir: Path = Path("./data/outputs")
    cache_data_dir: Path = Path("./data/cache")
    model_dir: Path = Path("./models")
    faiss_dir: Path = Path("./storage/faiss")
    bm25_dir: Path = Path("./storage/bm25")
    sqlite_dir: Path = Path("./storage/sqlite")
    sqlite_database_path: Path = Path("./storage/sqlite/legal.db")
    index_root_dir: Path = Path("./storage/indexes")
    checkpoint_dir: Path = Path("./storage/checkpoints")
    config_dir: Path = Path("./configs")

    base_config_path: Path = Path("./configs/base.yaml")
    models_config_path: Path = Path("./configs/models.yaml")
    chunking_config_path: Path = Path("./configs/chunking.yaml")
    retrieval_config_path: Path = Path("./configs/retrieval.yaml")
    generation_config_path: Path = Path("./configs/generation.yaml")
    evaluation_config_path: Path = Path("./configs/evaluation.yaml")

    llm_model_name: str
    embedding_model_name: str
    reranker_model_name: str
    model_local_files_only: bool = True
    model_trust_remote_code: bool = False
    model_device: str = "auto"
    model_dtype: str = "auto"
    model_quantization: str = "none"
    embedding_device: str = "cpu"
    reranker_device: str = "cpu"

    parent_target_tokens: int = Field(default=1024, gt=0)
    parent_max_tokens: int = Field(default=2048, gt=0)
    child_target_tokens: int = Field(default=256, gt=0)
    child_max_tokens: int = Field(default=384, gt=0)
    metadata_filter_enabled: bool = True
    metadata_filter_min_confidence: float = Field(default=0.8, ge=0, le=1)
    metadata_filter_fallback_to_full_corpus: bool = True
    dense_top_n: int = Field(
        default=20, gt=0, validation_alias=AliasChoices("DENSE_TOP_K", "DENSE_TOP_N")
    )
    bm25_top_n: int = Field(
        default=20, gt=0, validation_alias=AliasChoices("BM25_TOP_K", "BM25_TOP_N")
    )
    fusion_top_n: int = Field(
        default=30, gt=0, validation_alias=AliasChoices("RRF_TOP_K", "FUSION_TOP_N")
    )
    rerank_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("RERANKER_ENABLED", "RERANK_ENABLED"),
    )
    rerank_top_k: int = Field(
        default=10,
        gt=0,
        validation_alias=AliasChoices("RERANKER_TOP_K", "RERANK_TOP_K"),
    )
    rrf_k: int = Field(default=60, gt=0)
    retrieval_trace: bool = False
    max_new_tokens: int = Field(
        default=192,
        gt=0,
        validation_alias=AliasChoices("LLM_MAX_NEW_TOKENS", "MAX_NEW_TOKENS"),
    )
    temperature: float = Field(
        default=0.0,
        ge=0,
        validation_alias=AliasChoices("LLM_TEMPERATURE", "TEMPERATURE"),
    )
    top_p: float = Field(
        default=1.0, validation_alias=AliasChoices("LLM_TOP_P", "TOP_P"), ge=0, le=1
    )
    do_sample: bool = Field(
        default=False,
        validation_alias=AliasChoices("LLM_DO_SAMPLE", "DO_SAMPLE"),
    )
    min_new_tokens: int = Field(default=0, ge=0)
    repetition_penalty: float = Field(default=1.1, gt=0)
    llm_max_context_tokens: int = Field(default=4096, gt=0)
    llm_citation_repair_enabled: bool = True
    llm_citation_repair_max_retries: int = Field(default=1, ge=0, le=1)
    generation_trace: bool = False
    require_citation: bool = True
    grounded_only: bool = True
    abstain_when_insufficient: bool = True
    submission_filename: str = "submission.json"
    submission_encoding: str = "utf-8"
    submission_ensure_ascii: bool = False
    ingestion_resume: bool = True
    ingestion_chunk_batch_size: int = Field(default=1000, gt=0)
    ingestion_embedding_batch_size: int = Field(default=64, ge=1, le=128)
    ingestion_checksum_algorithm: str = "sha256"
    chunking_version: str = "v2"
    ingestion_continue_on_document_error: bool = True
    max_document_words_warning: int = Field(default=100_000, gt=0)
    sqlite_batch_size: int = Field(default=1000, gt=0)
    faiss_index_type: str = "auto"
    faiss_normalize: bool = True
    index_atomic_publish: bool = True
    context_neighbor_window: int = Field(default=1, ge=0)
    max_parents_per_document: int = Field(default=3, gt=0)
    context_max_tokens: int = Field(default=6000, gt=0)

    def paths(self) -> ProjectPaths:
        """Return all path settings as a typed value object."""
        return ProjectPaths(
            **{name: getattr(self, name) for name in ProjectPaths.model_fields}
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached validated settings."""
    return Settings()  # type: ignore[call-arg]


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load one YAML mapping without resolving environment references."""
    if not path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {path}")
    return value
