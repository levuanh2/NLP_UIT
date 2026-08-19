"""Configuration tests."""

from pathlib import Path

from app.core.config import Settings, load_yaml_config


def test_config_loads_environment_variables(monkeypatch) -> None:
    # Arrange
    monkeypatch.setenv("LLM_MODEL_NAME", "local/test-llm")
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "local/test-embedding")
    monkeypatch.setenv("RERANKER_MODEL_NAME", "local/test-reranker")
    monkeypatch.setenv("DENSE_TOP_N", "7")

    # Act
    settings = Settings(_env_file=None)

    # Assert
    assert settings.llm_model_name == "local/test-llm"
    assert settings.dense_top_n == 7


def test_yaml_config_has_mapping_root(tmp_path: Path) -> None:
    # Arrange
    config_path = tmp_path / "config.yaml"
    config_path.write_text("section:\n  enabled: true\n", encoding="utf-8")

    # Act
    config = load_yaml_config(config_path)

    # Assert
    assert config == {"section": {"enabled": True}}


def test_llm_prefixed_generation_settings_are_read(monkeypatch) -> None:
    """.env documents LLM_ prefixes; a field without the alias silently ignores them."""
    from app.core.config import Settings

    for name, value in (
        ("LLM_MIN_NEW_TOKENS", "700"),
        ("LLM_MAX_NEW_TOKENS", "1100"),
        ("LLM_REPETITION_PENALTY", "1.3"),
    ):
        monkeypatch.setenv(name, value)

    settings = Settings(
        llm_model_name="m", embedding_model_name="e", reranker_model_name="r"
    )

    assert settings.min_new_tokens == 700
    assert settings.max_new_tokens == 1100
    assert settings.repetition_penalty == 1.3
