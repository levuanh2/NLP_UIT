"""Legal cleaner test skeleton."""

import pytest

from app.ingestion.cleaners.legal_text_cleaner import LegalTextCleaner


@pytest.mark.skip(reason="TODO(phase-implementation): implement legal cleaning")
def test_legal_text_cleaner_preserves_article_structure() -> None:
    # Arrange
    text = "Điều 1. Phạm vi điều chỉnh\n\nNội dung."

    # Act
    cleaned = LegalTextCleaner().clean(text)

    # Assert
    assert "Điều 1." in cleaned
