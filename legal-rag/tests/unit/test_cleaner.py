"""Legal cleaner tests."""

from app.ingestion.cleaners.legal_text_cleaner import LegalTextCleaner


def test_legal_text_cleaner_preserves_article_structure() -> None:
    text = "Điều 1.  Phạm vi điều chỉnh\r\n\r\n\r\nNội dung."

    cleaned = LegalTextCleaner().clean(text)

    assert cleaned == "Điều 1. Phạm vi điều chỉnh\n\nNội dung."
