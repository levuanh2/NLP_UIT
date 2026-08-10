"""Legal cleaner tests."""

from app.ingestion.cleaners.legal_text_cleaner import LegalTextCleaner


def test_cleaner_preserves_legal_headings() -> None:
    text = " CHƯƠNG I\r\n\r\n  Điều 1.  Phạm vi  \r1. Nội dung\t"

    cleaned = LegalTextCleaner().clean(text)

    assert "CHƯƠNG I" in cleaned
    assert "Điều 1. Phạm vi" in cleaned
    assert "1. Nội dung" in cleaned
    assert "\r" not in cleaned
    assert "  " not in cleaned
