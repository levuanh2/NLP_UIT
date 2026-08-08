"""Conservative Vietnamese legal text normalization."""

import re
import unicodedata


class LegalTextCleaner:
    def clean(self, text: str) -> str:
        """Normalize passage layout while preserving legal markers."""
        if not isinstance(text, str):
            raise TypeError("Legal text must be a string.")
        normalized = unicodedata.normalize("NFC", text).replace("\u00a0", " ")
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        lines = [
            re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")
        ]
        output: list[str] = []
        blank = False
        for line in lines:
            if line:
                output.append(line)
                blank = False
            elif output and not blank:
                output.append("")
                blank = True
        return "\n".join(output).strip()
