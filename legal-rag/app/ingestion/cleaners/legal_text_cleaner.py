"""Conservative normalization for Vietnamese legal text."""

import re


class LegalTextCleaner:
    def clean(self, text: str) -> str:
        """Normalize passage layout while preserving legal markers."""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.replace("\u00a0", " ")
        lines: list[str] = []
        previous_blank = False
        for raw_line in normalized.split("\n"):
            line = re.sub(r"[\t ]+", " ", raw_line).strip()
            if not line:
                if lines and not previous_blank:
                    lines.append("")
                previous_blank = True
                continue
            lines.append(line)
            previous_blank = False
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)
