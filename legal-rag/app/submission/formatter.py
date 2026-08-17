"""Convert generated answers to the exact Subtask 2 schema."""

import re

from app.domain.generation import GeneratedAnswer
from app.domain.submission import SubmissionAnswer

_CITATION = re.compile(r"\s*\[\d+\]")
# "(Ngữ cảnh 1)" points at the prompt layout, not at anything a reader sees.
_CONTEXT_REF = re.compile(r"\s*\(\s*Ngữ cảnh\s*\d+\s*\)", flags=re.IGNORECASE)
# "Văn bản 2 (Ngữ cảnh 2):" is a heading the model invents for the prompt blocks.
_CONTEXT_HEADING = re.compile(
    r"\**\s*Văn bản\s*\d+\s*\**\s*:?\s*", flags=re.IGNORECASE
)
# Corpus filenames leak through the evidence metadata, e.g.
# "(Nghi-dinh-01-2021-ND-CP-dang-ky-doanh-nghiep-283247)". Require several
# hyphen-joined tokens and no spaces so real parentheticals survive.
_DOCUMENT_SLUG = re.compile(r"\s*\(\s*\w+(?:-\w+){3,}[^)]*\)")
# The answer can be cut off mid-slug when generation hits its token budget,
# leaving the parenthesis unclosed.
_TRUNCATED_SLUG = re.compile(r"\s*\(\s*\w+(?:-\w+){2,}[^)]*$")
# A filename tail can also ride along an inline reference:
# "Quyết định 02-2023-QD-KTNN-lap-tham-dinh-...-554186". Keep the document
# code, drop the descriptive slug; the year+agency shape keeps this narrow.
_INLINE_SLUG_TAIL = re.compile(
    r"(\d+-\d{4}-[A-ZĐ]+(?:-[A-ZĐ]+)*)-[a-z0-9][a-z0-9-]*"
)
_LEAD_IN = re.compile(
    r"^\s*(?:Dựa\s+(?:trên|vào)|Theo|Căn\s+cứ|Trong)\s+(?:vào\s+)?"
    r"ngữ\s+cảnh[^,.:;]*[,.:;]\s*",
    flags=re.IGNORECASE,
)
# The model sometimes closes by grading its own answer. Anchored at the end and
# spelled out, because over-deleting costs recall, which METEOR weights heavily.
_TRAILING_META = re.compile(
    r"\s*(?:Vì\s+vậy|Do\s+đó|Tóm\s+lại)?[,\s]*"
    r"(?:đây\s+là|câu\s+trả\s+lời)\s+(?:thông\s+tin\s+)?"
    r"(?:chính\s+xác|đầy\s+đủ)[^.!?]*[.!?]\s*$",
    flags=re.IGNORECASE,
)
_MARKDOWN = re.compile(r"[*_`#]+")
_LIST_MARKER = re.compile(r"^[ \t]*(?:\d+[.)]|[-•>]+)[ \t]+", flags=re.MULTILINE)
# Removing a heading can pull a bullet onto the previous line, e.g.
# "... sau: - Điều 50 ...".
_INLINE_BULLET = re.compile(r"(?<=[:;.])\s+[-•>]+\s+")
_DANGLING = re.compile(r"\s+([,.;:])")
_REPEATED_PUNCTUATION = re.compile(r"([,.;:])(?:\s*[,;:])+")
_WHITESPACE = re.compile(r"\s+")


def normalize_answer(answer: str) -> str:
    """Strip scaffolding the reference answers never contain.

    METEOR and ROUGE-L compare tokens against expert prose, so citation
    markers, prompt-layout references, corpus filenames, markdown and bullets
    only cost precision. Dropping ``[1]`` can leave ``Theo , ...``, so
    punctuation is repaired after the markers are gone.
    """
    text = _CITATION.sub("", answer)
    text = _CONTEXT_REF.sub("", text)
    text = _DOCUMENT_SLUG.sub("", text)
    text = _CONTEXT_HEADING.sub("", text)
    text = _MARKDOWN.sub(" ", text)
    text = _LIST_MARKER.sub("", text)
    text = _INLINE_BULLET.sub(" ", text)
    text = _LEAD_IN.sub("", text)
    text = _TRUNCATED_SLUG.sub("", text)
    text = _INLINE_SLUG_TAIL.sub(r"\1", text)
    text = _TRAILING_META.sub("", text)
    text = _DANGLING.sub(r"\1", text)
    text = _REPEATED_PUNCTUATION.sub(r"\1", text)
    text = _WHITESPACE.sub(" ", text).strip()
    text = _DANGLING.sub(r"\1", text)
    return text[:1].upper() + text[1:] if text else text


class SubmissionFormatter:
    def format(self, answers: list[GeneratedAnswer]) -> dict[str, SubmissionAnswer]:
        """Use question IDs as keys and retain only the answer field."""
        submission: dict[str, SubmissionAnswer] = {}
        for generated in answers:
            if not generated.question_id:
                raise ValueError("Generated answer is missing question_id")
            if generated.question_id in submission:
                raise ValueError(f"Duplicate question_id: {generated.question_id}")
            submission[generated.question_id] = SubmissionAnswer(
                answer=normalize_answer(generated.answer)
            )
        return submission
