"""Convert generated answers to the exact Subtask 2 schema."""

import re

from app.domain.generation import GeneratedAnswer
from app.domain.submission import SubmissionAnswer

_CITATION = re.compile(r"\s*\[\d+\]")
# The prompt labels its evidence blocks "Ngữ cảnh 1"; the model points back at
# them, numbered or not. Expert prose never says "ngữ cảnh", so every mention
# is scaffolding — only the amount of surrounding text to take with it varies.
_CONTEXT_WORD = (
    r"(?:các\s+|những\s+)?ngữ\s*cảnh"
    r"(?:\s*\d+(?:\s*(?:,|và|-|đến)\s*(?:ngữ\s*cảnh\s*)?\d+)*)?"
    r"(?:\s+(?:được\s+)?(?:cung\s+cấp|nêu(?:\s+rõ)?|trích|đề\s+cập))?"
)
# A preposition binds the reference into the sentence: "theo ngữ cảnh 2".
_CONTEXT_PHRASE = (
    rf"(?:(?:dựa\s+(?:trên|vào)|theo|tại|trong|từ|xem)\s+)?{_CONTEXT_WORD}"
)
# "(Ngữ cảnh 1)", "(theo Ngữ cảnh 4 và 5)" hold nothing but the reference.
_CONTEXT_REF = re.compile(
    rf"\s*[(\[]\s*{_CONTEXT_PHRASE}\s*[)\]]", flags=re.IGNORECASE
)
# "(theo Điều 2, Khoản 1, Ngữ cảnh 1)" also carries a real citation; keep it.
_CONTEXT_IN_PAREN = re.compile(
    rf"[,;]?\s*{_CONTEXT_PHRASE}\s*(?=[)\]])", flags=re.IGNORECASE
)
# Evidence metadata can surface verbatim: "(Document ID: 128691)" and
# "Child ID: doc:231961:article:40:38:segment:0:child:0".
_DOCUMENT_ID_PAREN = re.compile(
    r"\s*\(\s*(?:Document|Child)\s+ID\s*:?\s*[\w:]+\s*\)", flags=re.IGNORECASE
)
_DOCUMENT_ID = re.compile(
    r"[,;]?\s*(?:Document|Child)\s+ID\s*:?\s*[\w:]+", flags=re.IGNORECASE
)
# "Thông tin này được cung cấp trong Ngữ cảnh 4." grades the source instead of
# stating the law, so the whole clause goes. The demonstrative opener is what
# makes the clause pure meta: "Dựa trên thông tin từ ngữ cảnh, Sổ theo dõi là
# Mẫu TP-TVPL-01" also mentions both, and carries the answer.
_CONTEXT_META = re.compile(
    rf"(?:^|(?<=[.!?]))\s*"
    rf"(?:Đây\s+là|Điều\s+này|Thông\s+tin\s+này|Nội\s+dung\s+này|Chi\s+tiết\s+này)"
    rf"[^.!?:]*{_CONTEXT_WORD}[^.!?:]*[.!?:]",
    flags=re.IGNORECASE,
)
# "Ngữ cảnh 1 (Luật Thuế 2007): Theo Điều 8 ..." labels a block and then states
# the law; only the label goes.
_CONTEXT_LABEL = re.compile(
    rf"(?:^|(?<=[.!?:]))\s*{_CONTEXT_WORD}\s*(?:\([^)]*\))?\s*[:.]",
    flags=re.IGNORECASE,
)
# Whatever survives is a bare mention mid-sentence; drop the phrase alone and
# let the punctuation repair below close the gap.
_CONTEXT_MENTION = re.compile(rf"\s*{_CONTEXT_PHRASE}", flags=re.IGNORECASE)
# "Văn bản 2 (Ngữ cảnh 2):" is a heading the model invents for the prompt blocks.
# "Câu trả lời cho câu hỏi "..." là:" restates the prompt before answering.
_ANSWER_PREAMBLE = re.compile(
    r"^\s*Câu\s+trả\s+lời\s+(?:cho\s+câu\s+hỏi\s*[\"“”].*?[\"“”]\s*)?"
    r"là\s*:?\s*",
    flags=re.IGNORECASE | re.DOTALL,
)
# The model sometimes grades the evidence before stating the law: "Không có
# thông tin cụ thể về X. Tuy nhiên, theo Điều 8 ...". Reference answers never
# hedge, so the opener shares no tokens with them and splits the METEOR
# alignment into extra chunks, which the fragmentation penalty cubes. Only the
# text up to the pivot goes; without a pivot there is nothing left to keep.
_HEDGE_OPENER = re.compile(
    r"^.{0,400}?(?:không\s+có\s+thông\s+tin|chưa\s+đủ\s+căn\s+cứ"
    r"|không\s+tìm\s+thấy|không\s+nêu\s+rõ)"
    r".{0,400}?Tuy\s+nhiên,\s*",
    flags=re.IGNORECASE | re.DOTALL,
)
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
# The tail can mix case ("...-trong-nha-truong-Quan-doi-534108"), so it runs to
# the end of the hyphenated token, not to the first capital.
_INLINE_SLUG_TAIL = re.compile(
    r"(\d+-\d{4}-[A-ZĐ]+(?:-[A-ZĐ]+)*)-[a-z0-9][A-Za-z0-9-]*"
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
# Stripping a lead-in can expose the comma that followed it, or empty out a
# parenthetical that held only metadata.
_LEADING_PUNCTUATION = re.compile(r"^[\s,.;:)\]]+")
_ORPHAN_PAREN_COMMA = re.compile(r"[,;]\s*(?=[)\]])")
_EMPTY_PAREN = re.compile(r"\s*[(\[]\s*[)\]]")
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
    text = _DOCUMENT_ID_PAREN.sub("", text)
    text = _DOCUMENT_ID.sub("", text)
    text = _CONTEXT_REF.sub("", text)
    text = _CONTEXT_IN_PAREN.sub("", text)
    text = _CONTEXT_META.sub("", text)
    text = _CONTEXT_LABEL.sub("", text)
    text = _DOCUMENT_SLUG.sub("", text)
    text = _CONTEXT_HEADING.sub("", text)
    text = _MARKDOWN.sub(" ", text)
    text = _LIST_MARKER.sub("", text)
    text = _INLINE_BULLET.sub(" ", text)
    text = _LEAD_IN.sub("", text)
    text = _CONTEXT_MENTION.sub("", text)
    text = _ANSWER_PREAMBLE.sub("", text)
    text = _HEDGE_OPENER.sub("", text)
    text = _TRUNCATED_SLUG.sub("", text)
    text = _INLINE_SLUG_TAIL.sub(r"\1", text)
    text = _TRAILING_META.sub("", text)
    text = _ORPHAN_PAREN_COMMA.sub("", text)
    text = _EMPTY_PAREN.sub("", text)
    text = _DANGLING.sub(r"\1", text)
    text = _REPEATED_PUNCTUATION.sub(r"\1", text)
    text = _WHITESPACE.sub(" ", text).strip()
    text = _DANGLING.sub(r"\1", text)
    text = _LEADING_PUNCTUATION.sub("", text)
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
