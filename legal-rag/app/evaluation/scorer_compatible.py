"""Local metrics matching the organizer scoring program's tokenization."""

import re
from dataclasses import dataclass

_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class LegalQAScore:
    meteor: float
    rouge_l: float
    sample_count: int


class _NoWordNet:
    """Disable English synonym expansion for deterministic offline Vietnamese."""

    @staticmethod
    def synsets(*args: object, **kwargs: object) -> list[object]:
        return []


def organizer_rouge_tokens(text: str) -> list[str]:
    """Replicate the bundled scorer's ASCII-only ROUGE tokenizer."""
    normalized = _NON_ALPHANUMERIC.sub(" ", text.lower())
    return [token for token in _SPACES.split(normalized) if token]


def rouge_l_fmeasure(reference: str, prediction: str) -> float:
    """Compute ROUGE-L F-measure with organizer-compatible tokens."""
    reference_tokens = organizer_rouge_tokens(reference)
    prediction_tokens = organizer_rouge_tokens(prediction)
    if not reference_tokens or not prediction_tokens:
        return 0.0
    lcs_length = _lcs_length(reference_tokens, prediction_tokens)
    if lcs_length == 0:
        return 0.0
    precision = lcs_length / len(prediction_tokens)
    recall = lcs_length / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def meteor_score_offline(reference: str, prediction: str) -> float:
    """Compute scorer-style whitespace METEOR without network downloads."""
    try:
        from nltk.translate.meteor_score import meteor_score
    except ImportError as exc:  # pragma: no cover - dependency error
        raise RuntimeError("nltk is required for scorer-compatible METEOR.") from exc
    return float(
        meteor_score(
            [reference.split()],
            prediction.split(),
            wordnet=_NoWordNet(),
        )
    )


def score_answer_pairs(references: list[str], predictions: list[str]) -> LegalQAScore:
    """Return mean METEOR and ROUGE-L for aligned answers."""
    if len(references) != len(predictions):
        raise ValueError("Reference and prediction counts do not match.")
    if not references:
        return LegalQAScore(meteor=0.0, rouge_l=0.0, sample_count=0)
    meteor_values = [
        meteor_score_offline(reference, prediction)
        for reference, prediction in zip(references, predictions, strict=True)
    ]
    rouge_values = [
        rouge_l_fmeasure(reference, prediction)
        for reference, prediction in zip(references, predictions, strict=True)
    ]
    count = len(references)
    return LegalQAScore(
        meteor=sum(meteor_values) / count,
        rouge_l=sum(rouge_values) / count,
        sample_count=count,
    )


def _lcs_length(left: list[str], right: list[str]) -> int:
    """Compute LCS length with O(min(n, m)) memory."""
    if len(left) > len(right):
        left, right = right, left
    row = [0] * (len(left) + 1)
    for right_token in right:
        diagonal = 0
        for index, left_token in enumerate(left, start=1):
            previous = row[index]
            if left_token == right_token:
                row[index] = diagonal + 1
            else:
                row[index] = max(row[index], row[index - 1])
            diagonal = previous
    return row[-1]
