"""METEOR and ROUGE-L over Vietnamese answers, matching the competition scorer.

The reference implementation is nltk's ``meteor_score`` with default parameters.
Its stem and WordNet stages are English-only, so on Vietnamese they never fire
and the exact-match stage is the whole metric — which is what this module
computes, without needing the WordNet download to succeed.
"""

ALPHA = 0.9  # recall weight in F_mean; nltk's default
BETA = 3.0  # fragmentation exponent
GAMMA = 0.5  # fragmentation ceiling


def tokenize(text: str) -> list[str]:
    """Lowercased whitespace tokens, the only split that is safe for Vietnamese."""
    return text.lower().split()


def _align(hypothesis: list[str], reference: list[str]) -> list[tuple[int, int]]:
    """Pair each hypothesis token with the earliest unused identical reference token."""
    positions: dict[str, list[int]] = {}
    for index, token in enumerate(reference):
        positions.setdefault(token, []).append(index)
    cursor = dict.fromkeys(positions, 0)

    matches: list[tuple[int, int]] = []
    for hypothesis_index, token in enumerate(hypothesis):
        available = positions.get(token)
        if not available:
            continue
        offset = cursor[token]
        if offset < len(available):
            matches.append((hypothesis_index, available[offset]))
            cursor[token] = offset + 1
    return matches


def _chunks(matches: list[tuple[int, int]]) -> int:
    """Count runs that advance by one position in both sequences at once."""
    total = 0
    previous: tuple[int, int] | None = None
    for pair in matches:
        if previous is None or pair != (previous[0] + 1, previous[1] + 1):
            total += 1
        previous = pair
    return total


def meteor(prediction: str, reference: str) -> float:
    hypothesis_tokens, reference_tokens = tokenize(prediction), tokenize(reference)
    if not hypothesis_tokens or not reference_tokens:
        return 0.0

    matches = _align(hypothesis_tokens, reference_tokens)
    matched = len(matches)
    if not matched:
        return 0.0

    precision = matched / len(hypothesis_tokens)
    recall = matched / len(reference_tokens)
    f_mean = precision * recall / (ALPHA * precision + (1 - ALPHA) * recall)
    penalty = GAMMA * (_chunks(matches) / matched) ** BETA
    return f_mean * (1 - penalty)


def _lcs_length(left: list[str], right: list[str]) -> int:
    """Longest common subsequence length, one DP row at a time to stay O(n) memory."""
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0] * (len(right) + 1)
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """F1 over the longest common subsequence."""
    hypothesis_tokens, reference_tokens = tokenize(prediction), tokenize(reference)
    common = _lcs_length(hypothesis_tokens, reference_tokens)
    if not common:
        return 0.0
    precision = common / len(hypothesis_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_similarity(prediction: str, reference: str) -> float:
    """METEOR, the competition's primary metric."""
    return meteor(prediction, reference)
