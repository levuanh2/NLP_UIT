"""Retrieval metrics."""


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0 or not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def mean_reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    for rank, identifier in enumerate(ranked_ids, start=1):
        if identifier in relevant_ids:
            return 1.0 / rank
    return 0.0
