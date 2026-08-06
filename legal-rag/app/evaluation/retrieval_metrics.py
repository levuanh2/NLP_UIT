"""Retrieval metric skeletons."""


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    # TODO(phase-implementation):
    # Define edge-case policy and compute recall@k.
    raise NotImplementedError


def mean_reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    # TODO(phase-implementation):
    # Compute reciprocal rank for the first relevant result.
    raise NotImplementedError
