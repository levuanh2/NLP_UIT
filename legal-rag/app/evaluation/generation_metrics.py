"""Vietnamese answer-quality metrics."""


def answer_similarity(prediction: str, reference: str) -> float:
    from app.evaluation.scorer_compatible import meteor_score_offline

    if not prediction.strip() or not reference.strip():
        return 0.0
    return meteor_score_offline(reference, prediction)
