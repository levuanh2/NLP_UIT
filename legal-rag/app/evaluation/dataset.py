"""Evaluation dataset loading skeleton."""

from pathlib import Path

from app.domain.evaluation import EvaluationSample


class EvaluationDatasetLoader:
    def load(self, path: Path) -> list[EvaluationSample]:
        # TODO(phase-implementation):
        # Parse and validate the local Subtask 2 evaluation dataset.
        raise NotImplementedError
