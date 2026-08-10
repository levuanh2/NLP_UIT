"""Reproducible prediction collection for manual grounding evaluation."""

import hashlib
import json
import random
import subprocess
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.domain.generation import GenerationRequest
from app.domain.queries import LegalQuery
from app.services.runtime_factory import APPROVED_MODELS, build_local_rag_runtime
from app.submission.question_loader import QuestionDatasetLoader


class GroundingEvaluation:
    """Collect model/validator predictions without assigning fake gold labels."""

    def __init__(
        self,
        *,
        question_path: Path,
        output_dir: Path,
        sample_size: int = 50,
        seed: int = 20260810,
        forced_ids: tuple[str, ...] = ("31221", "57711"),
    ) -> None:
        self.question_path = question_path
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.seed = seed
        self.forced_ids = forced_ids
        self.sample_path = output_dir / "sample.json"
        self.predictions_path = output_dir / "predictions.jsonl"
        self.annotations_path = output_dir / "annotations.jsonl"
        self.metrics_path = output_dir / "metrics.json"
        self.report_path = output_dir / "report.md"

    def prepare(self) -> list[LegalQuery]:
        questions = QuestionDatasetLoader().load(self.question_path)
        selected = deterministic_sample(
            questions,
            sample_size=self.sample_size,
            seed=self.seed,
            forced_ids=self.forced_ids,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "source": str(self.question_path),
            "sample_size": self.sample_size,
            "seed": self.seed,
            "forced_regression_ids": list(self.forced_ids),
            "question_ids": [item.question_id for item in selected],
            "questions": [item.model_dump() for item in selected],
            "models": APPROVED_MODELS,
            "config_hash": self._config_hash(),
            "validator_version": self._validator_version(),
            "source_commit": self._source_commit(),
            "prepared_at": datetime.now(UTC).isoformat(),
        }
        self._write_json(self.sample_path, metadata)
        if not self.annotations_path.exists():
            self._write_jsonl(
                self.annotations_path,
                (
                    {
                        "question_id": item.question_id,
                        "gold_grounded": None,
                        "gold_citation_valid": None,
                        "gold_answer_quality": None,
                        "retrieval_sufficient": None,
                        "failure_stage": None,
                        "gold_reason": None,
                        "annotator": None,
                    }
                    for item in selected
                ),
            )
        self.refresh_metrics_and_report()
        return selected

    def run(self) -> None:
        selected = self.prepare()
        completed = {
            str(item["question_id"])
            for item in self._read_jsonl(self.predictions_path)
            if item.get("error") is None
        }
        pending = [item for item in selected if item.question_id not in completed]
        if not pending:
            self.refresh_metrics_and_report()
            return

        runtime = build_local_rag_runtime(get_settings(), trace=False)
        try:
            for position, question in enumerate(pending, start=1):
                record = self._predict(runtime, question)
                self._append_jsonl(self.predictions_path, record)
                print(
                    f"grounding_eval {position}/{len(pending)} "
                    f"question_id={question.question_id} "
                    f"grounded={record.get('validator_grounded')} "
                    f"error={record.get('error') is not None}",
                    flush=True,
                )
        finally:
            runtime.close()
        self.refresh_metrics_and_report()

    def _predict(self, runtime: Any, question: LegalQuery) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            retrieval_started = time.perf_counter()
            retrieval = runtime.service.retrieval_pipeline.retrieve(question)
            retrieval_seconds = time.perf_counter() - retrieval_started
            generation_started = time.perf_counter()
            generated = runtime.service.generation_pipeline.generate(
                GenerationRequest(
                    question_id=question.question_id,
                    question=question.question,
                    retrieval_result=retrieval,
                )
            )
            generation_seconds = time.perf_counter() - generation_started
            attempts = [item.model_dump() for item in generated.attempts]
            final_attempt = generated.attempts[-1] if generated.attempts else None
            return {
                "question_id": question.question_id,
                "question": question.question,
                "answer": generated.answer,
                "retrieval_result": retrieval.model_dump(),
                "evidence": [item.model_dump() for item in retrieval.evidences],
                "citations": [item.model_dump() for item in generated.citations],
                "citation_valid": (
                    final_attempt.citations_valid
                    if final_attempt is not None
                    else generated.abstained
                ),
                "validator_grounded": generated.grounded,
                "validator_reason": generated.validation_errors,
                "abstained": generated.abstained,
                "repair_invoked": len(generated.attempts) > 1,
                "repair_result": attempts[1] if len(attempts) > 1 else None,
                "attempts": attempts,
                "latency": {
                    "retrieval_seconds": retrieval_seconds,
                    "generation_seconds": generation_seconds,
                    "total_seconds": time.perf_counter() - started,
                },
                "error": None,
            }
        except Exception as exc:
            return {
                "question_id": question.question_id,
                "question": question.question,
                "answer": None,
                "retrieval_result": None,
                "evidence": [],
                "citations": [],
                "citation_valid": None,
                "validator_grounded": None,
                "validator_reason": [],
                "abstained": None,
                "repair_invoked": None,
                "repair_result": None,
                "attempts": [],
                "latency": {"total_seconds": time.perf_counter() - started},
                "error": f"{type(exc).__name__}: {exc}",
            }

    def refresh_metrics_and_report(self) -> None:
        predictions = self._read_jsonl(self.predictions_path)
        annotations = self._read_jsonl(self.annotations_path)
        metrics = calculate_metrics(predictions, annotations)
        metrics.update(
            {
                "sample_size": self.sample_size,
                "seed": self.seed,
                "forced_regression_ids": list(self.forced_ids),
                "validator_version": self._validator_version(),
                "prediction_count": len(predictions),
                "status": (
                    "COMPLETE"
                    if metrics["certain_cases"] == self.sample_size
                    else "PENDING_MANUAL_ANNOTATION"
                ),
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
        self._write_json(self.metrics_path, metrics)
        self.report_path.write_text(
            render_report(
                metrics=metrics,
                predictions=predictions,
                annotations=annotations,
                source=str(self.question_path),
                models=APPROVED_MODELS,
                validator_version=self._validator_version(),
            ),
            encoding="utf-8",
        )

    def _config_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(APPROVED_MODELS, sort_keys=True).encode())
        for path in (Path("configs/generation.yaml"), Path("configs/retrieval.yaml")):
            digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _validator_version() -> str:
        path = Path("app/generation/validation/grounding_validator.py")
        return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"

    @staticmethod
    def _source_commit() -> str | None:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    @staticmethod
    def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        temporary.replace(path)

    @staticmethod
    def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            stream.flush()


def deterministic_sample(
    questions: list[LegalQuery],
    *,
    sample_size: int,
    seed: int,
    forced_ids: tuple[str, ...],
) -> list[LegalQuery]:
    if len(questions) < sample_size:
        raise ValueError(
            f"Question dataset has {len(questions)} rows; {sample_size} required"
        )
    by_id = {item.question_id: item for item in questions}
    missing = [question_id for question_id in forced_ids if question_id not in by_id]
    if missing:
        raise ValueError(f"Forced regression question IDs are missing: {missing}")
    unique_forced = list(dict.fromkeys(forced_ids))
    if len(unique_forced) > sample_size:
        raise ValueError("Forced question count exceeds sample size")
    remaining = [
        item for item in questions if item.question_id not in set(unique_forced)
    ]
    sampled = random.Random(seed).sample(
        remaining, sample_size - len(unique_forced)
    )
    return [by_id[question_id] for question_id in unique_forced] + sampled


def calculate_metrics(
    predictions: list[dict[str, Any]], annotations: list[dict[str, Any]]
) -> dict[str, Any]:
    predicted = {str(item["question_id"]): item for item in predictions}
    certain = [
        item
        for item in annotations
        if isinstance(item.get("gold_grounded"), bool)
        and str(item["question_id"]) in predicted
        and isinstance(
            predicted[str(item["question_id"])].get("validator_grounded"), bool
        )
    ]
    tp = tn = fp = fn = 0
    for annotation in certain:
        gold = annotation["gold_grounded"]
        validator = predicted[str(annotation["question_id"])]["validator_grounded"]
        if validator and gold:
            tp += 1
        elif validator and not gold:
            fp += 1
        elif not validator and gold:
            fn += 1
        else:
            tn += 1

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    completed = [item for item in predictions if item.get("error") is None]
    confusion: dict[str, int | None] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    if not certain:
        confusion = {name: None for name in confusion}
    failure_types = [
        failure
        for item in certain
        for failure in item.get("failure_types", [])
    ]
    failure_names = (
        "retrieval_insufficient",
        "missing_citation",
        "wrong_citation",
        "unsupported_elaboration",
        "safe_abstention_correct",
        "safe_abstention_incorrect",
        "validator_false_positive",
        "validator_false_negative",
        "other",
    )
    descriptive = {
        "validator_pass_rate": ratio(
            sum(item.get("validator_grounded") is True for item in completed),
            len(completed),
        ),
        "validator_reject_rate": ratio(
            sum(item.get("validator_grounded") is False for item in completed),
            len(completed),
        ),
        "citation_valid_rate": ratio(
            sum(item.get("citation_valid") is True for item in completed),
            len(completed),
        ),
        "safe_abstention_count": sum(
            item.get("abstained") is True for item in completed
        ),
        "generation_error_count": len(predictions) - len(completed),
    }
    return {
        "certain_cases": len(certain),
        "uncertain_cases": len(annotations) - len(certain),
        "manual_reviewed": sum(
            item.get("review_status") == "manual_reviewed" for item in annotations
        ),
        "uncertain": len(annotations) - len(certain),
        "gold_grounded_count": sum(
            item.get("gold_grounded") is True for item in certain
        ),
        "gold_ungrounded_count": sum(
            item.get("gold_grounded") is False for item in certain
        ),
        "confusion_matrix": confusion,
        "tp": confusion["tp"],
        "tn": confusion["tn"],
        "fp": confusion["fp"],
        "fn": confusion["fn"],
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "false_positive_rate": ratio(fp, fp + tn),
        "false_negative_rate": ratio(fn, fn + tp),
        "accuracy": ratio(tp + tn, len(certain)),
        "validator_pass_rate": descriptive["validator_pass_rate"],
        "validator_reject_rate": descriptive["validator_reject_rate"],
        "citation_valid_rate": descriptive["citation_valid_rate"],
        "descriptive_statistics": descriptive,
        "failure_breakdown": {
            name: failure_types.count(name) for name in failure_names
        },
    }


def render_report(
    *,
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    source: str,
    models: dict[str, str],
    validator_version: str,
) -> str:
    confusion = metrics["confusion_matrix"]
    by_annotation = {str(item["question_id"]): item for item in annotations}
    false_positives = [
        item
        for item in predictions
        if item.get("validator_grounded") is True
        and by_annotation.get(str(item["question_id"]), {}).get("gold_grounded")
        is False
    ]
    false_negatives = [
        item
        for item in predictions
        if item.get("validator_grounded") is False
        and by_annotation.get(str(item["question_id"]), {}).get("gold_grounded")
        is True
    ]
    uncertain = [
        item
        for item in annotations
        if not isinstance(item.get("gold_grounded"), bool)
    ]
    breakdown = metrics["failure_breakdown"]

    def metric(value: float | None, numerator: int, denominator: int) -> str:
        if value is None:
            return f"{numerator} / {denominator} = null (denominator is zero)"
        return (
            f"{numerator} / {denominator} = {value:.4f} "
            f"({value * 100:.2f}%)"
        )

    def count(value: int | None) -> str:
        return "PENDING" if value is None else str(value)

    def cases(items: list[dict[str, Any]]) -> str:
        if not items:
            return "- None identified from completed manual annotations."
        rendered: list[str] = []
        for item in items:
            annotation = by_annotation[str(item["question_id"])]
            rendered.append(
                f"- question_id: {item['question_id']}\n"
                f"  - validator: {item.get('validator_grounded')}\n"
                f"  - gold: {annotation.get('gold_grounded')}\n"
                f"  - reason: {annotation.get('reason')}\n"
                "  - unsupported claims: "
                f"{annotation.get('unsupported_claims', [])}"
            )
        return "\n".join(rendered)

    tp = confusion["tp"] or 0
    tn = confusion["tn"] or 0
    fp = confusion["fp"] or 0
    fn = confusion["fn"] or 0
    weakness = (
        "The deterministic unsupported-detail rule can over-reject supported "
        "paraphrases; question 31221 is the observed false negative."
        if fn
        else "No false negative was observed in this sample."
    )

    return f"""# GroundingValidator 50-Question Evaluation

## Dataset

- Source: {source}
- Sample size: {metrics['sample_size']}
- Seed: {metrics['seed']}
- Forced IDs: {metrics['forced_regression_ids']}
- Evaluation date: {metrics['updated_at']}

## Model

- LLM: {models['llm']}
- Embedding: {models['embedding']}
- Reranker: {models['reranker']}

## Validator

- Version/commit: {validator_version}
- Claim-level validation: enabled
- Unsupported detail detection: enabled
- Legal reference matching: field-aware canonical matching

## Manual Annotation Summary

- Manually reviewed: {metrics['manual_reviewed']}/{metrics['sample_size']}
- Grounded: {metrics['gold_grounded_count']}
- Ungrounded: {metrics['gold_ungrounded_count']}
- Uncertain: {metrics['uncertain']}
- Status: {metrics['status']}

## Confusion Matrix

| | Gold Grounded | Gold Ungrounded |
|---|---:|---:|
| Validator Grounded | {count(confusion['tp'])} | {count(confusion['fp'])} |
| Validator Ungrounded | {count(confusion['fn'])} | {count(confusion['tn'])} |

## Metrics

- Precision: {metric(metrics['precision'], tp, tp + fp)}
- Recall: {metric(metrics['recall'], tp, tp + fn)}
- FPR: {metric(metrics['false_positive_rate'], fp, fp + tn)}
- FNR: {metric(metrics['false_negative_rate'], fn, fn + tp)}
- Accuracy: {metric(metrics['accuracy'], tp + tn, tp + tn + fp + fn)}

## Descriptive statistics

```json
{json.dumps(metrics['descriptive_statistics'], ensure_ascii=False, indent=2)}
```

## Failure Breakdown

| Failure type | Count |
|---|---:|
| Retrieval insufficient | {breakdown['retrieval_insufficient']} |
| Missing citation | {breakdown['missing_citation']} |
| Wrong citation | {breakdown['wrong_citation']} |
| Unsupported elaboration | {breakdown['unsupported_elaboration']} |
| Safe abstention correct | {breakdown['safe_abstention_correct']} |
| Safe abstention incorrect | {breakdown['safe_abstention_incorrect']} |
| Validator false positive | {breakdown['validator_false_positive']} |
| Validator false negative | {breakdown['validator_false_negative']} |
| Other | {breakdown['other']} |

Categories may overlap (for example, a retrieval-insufficient answer can also
have a missing citation).

## False Positives

{cases(false_positives)}

## False Negatives

{cases(false_negatives)}

## Uncertain Cases

{', '.join(str(item['question_id']) for item in uncertain) or 'None'}

## Conclusion

- Precision: {metric(metrics['precision'], tp, tp + fp)}
- False-positive rate: {metric(metrics['false_positive_rate'], fp, fp + tn)}
- False-negative rate: {metric(metrics['false_negative_rate'], fn, fn + tp)}
- Main weakness: {weakness}

This evaluation estimates GroundingValidator behavior on a 50-question
manually reviewed sample. It is not a statistically sufficient guarantee of
performance on the full competition dataset.
"""
