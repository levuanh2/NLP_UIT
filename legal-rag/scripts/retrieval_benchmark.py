"""Benchmark the four retrieval stages against each other, without the generator.

Answers one question: which retrieval stage actually improves evidence coverage,
and what does the reranker cost for what it returns. Generation is never called.

Gold labels: the competition ships no retrieval ground truth — no gold child ids,
no gold document ids. What exists is 7000 expert answers in train.json, and those
answers cite the law by number ("17/2022/TT-BVHTTDL", "Điều 25"). Those citations
are the gold used here, so every metric below is *legal-reference* recall, not
evidence-id recall. Evidence-id recall is reported as N/A because inventing the
labels would be worse than not having them.

Usage:
  MODEL_DEVICE=cpu python scripts/retrieval_benchmark.py
  MODEL_DEVICE=cpu python scripts/retrieval_benchmark.py --limit 20 --out-dir ...

MODEL_DEVICE=cpu keeps the unused LLM off the GPU without touching the factory.
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.domain.retrieval import RetrievalCandidate
from app.services.runtime_factory import build_local_rag_runtime

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = re.compile(r"(\d+/\d{4}/[A-ZĐ][A-ZĐ-]*|\d+-\d{4}-[A-ZĐ][A-ZĐ-]*)")
ARTICLE = re.compile(r"Điều\s+(\d+)")
RECALL_CUTOFFS = (1, 3, 5, 10, 20)
WARMUP_QUERIES = 2
SMOKE_VERSIONS = {"smoke-v2", "smoke-test", "smoke-test-v2"}


class BenchmarkError(RuntimeError):
    """Anything that makes the measurement untrustworthy. Never swallowed."""


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #


def read_index_state(index_root: Path) -> dict[str, Any]:
    """Resolve CURRENT and its manifest, refusing anything not fully built."""
    current_file = index_root / "CURRENT"
    if not current_file.is_file():
        raise BenchmarkError(f"BLOCKED: no CURRENT marker under {index_root}")
    version = current_file.read_text(encoding="utf-8").strip()
    if not version:
        raise BenchmarkError("BLOCKED: CURRENT is empty")
    if version in SMOKE_VERSIONS:
        raise BenchmarkError(
            f"BLOCKED: CURRENT is the smoke index ({version}); refusing to benchmark it"
        )
    resolved = index_root / version
    manifest_path = resolved / "manifest.json"
    if not manifest_path.is_file():
        raise BenchmarkError(f"BLOCKED: no manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ready":
        raise BenchmarkError(
            f"BLOCKED: manifest status is {manifest.get('status')!r}, not 'ready'"
        )

    faiss = sorted((resolved / "faiss").glob("*.index"))
    bm25 = sorted((resolved / "bm25").glob("*"))
    sqlite = sorted((resolved / "metadata").glob("*.sqlite"))
    if not faiss:
        raise BenchmarkError(f"BLOCKED: no FAISS shard under {resolved / 'faiss'}")
    if not bm25:
        raise BenchmarkError(f"BLOCKED: no BM25 artifact under {resolved / 'bm25'}")
    if not sqlite:
        raise BenchmarkError(f"BLOCKED: no SQLite metadata under {resolved/'metadata'}")
    for field_name in ("document_count", "parent_count", "child_count"):
        if not manifest.get(field_name):
            raise BenchmarkError(f"BLOCKED: manifest is missing {field_name}")

    return {
        "current": version,
        "resolved_path": str(resolved),
        "manifest": manifest,
        "faiss_shards": [p.name for p in faiss],
        "bm25_artifacts": [p.name for p in bm25],
        "sqlite_files": [p.name for p in sqlite],
    }


# SQLite in WAL mode rewrites these two sidecars on a read-only connection, so
# they move whatever the benchmark does. The durable database file next to them
# is the thing that must not change, and it is still checked.
TRANSIENT_SUFFIXES = ("-shm", "-wal", "-journal")


def fingerprint(paths: Iterable[Path]) -> dict[str, list[float]]:
    """Size and mtime of everything the benchmark must not touch."""
    marks: dict[str, list[float]] = {}

    def mark(path: Path) -> None:
        if path.name.endswith(TRANSIENT_SUFFIXES):
            return
        stat = path.stat()
        marks[str(path)] = [stat.st_size, stat.st_mtime]

    for path in paths:
        if path.is_file():
            mark(path)
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    mark(child)
    return marks


def environment() -> dict[str, Any]:
    import torch
    import transformers

    cuda = torch.cuda.is_available()
    return {
        "cuda_available": cuda,
        "gpu_count": torch.cuda.device_count() if cuda else 0,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "cuda_device": torch.cuda.current_device() if cuda else None,
        "cuda_version": torch.version.cuda if cuda else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
    }


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError:
        return None
    return out.stdout.strip() or None


# --------------------------------------------------------------------------- #
# gold labels
# --------------------------------------------------------------------------- #


@dataclass
class Question:
    question_id: str
    question: str
    gold_documents: set[str]
    gold_articles: set[str]


def load_questions(
    questions_path: Path, train_path: Path, limit: int | None
) -> list[Question]:
    """Questions that have an expert answer, so gold citations can be parsed."""
    raw = json.loads(questions_path.read_text(encoding="utf-8"))
    train = json.loads(train_path.read_text(encoding="utf-8"))
    chosen: list[Question] = []
    for question_id in raw:
        record = train.get(question_id)
        if record is None:
            continue
        answer = record["answer"]
        chosen.append(
            Question(
                question_id=question_id,
                question=record["question"],
                gold_documents=set(DOCUMENT.findall(answer)),
                gold_articles=set(ARTICLE.findall(answer)),
            )
        )
    if not chosen:
        raise BenchmarkError(
            f"BLOCKED: none of {questions_path.name} has an expert answer in "
            f"{train_path.name}; there is no gold to score against"
        )
    without_gold = [q for q in chosen if not q.gold_documents and not q.gold_articles]
    if len(without_gold) == len(chosen):
        raise BenchmarkError("BLOCKED: no question carries a parseable legal reference")
    return chosen[:limit] if limit else chosen


# --------------------------------------------------------------------------- #
# stage runners
# --------------------------------------------------------------------------- #


@dataclass
class StageTiming:
    """Latency of one configuration on one question, split by sub-stage."""

    parts: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return sum(self.parts.values())


class Clock:
    """Monotonic timing that waits for the GPU before it reads the clock."""

    def __init__(self) -> None:
        import torch

        self.torch = torch
        self.cuda = torch.cuda.is_available()

    def sync(self) -> None:
        if self.cuda:
            self.torch.cuda.synchronize()

    def time(self, work: Callable[[], Any]) -> tuple[Any, float]:
        self.sync()
        started = time.perf_counter()
        result = work()
        self.sync()
        return result, (time.perf_counter() - started) * 1000.0


def resolve_documents(
    repository: Any, candidates: list[RetrievalCandidate]
) -> list[dict[str, Any]]:
    """Attach the text and document name a citation match needs."""
    resolved = []
    for candidate in candidates:
        child = repository.get_child(candidate.child_id)
        if child is None:
            raise BenchmarkError(
                f"malformed retrieval: child_id {candidate.child_id!r} is not in SQLite"
            )
        resolved.append(
            {
                "child_id": candidate.child_id,
                "document_id": child.document_id,
                "document_name": child.document_name or "",
                "article": child.article or "",
                "text": child.text,
            }
        )
    return resolved


def hits(item: dict[str, Any], question: Question) -> tuple[set[str], set[str]]:
    """Which gold citations this one retrieved chunk actually contains."""
    haystack = f"{item['document_name']}\n{item['article']}\n{item['text']}"
    documents = question.gold_documents & set(DOCUMENT.findall(haystack))
    articles = question.gold_articles & set(ARTICLE.findall(haystack))
    return documents, articles




def recall_at(
    resolved: list[dict[str, Any]], question: Question, cutoff: int, kind: str
) -> float | None:
    """Share of gold citations covered by the first `cutoff` retrieved chunks."""
    gold = question.gold_documents if kind == "document" else question.gold_articles
    if not gold:
        return None
    found: set[str] = set()
    for item in resolved[:cutoff]:
        documents, articles = hits(item, question)
        found |= documents if kind == "document" else articles
    return len(found) / len(gold)


def first_hit_rank(resolved: list[dict[str, Any]], question: Question) -> int | None:
    """1-based rank of the first chunk carrying any gold citation."""
    for position, item in enumerate(resolved, 1):
        if any(hits(item, question)):
            return position
    return None


# --------------------------------------------------------------------------- #
# the four configurations
# --------------------------------------------------------------------------- #


@dataclass
class Observation:
    """One configuration on one question: what came back and how long it took."""

    candidates: list[RetrievalCandidate]
    parts: dict[str, float]

    @property
    def total_ms(self) -> float:
        return sum(self.parts.values())


def filter_decision(pipeline: Any, question: str) -> dict[str, Any]:
    """What the production metadata pre-filter would do with this question.

    Recorded rather than assumed: the filter used to fire on a phrase lifted out
    of prose and match nothing, and the only way to show that is fixed is to
    count applied / ambiguous / empty separately instead of collapsing them into
    "not applied".
    """
    metadata = pipeline.query_analyzer.analyze(question)
    result = pipeline.metadata_filter.build_filter(metadata)
    return {
        "document_number": metadata.document_number,
        "document_name": metadata.document_name,
        "confidence": metadata.confidence,
        "fields": list(result.fields),
        "applied": result.applied,
        "ambiguous": result.ambiguous,
        "empty_lookup": result.empty_lookup,
        "candidate_count": result.matched_count,
        "candidate_ids": result.candidate_ids,
    }


def observe(
    pipeline: Any,
    clock: Clock,
    question: str,
    config: str,
    candidate_ids: set[str] | None = None,
) -> tuple[Observation, list[RetrievalCandidate]]:
    """Run one configuration end to end, timing each sub-stage it actually uses.

    Every configuration re-runs the stages it depends on rather than reusing an
    earlier run's output, so each reported latency is the real cost of choosing
    that configuration in production. The stages are deterministic, so this
    changes only the timing, never the ranking.
    """
    parts: dict[str, float] = {}
    dense: list[RetrievalCandidate] = []
    bm25: list[RetrievalCandidate] = []
    fused: list[RetrievalCandidate] = []

    if config in {"dense", "rrf", "rrf_reranker"}:
        dense, ms = clock.time(
            lambda: pipeline.dense_retriever.retrieve(
                question, candidate_ids=candidate_ids, top_k=pipeline.dense_top_k
            )
        )
        parts["dense_ms"] = ms
    if config in {"bm25", "rrf", "rrf_reranker"}:
        bm25, ms = clock.time(
            lambda: pipeline.bm25_retriever.retrieve(
                question, candidate_ids=candidate_ids, top_k=pipeline.bm25_top_k
            )
        )
        parts["bm25_ms"] = ms
    if config in {"rrf", "rrf_reranker"}:
        fused, ms = clock.time(
            lambda: pipeline.fusion.fuse(
                dense, bm25, k=pipeline.rrf_k, top_k=pipeline.rrf_top_k
            )
        )
        parts["rrf_merge_ms"] = ms

    if config == "dense":
        return Observation(dense, parts), []
    if config == "bm25":
        return Observation(bm25, parts), []
    if config == "rrf":
        return Observation(fused, parts), []

    reranked, ms = clock.time(
        lambda: pipeline.reranker.rerank(
            question, fused, top_k=pipeline.reranker_top_k
        )
    )
    parts["reranker_ms"] = ms
    return Observation(reranked, parts), fused


CONFIGS = ("dense", "bm25", "rrf", "rrf_reranker")


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #


def summarise(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "min": None, "max": None}
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "mean": round(statistics.mean(ordered), 2),
        "median": round(statistics.median(ordered), 2),
        "p95": round(ordered[index], 2),
        "min": round(ordered[0], 2),
        "max": round(ordered[-1], 2),
    }


def mean_or_none(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def aggregate(rows: list[dict[str, Any]], config: str) -> dict[str, Any]:
    """Roll per-question records for one configuration into the reported metrics."""
    latencies = [row[config]["total_ms"] for row in rows]
    metrics: dict[str, Any] = {
        "questions": len(rows),
        "evidence_id_recall": "N/A: the task ships no gold evidence ids",
    }
    for cutoff in RECALL_CUTOFFS:
        for kind in ("document", "article"):
            scores = [
                row[config][f"recall_{kind}@{cutoff}"]
                for row in rows
                if row[config][f"recall_{kind}@{cutoff}"] is not None
            ]
            metrics[f"recall_{kind}@{cutoff}"] = mean_or_none(scores)
    reciprocal = [
        1.0 / row[config]["first_hit_rank"] if row[config]["first_hit_rank"] else 0.0
        for row in rows
    ]
    metrics["mrr"] = mean_or_none(reciprocal)
    metrics["returned_median"] = statistics.median(
        [row[config]["returned"] for row in rows]
    )
    hit_flags = [row[config]["any_hit"] for row in rows]
    metrics["questions_with_any_gold_hit"] = round(
        sum(hit_flags) / len(hit_flags), 4
    )
    full = [row[config]["full_document_coverage"] for row in rows]
    metrics["questions_with_all_gold_documents"] = round(sum(full) / len(full), 4)
    document_scores = [
        row[config]["document_coverage"]
        for row in rows
        if row[config]["document_coverage"] is not None
    ]
    metrics["document_coverage"] = mean_or_none(document_scores)
    metrics["latency_ms"] = summarise(latencies)
    stage_names: set[str] = set()
    for row in rows:
        stage_names |= set(row[config]["parts"])
    metrics["stage_latency_ms"] = {
        name: summarise([row[config]["parts"].get(name, 0.0) for row in rows])
        for name in sorted(stage_names)
    }
    return metrics


def gain(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    """Difference on the metrics that decide whether a stage earns its place."""
    keys = [f"recall_document@{k}" for k in RECALL_CUTOFFS]
    keys += [f"recall_article@{k}" for k in RECALL_CUTOFFS]
    keys += ["mrr", "document_coverage"]
    out: dict[str, Any] = {}
    for key in keys:
        left, right = after.get(key), before.get(key)
        both = left is not None and right is not None
        out[key] = round(left - right, 4) if both else None
    return out


def classify_failure(row: dict[str, Any]) -> str:
    """Name the stage that lost the evidence, or admit the trace cannot say."""
    dense_hit = row["dense"]["any_hit"]
    bm25_hit = row["bm25"]["any_hit"]
    rrf_hit = row["rrf"]["any_hit"]
    fused_hit = row["rrf_reranker"]["fused_any_hit"]
    final_hit = row["rrf_reranker"]["any_hit"]
    if final_hit:
        return "none"
    if not dense_hit and not bm25_hit:
        return "both_miss"
    if not dense_hit and bm25_hit and not rrf_hit:
        return "rrf_ranking"
    if dense_hit and not bm25_hit and not rrf_hit:
        return "rrf_ranking"
    if fused_hit and not final_hit:
        return "reranker_ranking"
    if not rrf_hit and (dense_hit or bm25_hit):
        return "insufficient_top_k"
    return "PENDING_MANUAL_REVIEW"


def build_row(
    question: Question,
    observations: dict[str, Observation],
    fused: list[RetrievalCandidate],
    repository: Any,
    rerank_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Everything measured for one question, across all four configurations."""
    row: dict[str, Any] = {
        "question_id": question.question_id,
        "gold_documents": sorted(question.gold_documents),
        "gold_articles": sorted(question.gold_articles),
    }
    resolved_fused = resolve_documents(repository, fused)
    # The reranker scores the whole fused pool, so gold lookups for dropped
    # candidates need the pool resolved, not just the returned top-k.
    resolved_by_id = {item["child_id"]: item for item in resolved_fused}
    for config, observation in observations.items():
        resolved = resolve_documents(repository, observation.candidates)
        entry: dict[str, Any] = {
            "returned": len(resolved),
            "parts": {name: round(ms, 3) for name, ms in observation.parts.items()},
            "total_ms": round(observation.total_ms, 3),
            "child_ids": [item["child_id"] for item in resolved],
        }
        # E5 instrumentation: persist the scores the pipeline already computed
        # instead of leaving only ids behind. Nothing here feeds back into
        # ranking; it is written after the candidates are fixed.
        entry["candidates"] = [
            {
                "child_id": candidate.child_id,
                "rank": candidate.rank or position,
                "score": candidate.score,
                "dense_score": candidate.dense_score,
                "bm25_score": candidate.bm25_score,
                "fusion_score": candidate.fusion_score,
                "rerank_score": candidate.rerank_score,
                "is_gold": bool(any(hits(item, question))),
            }
            for position, (candidate, item) in enumerate(
                zip(observation.candidates, resolved, strict=True), start=1
            )
        ]
        for cutoff in RECALL_CUTOFFS:
            for kind in ("document", "article"):
                entry[f"recall_{kind}@{cutoff}"] = recall_at(
                    resolved, question, cutoff, kind
                )
        rank = first_hit_rank(resolved, question)
        entry["first_hit_rank"] = rank
        entry["any_hit"] = rank is not None
        document_score = recall_at(resolved, question, len(resolved) or 1, "document")
        entry["document_coverage"] = document_score
        entry["full_document_coverage"] = bool(
            document_score is not None and document_score >= 1.0
        )
        if config == "rrf_reranker":
            fused_rank = first_hit_rank(resolved_fused, question)
            entry["fused_first_hit_rank"] = fused_rank
            entry["fused_any_hit"] = fused_rank is not None
            entry["rank_change"] = (
                fused_rank - rank if fused_rank and rank else None
            )
            # E5: the reranker scores every fused candidate and then returns only
            # top_k, so the score of a dropped gold chunk is invisible in the
            # returned list. Read it from the full table the reranker published.
            scores = rerank_scores or {}
            ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
            entry["rerank_scored_count"] = len(ordered)
            entry["rerank_top1_score"] = ordered[0][1] if ordered else None
            entry["rerank_top2_score"] = ordered[1][1] if len(ordered) > 1 else None
            entry["rerank_margin_top1_top2"] = (
                ordered[0][1] - ordered[1][1] if len(ordered) > 1 else None
            )
            gold_rows = [
                (position, child_id, value)
                for position, (child_id, value) in enumerate(ordered, start=1)
                if child_id in resolved_by_id
                and any(hits(resolved_by_id[child_id], question))
            ]
            entry["gold_in_rerank_pool"] = bool(gold_rows)
            if gold_rows:
                position, child_id, value = gold_rows[0]
                entry["gold_best_rerank_rank"] = position
                entry["gold_best_rerank_score"] = value
                entry["gold_best_child_id"] = child_id
                entry["gold_score_gap_to_top1"] = ordered[0][1] - value
                # What outranked it, and by how little.
                entry["blockers_above_gold"] = [
                    {"child_id": cid, "score": val, "rank": idx}
                    for idx, (cid, val) in enumerate(ordered[: position - 1], start=1)
                ][-3:]
            else:
                entry["gold_best_rerank_rank"] = None
                entry["gold_best_rerank_score"] = None
                entry["gold_best_child_id"] = None
                entry["gold_score_gap_to_top1"] = None
                entry["blockers_above_gold"] = []
        row[config] = entry
    row["failure_stage"] = classify_failure(row)
    return row


def reranker_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Did reranking move gold evidence up, down, or out of the returned list?"""
    improved, unchanged, worsened, lost = [], [], [], []
    for row in rows:
        entry = row["rrf_reranker"]
        before, after = entry.get("fused_first_hit_rank"), entry.get("first_hit_rank")
        if before is None and after is None:
            unchanged.append(row["question_id"])
        elif before is not None and after is None:
            lost.append(row["question_id"])
        elif before is None and after is not None:
            improved.append(row["question_id"])
        elif after < before:
            improved.append(row["question_id"])
        elif after > before:
            worsened.append(row["question_id"])
        else:
            unchanged.append(row["question_id"])
    return {
        "improved": improved,
        "unchanged": unchanged,
        "worsened": worsened,
        "gold_dropped_from_top_k": lost,
        "counts": {
            "improved": len(improved),
            "unchanged": len(unchanged),
            "worsened": len(worsened),
            "gold_dropped_from_top_k": len(lost),
        },
    }


def recommend(metrics: dict[str, Any], verdict: dict[str, Any]) -> tuple[str, str]:
    """Turn the measured reranker effect into one of the four allowed verdicts."""
    gains = metrics["reranker_gain_vs_rrf"]
    at5 = gains.get("recall_document@5")
    overhead = metrics["reranker_latency_overhead_ms"]
    counts = verdict["counts"]
    if at5 is None:
        return "NEED MORE EVIDENCE", "Recall could not be computed on this set."
    if at5 > 0.01 and counts["improved"] > counts["worsened"]:
        return (
            "KEEP RERANKER",
            f"Recall@5 rises {at5:+.4f} and gold is promoted on "
            f"{counts['improved']} questions against {counts['worsened']} demoted, "
            f"for {overhead:.0f} ms a question.",
        )
    if at5 < -0.01 or counts["worsened"] > max(1, counts["improved"]) * 2:
        return (
            "REMOVE RERANKER",
            f"Recall@5 moves {at5:+.4f} while gold is demoted on "
            f"{counts['worsened']} questions against {counts['improved']} promoted, "
            f"and it costs {overhead:.0f} ms a question.",
        )
    if abs(at5) <= 0.01 and overhead > 200:
        return (
            "RERANKER ONLY AT LARGER CANDIDATE K",
            f"Recall@5 moves only {at5:+.4f} for {overhead:.0f} ms a question. At "
            "this candidate depth its ranking is indistinguishable from RRF; it can "
            "only pay for itself if it is given more candidates to choose from.",
        )
    return (
        "NEED MORE EVIDENCE",
        f"Recall@5 moves {at5:+.4f}, inside the noise at this question count.",
    )


def cell(value: Any) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def signed(value: Any) -> str:
    return "N/A" if value is None else f"{value:+.4f}"


def latency_row(label: str, metrics: dict[str, Any]) -> str:
    stats = metrics["latency_ms"]
    return (
        f"| {label} | {stats['mean']} | {stats['median']} | {stats['p95']} | "
        f"{stats['min']} | {stats['max']} |"
    )


def coverage_row(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {label} | {cell(metrics['document_coverage'])} | "
        f"{cell(metrics['questions_with_any_gold_hit'])} | "
        f"{cell(metrics['questions_with_all_gold_documents'])} |"
    )


LABELS = (
    ("dense", "Dense"),
    ("bm25", "BM25"),
    ("rrf", "RRF"),
    ("rrf_reranker", "RRF+Reranker"),
)


def render_report(payload: dict[str, Any]) -> str:
    """The human-readable half of the artifacts."""
    metrics = payload["metrics"]
    config = payload["config"]
    verdict = payload["reranker"]
    rows = payload["rows"]
    index = config["index"]
    manifest = index["manifest"]
    env = config["environment"]
    devices = config["devices"]
    ks = config["k_values"]
    gains = metrics["reranker_gain_vs_rrf"]
    overhead = metrics["reranker_latency_overhead_ms"]
    recommendation, because = payload["recommendation"]

    main_table = "\n".join(
        f"| {label} | {cell(metrics[name]['recall_document@5'])} | "
        f"{cell(metrics[name]['recall_document@10'])} | "
        f"{cell(metrics[name]['recall_document@20'])} | "
        f"{cell(metrics[name]['document_coverage'])} | "
        f"{cell(metrics[name]['mrr'])} | "
        f"{metrics[name]['latency_ms']['mean']} | "
        f"{metrics[name]['latency_ms']['p95']} |"
        for name, label in LABELS
    )
    latency_table = "\n".join(
        latency_row(label, metrics[name]) for name, label in LABELS
    )
    coverage_table = "\n".join(
        coverage_row(label, metrics[name]) for name, label in LABELS
    )

    failures = [r["question_id"] for r in rows if not r["rrf_reranker"]["any_hit"]]
    by_stage: dict[str, int] = {}
    for row in rows:
        if row["failure_stage"] != "none":
            by_stage[row["failure_stage"]] = by_stage.get(row["failure_stage"], 0) + 1

    parts = []
    parts.append(
        "# Retrieval Benchmark\n\n"
        "Retrieval only. The generator is never called, and nothing under `storage/` "
        "was written.\n"
    )
    parts.append(
        "## Environment\n\n"
        f"- GPU: {env['gpu_name']} x{env['gpu_count']}, CUDA {env['cuda_version']}\n"
        f"- torch {env['torch']}, transformers {env['transformers']}, "
        f"python {env['python']}\n"
        f"- embedding device `{devices['embedding']}`, reranker device "
        f"`{devices['reranker']}`, dtype `{devices['dtype']}`\n"
        f"- LLM device `{devices['llm']}` — built by the shared factory, never called\n"
    )
    parts.append(
        "## Index\n\n"
        f"- CURRENT `{index['current']}` at `{index['resolved_path']}`, manifest "
        f"status `{manifest['status']}`\n"
        f"- {manifest['document_count']} documents "
        f"({manifest['valid_document_count']} valid), "
        f"{manifest['parent_count']} parents, {manifest['child_count']} children\n"
        f"- embedding dimension {manifest['embedding_dimension']}, FAISS "
        f"`{manifest['faiss_index_type']}`, {len(index['faiss_shards'])} shard(s)\n"
        f"- BM25: {', '.join(index['bm25_artifacts'])}\n"
        f"- SQLite: {', '.join(index['sqlite_files'])}\n"
    )
    parts.append(
        "## Benchmark Questions\n\n"
        f"{config['question_count']} questions from `{config['questions_file']}`, kept "
        "only where `train.json` carries an expert answer to parse gold citations "
        "from.\n\n"
        "**Gold labels here are legal references, not evidence ids.** The task ships "
        "no retrieval ground truth: no gold child ids, no gold document ids. Document "
        "numbers (`17/2022/TT-BVHTTDL`) and article numbers (`Điều 25`) are parsed out "
        "of the expert answer, and a retrieved chunk counts as a hit when the same "
        "reference appears in its text, document name, or article field. Child-level "
        "recall and document-id recall are **N/A** — those labels do not exist, and "
        "inventing them would make every number here meaningless.\n\n"
        "The 15-question retrieval qualification set named in the task brief does not "
        "exist in this repository or on the server; the 50-question reviewed sample in "
        "`data/evaluation/grounding_50/` draws from the blind test set, so it carries "
        "no expert answer and no gold of any kind. Question ids used are listed in "
        "`config.json`.\n"
    )
    parts.append(
        "## Configuration\n\n"
        "Read from the running production settings, unchanged:\n\n"
        f"- dense top-k {ks['dense_top_k']}, BM25 top-k {ks['bm25_top_k']}\n"
        f"- RRF k {ks['rrf_k']}, RRF top-k {ks['rrf_top_k']}\n"
        f"- reranker top-k {ks['reranker_top_k']}, enabled {ks['reranker_enabled']}\n"
        f"- embedding `{config['models']['embedding']}`\n"
        f"- reranker `{config['models']['reranker']}`\n"
        "- the metadata pre-filter is bypassed for all four configurations so each "
        "sees the same candidate pool; production leaves it on.\n"
    )
    parts.append(
        "## Results\n\n"
        "| System | Recall@5 | Recall@10 | Recall@20 | Doc coverage | MRR | Mean ms | "
        "P95 ms |\n|---|---|---|---|---|---|---|---|\n" + main_table + "\n\n"
        "Recall is document-level legal-reference recall. RRF+Reranker returns only "
        f"{ks['reranker_top_k']} items, so its Recall@20 cannot exceed its "
        f"Recall@{ks['reranker_top_k']}: that column is capped by the configuration, "
        "not by a failure.\n\n"
        "### Gains\n\n"
        f"- RRF over Dense: "
        f"`{json.dumps(metrics['rrf_gain_vs_dense'], ensure_ascii=False)}`\n"
        f"- RRF over BM25: "
        f"`{json.dumps(metrics['rrf_gain_vs_bm25'], ensure_ascii=False)}`\n"
        f"- Reranker over RRF: `{json.dumps(gains, ensure_ascii=False)}`\n"
    )
    parts.append(
        "## Latency\n\n"
        "| System | Mean | Median | P95 | Min | Max |\n|---|---|---|---|---|---|\n"
        + latency_table
        + "\n\nMilliseconds, `perf_counter` with a CUDA synchronise on both sides of "
        "every measured call.\n"
    )
    parts.append(
        "## Coverage\n\n"
        "| System | Doc coverage | Questions with >=1 gold doc | Questions with all "
        "gold docs |\n|---|---|---|---|\n" + coverage_table + "\n"
    )
    parts.append(
        "## Reranker Impact\n\n"
        f"1. **Recall@5?** {signed(gains['recall_document@5'])}\n"
        f"2. **Recall@10?** {signed(gains['recall_document@10'])}\n"
        f"3. **Recall@20?** {signed(gains['recall_document@20'])} (capped, see above)\n"
        f"4. **Latency added:** {overhead:.1f} ms a question\n"
        f"5. **Does it ever make retrieval worse?** Gold is demoted on "
        f"{verdict['counts']['worsened']} questions, and leaves the returned list "
        f"entirely on {verdict['counts']['gold_dropped_from_top_k']}.\n\n"
        f"- promoted gold: {', '.join(verdict['improved'][:20]) or 'none'}\n"
        f"- demoted gold: {', '.join(verdict['worsened'][:20]) or 'none'}\n"
        f"- dropped gold out of top-k: "
        f"{', '.join(verdict['gold_dropped_from_top_k'][:20]) or 'none'}\n"
    )
    parts.append(
        "## Failure Analysis\n\n"
        f"{len(failures)} of {config['question_count']} questions end with no gold "
        "reference anywhere in the final RRF+Reranker output.\n\n"
        f"```json\n{json.dumps(by_stage, ensure_ascii=False, indent=2)}\n```\n\n"
        f"Failed question ids: {', '.join(failures) or 'none'}\n\n"
        "Per-question detail, including per-stage ranks and latencies, is in "
        "`per_question.jsonl`.\n"
    )
    parts.append(f"## Recommendation\n\n**{recommendation}**\n\n{because}\n")
    return "\n".join(parts)


README = """# retrieval_benchmark

Written by `scripts/retrieval_benchmark.py`. Retrieval only — the LLM is never
called, and the benchmark refuses to start unless CURRENT is a ready, complete
index.

- `config.json` — environment, index state, models, k values, question ids
- `per_question.jsonl` — one record per question with all four configurations
- `metrics.json` — aggregated recall, MRR, coverage, latency, and the gains
- `report.md` — the readable write-up and the recommendation

Gold labels are legal references parsed from expert answers in `train.json`, not
evidence ids: the task ships no retrieval ground truth. Evidence-id and
document-id recall are reported as N/A rather than invented.

Re-running writes a fresh directory; existing artifacts are never overwritten.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions", type=Path, default=ROOT / "data/questions/dev200.json"
    )
    parser.add_argument("--train", type=Path, default=ROOT / "data/train/train.json")
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "data/evaluation/retrieval_benchmark"
    )
    parser.add_argument(
        "--limit", type=int, help="Benchmark only the first N questions."
    )
    args = parser.parse_args()

    settings = get_settings()
    index_root = Path(settings.index_root_dir)
    if not index_root.is_absolute():
        index_root = (ROOT / index_root).resolve()
    index = read_index_state(index_root)

    questions = load_questions(args.questions, args.train, args.limit)
    if len(questions) < 15:
        raise BenchmarkError(
            f"BLOCKED: only {len(questions)} usable questions; the brief requires 15"
        )

    print(f"INDEX: CURRENT={index['current']} status={index['manifest']['status']}")
    print(f"       {index['resolved_path']}")
    print(
        f"       docs={index['manifest']['document_count']} "
        f"parents={index['manifest']['parent_count']} "
        f"children={index['manifest']['child_count']} "
        f"dim={index['manifest']['embedding_dimension']}"
    )
    print(
        f"       faiss={len(index['faiss_shards'])} shard(s), "
        f"bm25={index['bm25_artifacts']}"
    )
    print(f"QUESTIONS: {len(questions)} from {args.questions.name}")

    env = environment()
    print(
        f"DEVICE: cuda={env['cuda_available']} {env['gpu_name']} "
        f"torch={env['torch']} transformers={env['transformers']}"
    )

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        args.out_dir = args.out_dir.parent / f"{args.out_dir.name}-{stamp}"
        print(f"NOTE: previous artifacts kept; writing to {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    guarded = fingerprint(
        [
            index_root / "CURRENT",
            Path(index["resolved_path"]),
            ROOT / ".env",
            args.questions,
            args.train,
        ]
    )

    runtime = build_local_rag_runtime(settings)
    pipeline = runtime.service.retrieval_pipeline
    repository = pipeline.dense_retriever.repository
    clock = Clock()
    rows: list[dict[str, Any]] = []
    try:
        for config in CONFIGS:
            for question in questions[:WARMUP_QUERIES]:
                observe(pipeline, clock, question.question, config, None)
        print(f"warmup done ({WARMUP_QUERIES} queries per configuration)")

        for position, question in enumerate(questions, 1):
            observations: dict[str, Observation] = {}
            fused: list[RetrievalCandidate] = []
            rerank_scores: dict[str, float] = {}
            decision = filter_decision(pipeline, question.question)
            # Production narrows retrieval only when the filter applies; every
            # configuration sees the same candidate pool so the comparison is
            # about ranking, not about who got filtered.
            candidate_ids = (
                decision.pop("candidate_ids") if decision["applied"] else None
            )
            for config in CONFIGS:
                observation, fused_from_rerank = observe(
                    pipeline, clock, question.question, config, candidate_ids
                )
                observations[config] = observation
                if config == "rrf_reranker":
                    fused = fused_from_rerank
                    # Read the score table right after the call that produced it,
                    # rather than assuming rrf_reranker is the last config.
                    rerank_scores = dict(
                        getattr(pipeline.reranker, "last_scores", None) or {}
                    )
            decision.pop("candidate_ids", None)
            row = build_row(question, observations, fused, repository, rerank_scores)
            row["metadata_filter"] = decision
            rows.append(row)
            if position % 25 == 0:
                print(f"  {position}/{len(questions)}", flush=True)
    finally:
        runtime.close()

    after = fingerprint(
        [
            index_root / "CURRENT",
            Path(index["resolved_path"]),
            ROOT / ".env",
            args.questions,
            args.train,
        ]
    )
    if after != guarded:
        changed = [k for k in after if guarded.get(k) != after[k]]
        raise BenchmarkError(
            f"SAFETY: the benchmark changed protected files: {changed}"
        )

    metrics = {name: aggregate(rows, name) for name in CONFIGS}
    metrics["rrf_gain_vs_dense"] = gain(metrics["rrf"], metrics["dense"])
    metrics["rrf_gain_vs_bm25"] = gain(metrics["rrf"], metrics["bm25"])
    metrics["reranker_gain_vs_rrf"] = gain(metrics["rrf_reranker"], metrics["rrf"])
    metrics["reranker_latency_overhead_ms"] = round(
        metrics["rrf_reranker"]["latency_ms"]["mean"]
        - metrics["rrf"]["latency_ms"]["mean"],
        2,
    )
    metrics["metadata_filter"] = {
        "questions": len(rows),
        "applied": sum(1 for r in rows if r["metadata_filter"]["applied"]),
        "ambiguous": sum(1 for r in rows if r["metadata_filter"]["ambiguous"]),
        "empty_lookup": sum(1 for r in rows if r["metadata_filter"]["empty_lookup"]),
        "fallback_unfiltered": sum(
            1 for r in rows if not r["metadata_filter"]["applied"]
        ),
        "questions_with_valid_identifier": sum(
            1 for r in rows if r["metadata_filter"]["document_number"]
        ),
        "identifiers": sorted(
            {
                r["metadata_filter"]["document_number"]
                for r in rows
                if r["metadata_filter"]["document_number"]
            }
        ),
        "candidate_count_when_applied": [
            r["metadata_filter"]["candidate_count"]
            for r in rows
            if r["metadata_filter"]["applied"]
        ],
    }
    verdict = reranker_verdict(rows)
    recommendation = recommend(metrics, verdict)

    config_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_commit(),
        "index": index,
        "questions_file": str(args.questions),
        "question_count": len(questions),
        "question_ids": [q.question_id for q in questions],
        "k_values": {
            "dense_top_k": pipeline.dense_top_k,
            "bm25_top_k": pipeline.bm25_top_k,
            "rrf_k": pipeline.rrf_k,
            "rrf_top_k": pipeline.rrf_top_k,
            "reranker_top_k": pipeline.reranker_top_k,
            "reranker_enabled": pipeline.reranker_enabled,
        },
        "models": {
            "embedding": settings.embedding_model_name,
            "reranker": settings.reranker_model_name,
            "llm_not_called": settings.llm_model_name,
        },
        "devices": {
            "embedding": settings.embedding_device,
            "reranker": settings.reranker_device,
            "llm": runtime.device,
            "dtype": runtime.dtype,
        },
        "environment": env,
        "parameter_counts": runtime.parameter_counts,
    }

    payload = {
        "config": config_payload,
        "metrics": metrics,
        "rows": rows,
        "reranker": verdict,
        "recommendation": recommendation,
    }

    (args.out_dir / "config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.out_dir / "per_question.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "metrics.json").write_text(
        json.dumps(
            {**metrics, "reranker_effect": verdict, "recommendation": recommendation},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.out_dir / "report.md").write_text(render_report(payload), encoding="utf-8")
    (args.out_dir / "README.md").write_text(README, encoding="utf-8")

    print()
    print(f"INDEX: {index['current']} ({index['manifest']['child_count']} children)")
    print(f"QUESTIONS: {len(questions)}")
    print()
    print("RESULT:")
    for name, label in LABELS:
        m = metrics[name]
        print(
            f"  {label}: Recall@10 = {cell(m['recall_document@10'])}  "
            f"MRR = {cell(m['mrr'])}  "
            f"Latency = {m['latency_ms']['mean']} ms (p95 {m['latency_ms']['p95']})"
        )
    print()
    print(
        f"RERANKER: Recall@5 gain = {signed(gains_of(metrics))}  "
        f"Latency overhead = {metrics['reranker_latency_overhead_ms']} ms"
    )
    print(
        f"          improved {verdict['counts']['improved']} / "
        f"unchanged {verdict['counts']['unchanged']} / "
        f"worsened {verdict['counts']['worsened']} / "
        f"gold dropped {verdict['counts']['gold_dropped_from_top_k']}"
    )
    print()
    print(f"RECOMMENDATION: {recommendation[0]}")
    print(f"  {recommendation[1]}")
    print()
    print(f"artifacts: {args.out_dir}")
    return 0


def gains_of(metrics: dict[str, Any]) -> Any:
    return metrics["reranker_gain_vs_rrf"]["recall_document@5"]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
