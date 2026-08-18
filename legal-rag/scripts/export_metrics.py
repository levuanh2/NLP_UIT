"""Write the training curve and the per-question scores out as CSV.

Two files, because they answer different questions. The history is what the
trainer recorded step by step; the scores are what the finished adapter does on
questions it never trained on.

Usage:
  python scripts/export_metrics.py --run models/qlora-answerer \
      --score data/outputs/dev-qlora/submission.json
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from app.evaluation.generation_metrics import meteor, rouge_l

ROOT = Path(__file__).resolve().parents[1]


def write_history(run: Path, output: Path) -> int:
    """Merge every checkpoint's log so a trimmed save_total_limit loses nothing."""
    merged: dict[tuple[int, str], dict] = {}
    states = sorted(
        run.glob("checkpoint-*/trainer_state.json"),
        key=lambda path: int(path.parent.name.split("-")[1]),
    )
    for path in states:
        for entry in json.loads(path.read_text(encoding="utf-8"))["log_history"]:
            kind = "eval" if "eval_loss" in entry else "train"
            merged[(entry.get("step", -1), kind)] = entry

    rows = []
    for (step, kind), entry in sorted(merged.items()):
        rows.append(
            {
                "step": step,
                "epoch": round(entry.get("epoch", 0.0), 4),
                "split": kind,
                "learning_rate": entry.get("learning_rate", ""),
                "loss": entry.get("loss", entry.get("eval_loss", "")),
                "grad_norm": entry.get("grad_norm", ""),
                "runtime_seconds": entry.get("eval_runtime", ""),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["step"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} | {len(rows)} rows from {len(states)} checkpoints")
    return len(rows)


def write_scores(submission: Path, train: Path, output: Path) -> None:
    """Score every answer against its expert reference, one row per question."""
    references = json.loads(train.read_text(encoding="utf-8"))
    answers = json.loads(submission.read_text(encoding="utf-8"))

    rows = []
    for question_id, record in answers.items():
        if question_id not in references:
            continue
        prediction = record["answer"]
        reference = references[question_id]["answer"]
        rows.append(
            {
                "question_id": question_id,
                "meteor": round(meteor(prediction, reference), 6),
                "rouge_l": round(rouge_l(prediction, reference), 6),
                "predicted_words": len(prediction.split()),
                "reference_words": len(reference.split()),
            }
        )
    if not rows:
        print(f"no scorable answers in {submission}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    meteors = [row["meteor"] for row in rows]
    rouges = [row["rouge_l"] for row in rows]
    print(
        f"wrote {output} | {len(rows)} answers | "
        f"METEOR {statistics.mean(meteors):.4f} "
        f"ROUGE-L {statistics.mean(rouges):.4f} | "
        f"words {statistics.median(row['predicted_words'] for row in rows):.0f} "
        f"vs {statistics.median(row['reference_words'] for row in rows):.0f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "models/qlora-answerer")
    parser.add_argument("--train", type=Path, default=ROOT / "data/train/train.json")
    parser.add_argument("--score", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/outputs/metrics")
    args = parser.parse_args()

    if args.run.is_dir():
        write_history(args.run, args.out_dir / "training_history.csv")
    else:
        print(f"no run directory at {args.run}; skipping the training curve")

    for submission in args.score:
        if not submission.is_file():
            print(f"missing submission {submission}")
            continue
        name = submission.parent.name or submission.stem
        write_scores(submission, args.train, args.out_dir / f"scores_{name}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
