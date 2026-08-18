"""Score answers against the BTC train answers, or carve a dev slice out of them.

The competition reports METEOR (primary) and ROUGE-L, and train.json carries
7000 expert answers, so a change can be measured here instead of spent on a
leaderboard submission.

Usage:
  python scripts/eval_dev.py --make 50            # write a dev question file
  python scripts/eval_dev.py --score data/outputs/dev/submission.json
"""

import argparse
import json
import random
import statistics
from pathlib import Path

from app.evaluation.generation_metrics import meteor, rouge_l

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data/train/train.json")
    parser.add_argument("--make", type=int, help="Sample N questions into --questions.")
    parser.add_argument(
        "--questions", type=Path, default=ROOT / "data/questions/dev.json"
    )
    parser.add_argument("--score", type=Path, help="Submission JSON to score.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--worst", type=int, default=5, help="Show the N worst answers."
    )
    args = parser.parse_args()

    train = json.loads(args.train.read_text(encoding="utf-8"))

    if args.make:
        random.seed(args.seed)
        chosen = random.sample(sorted(train), args.make)
        payload = {
            qid: {"question": train[qid]["question"], "answer": None} for qid in chosen
        }
        args.questions.parent.mkdir(parents=True, exist_ok=True)
        with args.questions.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=4)
            stream.write("\n")
        print(f"wrote {args.questions} | {len(payload)} questions | seed {args.seed}")
        return 0

    if not args.score:
        parser.error("pass --make N or --score <submission.json>")

    submission = json.loads(args.score.read_text(encoding="utf-8"))
    missing = [qid for qid in submission if qid not in train]
    if missing:
        print(f"skipping {len(missing)} ids absent from train: {missing[:5]}")

    scored = []
    for qid, record in submission.items():
        if qid not in train:
            continue
        prediction, reference = record["answer"], train[qid]["answer"]
        scored.append(
            (meteor(prediction, reference), rouge_l(prediction, reference), qid)
        )
    if not scored:
        print("nothing to score")
        return 1

    meteors = [row[0] for row in scored]
    rouges = [row[1] for row in scored]
    lengths = [len(submission[row[2]]["answer"].split()) for row in scored]
    reference_lengths = [len(train[row[2]]["answer"].split()) for row in scored]
    print(f"scored {len(scored)} answers")
    print(
        f"METEOR   {statistics.mean(meteors):.4f}   "
        f"ROUGE-L  {statistics.mean(rouges):.4f}"
    )
    print(
        f"words    predicted median {statistics.median(lengths):.0f}  "
        f"reference median {statistics.median(reference_lengths):.0f}"
    )

    scored.sort()
    print(f"worst {args.worst}:")
    for score, _, qid in scored[: args.worst]:
        words = len(submission[qid]["answer"].split())
        print(f"  {score:.3f}  {qid}  {words} words  {submission[qid]['answer'][:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
