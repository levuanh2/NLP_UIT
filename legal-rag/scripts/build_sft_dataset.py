"""Turn train.json into QLoRA examples that look exactly like inference.

The adapter has to see what the model will see at submission time, so each
example is the real grounded prompt — system rules plus retrieved evidence plus
the question — paired with the expert answer as the completion. Training on the
question alone would teach the model to answer from memory and then meet a
context-filled prompt it never saw.

Retrieval is the slow half, so finished examples are appended to the output and
a re-run resumes from what is already there.

Usage:
  python scripts/build_sft_dataset.py [--limit N] [--output ...]
"""

import argparse
import json
import random
import time
from pathlib import Path

from app.core.config import get_settings
from app.domain.queries import LegalQuery
from app.generation.llm.qwen_generator import QwenGenerator
from app.generation.prompt import LegalPromptBuilder
from app.services.runtime_factory import build_local_rag_runtime

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data/train/train.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/train/sft.jsonl")
    parser.add_argument("--limit", type=int, help="Use only the first N questions.")
    parser.add_argument(
        "--exclude",
        type=Path,
        default=ROOT / "data/questions/dev.json",
        help="Question file whose IDs must stay out of training. The dev slice "
        "comes out of train.json, so training on it would make its score "
        "meaningless. Pass a missing path to disable.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train = json.loads(args.train.read_text(encoding="utf-8"))
    held_out: set[str] = set()
    if args.exclude and args.exclude.is_file():
        held_out = set(json.loads(args.exclude.read_text(encoding="utf-8")))
    question_ids = [qid for qid in sorted(train) if qid not in held_out]
    print(f"held out {len(held_out)} dev ids from {args.exclude}")
    random.seed(args.seed)
    random.shuffle(question_ids)
    if args.limit:
        question_ids = question_ids[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.output.is_file():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["question_id"])
                except ValueError:
                    continue
    pending = [qid for qid in question_ids if qid not in done]
    print(
        f"train={len(train)} target={len(question_ids)} "
        f"done={len(done)} todo={len(pending)}"
    )
    if not pending:
        return 0

    settings = get_settings()
    runtime = build_local_rag_runtime(settings)
    builder = LegalPromptBuilder(
        max_context_tokens=settings.context_max_tokens,
        reserved_generation_tokens=settings.max_new_tokens,
    )

    started = time.perf_counter()
    with args.output.open("a", encoding="utf-8", newline="\n") as stream:
        for position, question_id in enumerate(pending, start=1):
            question = train[question_id]["question"]
            retrieval = runtime.service.retrieval_pipeline.retrieve(
                LegalQuery(question_id=question_id, question=question)
            )
            prompt = builder.build(question, retrieval)
            record = {
                "question_id": question_id,
                "messages": QwenGenerator._conversation(prompt),
                "answer": train[question_id]["answer"],
                "evidence_count": len(retrieval.evidences),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            if position % 50 == 0 or position == len(pending):
                rate = (time.perf_counter() - started) / position
                remaining = (len(pending) - position) * rate / 60
                print(
                    f"{position}/{len(pending)} {rate:.2f}s/q eta {remaining:.0f}m",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
