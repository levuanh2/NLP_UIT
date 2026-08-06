"""Build reusable semantic embeddings for LegalQA answer-memory tuning."""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--include-answers",
        action="store_true",
        help="Also encode long answers (substantially slower on CPU).",
    )
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    payload = json.loads(args.train.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Training dataset root must be an object.")
    ids = list(payload)
    questions = [payload[question_id]["question"] for question_id in ids]
    answers = [payload[question_id]["answer"] for question_id in ids]

    model = SentenceTransformer(
        str(args.model),
        device="cpu",
        local_files_only=True,
        trust_remote_code=True,
    )
    encode_options = {
        "batch_size": args.batch_size,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": True,
    }
    question_embeddings = model.encode(
        [f"passage: {question}" for question in questions],
        **encode_options,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matrices = {
        "ids": np.asarray(ids),
        "question_embeddings": question_embeddings.astype(np.float32),
    }
    if args.include_answers:
        matrices["answer_embeddings"] = model.encode(
            [f"passage: {answer}" for answer in answers],
            **encode_options,
        ).astype(np.float32)
    np.savez(args.output, **matrices)
    print(f"Saved {len(ids)} question embeddings to {args.output}")


if __name__ == "__main__":
    main()
