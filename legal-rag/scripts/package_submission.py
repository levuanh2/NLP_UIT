"""Build submission.json + submission.zip from the run checkpoint.

Normalization runs here, not during generation, so text fixes can be re-applied
to answers that are already generated without touching the model.

Usage:
  python scripts/package_submission.py [--checkpoint ...] [--questions ...]
                                       [--output ...] [--allow-missing]
"""

import argparse
import json
import zipfile
from pathlib import Path

from app.domain.generation import GeneratedAnswer
from app.generation.prompts.system import ABSTENTION_ANSWER
from app.submission.formatter import SubmissionFormatter
from app.submission.validator import SubmissionValidator
from app.submission.writer import SubmissionWriter

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_MEMBER = "submission.json"  # the name the organisers require inside the zip


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "data/outputs/submission.partial.jsonl",
    )
    parser.add_argument(
        "--questions", type=Path, default=ROOT / "data/questions/public-official.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/outputs/submission.json"
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Escape Vietnamese as \\uXXXX. The parsed answers are identical, but "
        "the file is then pure ASCII and cannot be mis-decoded by the grader.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Fill unanswered questions with the abstention sentence instead of "
        "refusing to package an incomplete run.",
    )
    args = parser.parse_args()

    expected = list(json.loads(args.questions.read_text(encoding="utf-8")))
    answers: dict[str, GeneratedAnswer] = {}
    for line in args.checkpoint.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            answer = GeneratedAnswer.model_validate_json(line)
        except ValueError:
            continue  # a torn final line from a hard kill
        answers[answer.question_id] = answer

    missing = [question_id for question_id in expected if question_id not in answers]
    print(f"checkpoint={len(answers)} expected={len(expected)} missing={len(missing)}")
    if missing and not args.allow_missing:
        print(f"first missing: {', '.join(missing[:5])}")
        print("refusing to package an incomplete run; pass --allow-missing to fill")
        return 1
    for question_id in missing:
        answers[question_id] = GeneratedAnswer(
            question_id=question_id, answer=ABSTENTION_ANSWER, grounded=False
        )

    ordered = [answers[question_id] for question_id in expected]
    submission = SubmissionFormatter().format(ordered)
    SubmissionWriter(ensure_ascii=args.ascii).write(
        submission, args.output, expected_question_ids=set(expected)
    )

    result = SubmissionValidator().validate(args.output, set(expected))
    if not result.valid:
        for error in result.errors[:10]:
            print(error)
        return 1

    archive = args.output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(args.output, arcname=ARCHIVE_MEMBER)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == [ARCHIVE_MEMBER], bundle.namelist()
        payload = json.loads(bundle.read(ARCHIVE_MEMBER).decode("utf-8"))
    assert set(payload) == set(expected), "packaged IDs differ from the question set"

    lengths = sorted(len(value["answer"]) for value in payload.values())
    print(f"wrote {args.output} and {archive}")
    print(
        f"{len(payload)} answers | chars min={lengths[0]} "
        f"median={lengths[len(lengths) // 2]} max={lengths[-1]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
