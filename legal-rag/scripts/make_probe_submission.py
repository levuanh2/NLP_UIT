"""Build a minimal submission that isolates the platform from our answers.

Same ids, same schema, one short sentence per answer. If the organizers' scorer
still fails on this, the fault is the platform or the phase, not our text.

Usage:
  python scripts/make_probe_submission.py [--questions ...] [--output ...]
"""

import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_MEMBER = "submission.json"
PROBE_ANSWER = "Theo quy định của pháp luật Việt Nam hiện hành."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions", type=Path, default=ROOT / "data/questions/public-official.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/outputs/probe/submission.json"
    )
    args = parser.parse_args()

    question_ids = list(json.loads(args.questions.read_text(encoding="utf-8")))
    payload = {question_id: {"answer": PROBE_ANSWER} for question_id in question_ids}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=4)
        stream.write("\n")

    archive = args.output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(args.output, arcname=ARCHIVE_MEMBER)

    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == [ARCHIVE_MEMBER], bundle.namelist()
        restored = json.loads(bundle.read(ARCHIVE_MEMBER).decode("utf-8"))
    assert set(restored) == set(question_ids)
    print(f"wrote {archive} | {len(restored)} ids | {archive.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
