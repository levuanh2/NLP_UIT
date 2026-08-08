"""Explicitly download the three local model snapshots used by Legal RAG."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = {
    "embedding": (
        "bqbbao6/vietnamese-legal-embedding",
        "vietnamese-legal-embedding",
    ),
    "reranker": ("AITeamVN/Vietnamese_Reranker", "Vietnamese_Reranker"),
    "llm": ("AITeamVN/Vi-Qwen2-1.5B-RAG", "Vi-Qwen2-1.5B-RAG"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download versioned Hugging Face snapshots for offline inference."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Destination root (default: models).",
    )
    parser.add_argument(
        "--only",
        choices=["all", *MODELS],
        default="all",
        help="Download one component or all three (default: all).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hub revision/tag/commit to resolve (default: main).",
    )
    return parser.parse_args()


def download_models(model_dir: Path, only: str, revision: str) -> list[Path]:
    """Download selected snapshots and return their resolved local directories."""
    model_dir = model_dir.expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    selected = MODELS.items() if only == "all" else [(only, MODELS[only])]
    downloaded: list[Path] = []

    for name, (repo_id, directory_name) in selected:
        destination = model_dir / directory_name
        print(f"[{name}] {repo_id} -> {destination}")
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=destination,
        )
        downloaded.append(destination)

    return downloaded


def main() -> None:
    args = parse_args()
    paths = download_models(args.model_dir, args.only, args.revision)
    print("Downloaded models:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
