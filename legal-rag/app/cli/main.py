"""Legal RAG CLI entrypoint."""

import sys
from typing import TextIO

import typer

from app.cli.ask import ask
from app.cli.ask_rag import ask_rag
from app.cli.baseline_evaluate import evaluate_baseline
from app.cli.corpus import build_corpus_index
from app.cli.evaluate import evaluate
from app.cli.index import index
from app.cli.ingest import ingest
from app.cli.solve import solve
from app.cli.solve_corpus import solve_corpus
from app.cli.solve_rag import solve_rag
from app.cli.submit import submit, validate_submission


def configure_utf8_console(*streams: TextIO) -> None:
    """Ensure Vietnamese JSON can be written on legacy Windows consoles."""
    for stream in streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


configure_utf8_console(sys.stdout, sys.stderr)

app = typer.Typer(
    name="legal-rag",
    help="Local end-to-end Vietnamese Legal RAG pipeline for Subtask 2.",
    no_args_is_help=True,
)
app.command("ingest")(ingest)
app.command("build-corpus-index")(build_corpus_index)
app.command("index")(index)
app.command("ask")(ask)
app.command("ask-rag")(ask_rag)
app.command("submit")(submit)
app.command("solve")(solve)
app.command("solve-corpus")(solve_corpus)
app.command("solve-rag")(solve_rag)
app.command("validate-submission")(validate_submission)
app.command("evaluate")(evaluate)
app.command("evaluate-baseline")(evaluate_baseline)


if __name__ == "__main__":
    app()
