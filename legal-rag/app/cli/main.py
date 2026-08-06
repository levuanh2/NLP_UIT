"""Legal RAG CLI entrypoint."""

import typer

from app.cli.ask import ask
from app.cli.baseline_evaluate import evaluate_baseline
from app.cli.corpus import build_corpus_index
from app.cli.evaluate import evaluate
from app.cli.index import index
from app.cli.ingest import ingest
from app.cli.solve import solve
from app.cli.submit import submit, validate_submission

app = typer.Typer(
    name="legal-rag",
    help="Local Vietnamese Legal RAG scaffold for Subtask 2.",
    no_args_is_help=True,
)
app.command("ingest")(ingest)
app.command("build-corpus-index")(build_corpus_index)
app.command("index")(index)
app.command("ask")(ask)
app.command("submit")(submit)
app.command("solve")(solve)
app.command("validate-submission")(validate_submission)
app.command("evaluate")(evaluate)
app.command("evaluate-baseline")(evaluate_baseline)


if __name__ == "__main__":
    app()
