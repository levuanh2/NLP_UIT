"""Legal RAG CLI entrypoint."""

import typer

from app.cli.answer import answer
from app.cli.ask import ask
from app.cli.evaluate import evaluate
from app.cli.evaluate_grounding import evaluate_grounding
from app.cli.index import index
from app.cli.ingest import ingest
from app.cli.submit import submit, validate_submission

app = typer.Typer(
    name="legal-rag",
    help="Local Vietnamese Legal RAG for Subtask 2.",
    no_args_is_help=True,
)
app.command("ingest")(ingest)
app.command("index")(index)
app.command("ask")(ask)
app.command("answer")(answer)
app.command("submit")(submit)
app.command("validate-submission")(validate_submission)
app.command("evaluate")(evaluate)
app.command("evaluate-grounding")(evaluate_grounding)


if __name__ == "__main__":
    app()
