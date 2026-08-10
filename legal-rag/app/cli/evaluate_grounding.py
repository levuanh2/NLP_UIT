"""CLI for a reproducible, bounded GroundingValidator evaluation."""

from pathlib import Path

import typer

from app.evaluation.grounding_evaluation import GroundingEvaluation


def evaluate_grounding(
    questions: Path = typer.Option(..., "--questions", "-q"),
    output_dir: Path = typer.Option(
        Path("data/evaluation/grounding_50"), "--output-dir", "-o"
    ),
    sample_size: int = typer.Option(50, min=1),
    seed: int = typer.Option(20260810),
    prepare_only: bool = typer.Option(False),
) -> None:
    """Generate predictions and a human annotation template for sampled questions."""
    evaluator = GroundingEvaluation(
        question_path=questions,
        output_dir=output_dir,
        sample_size=sample_size,
        seed=seed,
        forced_ids=("31221", "57711"),
    )
    evaluator.prepare()
    if prepare_only:
        typer.echo(f"Prepared grounding evaluation at {output_dir}")
        return
    evaluator.run()
    typer.echo(f"Grounding evaluation predictions complete: {output_dir}")
