"""Typer command-line interface for Sharper's minimal Task 05 workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from sharper.io import load_csv
from sharper.reporting import generate_analysis_report
from sharper.workflow import run_analysis

app = typer.Typer(help="Analyze structured tabular data.")


@app.callback()
def main() -> None:
    """Run Sharper commands."""


@app.command()
def analyze(
    input_path: Annotated[Path, typer.Argument(metavar="INPUT")],
    output_path: Annotated[
        Path,
        typer.Option("--output", "-o", help="Markdown output file."),
    ],
    target: Annotated[str | None, typer.Option("--target")] = None,
    task: Annotated[str | None, typer.Option("--task")] = None,
    id_columns: Annotated[list[str] | None, typer.Option("--id-column")] = None,
    exclude_columns: Annotated[
        list[str] | None, typer.Option("--exclude-column")
    ] = None,
    random_state: Annotated[int, typer.Option("--random-state")] = 42,
    format: Annotated[str, typer.Option("--format")] = "markdown",
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = True,
) -> None:
    """Analyze INPUT CSV and write a deterministic Markdown report."""
    try:
        frame = load_csv(input_path)
        run = run_analysis(
            frame,
            target=target,
            task=task,
            id_columns=tuple(id_columns or ()),
            exclude_columns=tuple(exclude_columns or ()),
            random_state=random_state,
        )
        artifact = generate_analysis_report(
            run,
            output_path,
            format=format,
            overwrite=overwrite,
        )
    except (ValueError, OSError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Report written to: {artifact.path}")
