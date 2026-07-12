"""Task 13 CLI: loading, one workflow call, one reporting call."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from sharper import __version__
from sharper.io import load_csv, load_excel
from sharper.reporting import generate_analysis_report
from sharper.workflow import AnalysisRun, run_analysis

app = typer.Typer(help="Analyze structured tabular data.")


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            is_eager=True,
            callback=lambda value: _version_callback(value),
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """Run Sharper commands."""


def _root_version() -> None:
    argv = sys.argv[1:]
    segment = argv[: argv.index("analyze")] if "analyze" in argv else argv
    if "--version" in segment:
        typer.echo(f"sharper {__version__}")
        raise typer.Exit(0)


def _version_callback(value: bool) -> bool:
    if value:
        typer.echo(f"sharper {__version__}")
        raise typer.Exit(0)
    return value


@app.command()
def analyze(
    input_path: Annotated[Path, typer.Argument(metavar="INPUT")],
    output_path: Annotated[Path, typer.Option("--output", "-o")],
    format: Annotated[str, typer.Option("--format")] = "markdown",
    target: Annotated[str | None, typer.Option("--target")] = None,
    task: Annotated[str | None, typer.Option("--task")] = None,
    id_columns: Annotated[list[str] | None, typer.Option("--id-column")] = None,
    exclude_columns: Annotated[
        list[str] | None, typer.Option("--exclude-column")
    ] = None,
    features: Annotated[list[str] | None, typer.Option("--feature")] = None,
    time_column: Annotated[str | None, typer.Option("--time-column")] = None,
    group_by: Annotated[str | None, typer.Option("--group-by")] = None,
    reference_date: Annotated[str | None, typer.Option("--reference-date")] = None,
    max_suggestions: Annotated[int, typer.Option("--max-suggestions")] = 50,
    model: Annotated[bool, typer.Option("--model/--no-model")] = False,
    test_size: Annotated[float, typer.Option("--test-size")] = 0.20,
    random_state: Annotated[int, typer.Option("--random-state")] = 42,
    overwrite: Annotated[bool, typer.Option("--overwrite/--no-overwrite")] = True,
    debug: Annotated[bool, typer.Option("--debug/--no-debug")] = False,
) -> None:
    """Analyze a local CSV or XLSX file and write a report bundle."""
    run: AnalysisRun | None = None
    acquired = False
    reporting_consumed = False
    try:
        suffix = input_path.suffix.lower()
        if suffix == ".csv":
            frame = load_csv(input_path)
        elif suffix == ".xlsx":
            frame = load_excel(input_path)
        else:
            raise ValueError("only .csv and .xlsx inputs are supported by analyze")
        run = run_analysis(
            frame,
            target=target,
            task=task,
            include_model=model,
            id_columns=tuple(id_columns or ()),
            exclude_columns=tuple(exclude_columns or ()),
            features=None if features is None else tuple(features),
            time_column=time_column,
            group_by=group_by,
            reference_date=reference_date,
            max_suggestions=max_suggestions,
            test_size=test_size,
            random_state=random_state,
        )
        artifact = generate_analysis_report(
            run, output_path, format=format, overwrite=overwrite
        )
        acquired = True
    except (ValueError, TypeError, OSError, FileExistsError) as error:
        reporting_consumed = bool(getattr(error, "_sharper_figures_consumed", False))
        if debug:
            raise
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    finally:
        if run is not None and not acquired and not reporting_consumed:
            _close_run_figures(run)
    typer.echo(f"Report written to: {artifact.path}")


def _close_run_figures(run: AnalysisRun) -> None:
    from matplotlib import pyplot as plt

    values = (
        run.distribution_plots,
        run.missingness_plot,
        run.correlation_plot,
        run.outlier_plots,
        run.group_plots,
        run.target_plots,
        run.evaluation_plots,
    )
    seen: set[int] = set()
    for value in values:
        plots = (
            ()
            if value is None
            else (value.plots if hasattr(value, "plots") else (value,))
        )
        for plot in plots:
            figure = plot.figure
            if id(figure) not in seen and figure.number in plt.get_fignums():
                seen.add(id(figure))
                plt.close(figure)


if __name__ == "__main__":
    try:
        _root_version()
    except typer.Exit as error:
        raise SystemExit(error.exit_code) from None
    app()
