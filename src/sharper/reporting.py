"""Deterministic Task 13 Markdown/HTML report bundles."""

from __future__ import annotations

import html
import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from sharper.analysis import (
    CategoricalAnalysis,
    CorrelationAnalysis,
    GroupComparison,
    NumericAnalysis,
    OutlierAnalysis,
    TargetAnalysis,
)
from sharper.evaluation import ClassificationEvaluation, RegressionEvaluation
from sharper.features import FeatureSuggestionReport
from sharper.modeling import RegressionTrainingResult, TrainingResult
from sharper.quality import QualityReport
from sharper.schema import SchemaReport
from sharper.summary import DataFrameSummary
from sharper.visualization import PlotCollection, PlotResult
from sharper.workflow import AnalysisRun


@dataclass(frozen=True)
class ReportArtifact:
    """A written report bundle's final report path, format, and cleaned title."""

    path: Path
    format: str
    title: str


def generate_analysis_report(
    run: AnalysisRun,
    output_path: str | Path,
    *,
    title: str = "Sharper Analysis Report",
    format: Literal["markdown", "html"] = "markdown",
    overwrite: bool = True,
) -> ReportArtifact:
    """Write an existing analysis run as a deterministic report-and-PNG bundle.

    Reporting never recomputes analysis. It acquires and closes run Figures only
    after all validation and in-memory rendering succeed.
    """
    path, cleaned_title = _arguments(output_path, title, format, overwrite)
    assets = path.parent / f"{path.stem}_assets"
    staging_report = path.parent / f".{path.name}.sharper-staging"
    staging_assets = path.parent / f".{assets.name}.sharper-staging"
    backup_report = path.parent / f".{path.name}.sharper-backup"
    backup_assets = path.parent / f".{assets.name}.sharper-backup"
    if not overwrite and (path.exists() or assets.exists()):
        raise FileExistsError("output file or asset directory already exists")
    if any(
        p.exists()
        for p in (staging_report, staging_assets, backup_report, backup_assets)
    ):
        raise FileExistsError("staging or backup path already exists")
    figures = _preflight(run)
    try:
        body = _render(run, cleaned_title, format, figures, assets.name)
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception as error:
        raise ValueError("analysis run has invalid schema") from error
    # Ownership is acquired immediately before the first staging side effect.
    assets_backed_up = False
    report_backed_up = False
    assets_committed = False
    try:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            staging_assets.mkdir()
            for number, (_, figure, _) in enumerate(figures, 1):
                figure.savefig(staging_assets / f"plot-{number:03d}.png")
            with staging_report.open("w", encoding="utf-8") as handle:
                handle.write(body)
                handle.flush()
        except Exception as error:
            try:
                _clean_staging(staging_report, staging_assets)
            except Exception as compensation_error:
                raise OSError("failed to write report output") from compensation_error
            raise OSError("failed to write report output") from error
        try:
            if assets.exists():
                assets.replace(backup_assets)
                assets_backed_up = True
            if path.exists():
                path.replace(backup_report)
                report_backed_up = True
        except Exception as error:
            try:
                if assets_backed_up:
                    _restore(backup_assets, assets)
                _clean_staging(staging_report, staging_assets)
            except Exception as compensation_error:
                raise OSError("failed to write report output") from compensation_error
            raise OSError("failed to write report output") from error
        try:
            staging_assets.replace(assets)
            assets_committed = True
            staging_report.replace(path)
        except Exception as error:
            try:
                if assets_committed:
                    _remove(assets)
                if report_backed_up:
                    _restore(backup_report, path)
                if assets_backed_up:
                    _restore(backup_assets, assets)
                _clean_staging(staging_report, staging_assets)
            except Exception as compensation_error:
                raise OSError("failed to write report output") from compensation_error
            raise OSError("failed to write report output") from error
        try:
            _remove(backup_report)
        except Exception as error:
            raise OSError("failed to write report output") from error
        try:
            _remove(backup_assets)
        except Exception as error:
            raise OSError("failed to write report output") from error
        return ReportArtifact(path=path, format=format, title=cleaned_title)
    except OSError as error:
        # Private hand-off for the CLI: this exception occurred after ownership.
        error._sharper_figures_consumed = True  # type: ignore[attr-defined]
        raise
    finally:
        for _, figure, _ in figures:
            if figure.number in plt.get_fignums():
                plt.close(figure)


def _arguments(
    output_path: str | Path, title: object, format: object, overwrite: object
) -> tuple[Path, str]:
    if not isinstance(output_path, (str, Path)):
        raise ValueError("output_path must be a string or Path")
    path = Path(output_path)
    if path.exists() and path.is_dir():
        raise ValueError("output_path must be a file path")
    if not isinstance(title, str):
        raise ValueError("title must be a string")
    if format not in {"markdown", "html"}:
        raise ValueError("format must be markdown or html")
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean")
    cleaned = " ".join(title.splitlines()).strip() or "Sharper Analysis Report"
    return path, cleaned


def _preflight(run: object) -> list[tuple[PlotResult, Figure, str]]:
    if type(run) is not AnalysisRun:
        raise TypeError("run must be an AnalysisRun")
    try:
        expected = [field.name for field in fields(AnalysisRun)]
        if list(run.__dataclass_fields__) != expected:
            raise ValueError("fields")
        plots: list[PlotResult] = []
        for value, collection in (
            (run.distribution_plots, True),
            (run.missingness_plot, False),
            (run.correlation_plot, False),
            (run.outlier_plots, True),
            (run.group_plots, True),
            (run.target_plots, True),
            (run.evaluation_plots, True),
        ):
            if value is None:
                continue
            if collection:
                if type(value) is not PlotCollection or value.actual_count != len(
                    value.plots
                ):
                    raise ValueError("plots")
                plots.extend(value.plots)
            else:
                if type(value) is not PlotResult:
                    raise ValueError("plot")
                plots.append(value)
        seen: set[int] = set()
        result: list[tuple[PlotResult, Figure, str]] = []
        for index, plot in enumerate(plots, 1):
            if (
                type(plot) is not PlotResult
                or not isinstance(plot.figure, Figure)
                or plot.figure.number not in plt.get_fignums()
                or id(plot.figure) in seen
            ):
                raise ValueError("figure")
            seen.add(id(plot.figure))
            result.append((plot, plot.figure, f"plot-{index:03d}.png"))
        if run.task not in {None, "classification", "regression"}:
            raise ValueError("task")
        _exact_results(run)
        if (run.training is None) != (run.evaluation is None):
            raise ValueError("union")
        if run.task is None and (
            run.training is not None or run.target_analysis is not None
        ):
            raise ValueError("optional")
        if run.include_model:
            if run.task == "classification" and (
                type(run.training) is not TrainingResult
                or type(run.evaluation) is not ClassificationEvaluation
            ):
                raise ValueError("classification union")
            if run.task == "regression" and (
                type(run.training) is not RegressionTrainingResult
                or type(run.evaluation) is not RegressionEvaluation
            ):
                raise ValueError("regression union")
        elif (
            run.training is not None
            or run.evaluation is not None
            or run.evaluation_plots is not None
        ):
            raise ValueError("model optional")
        if run.training is not None and (
            run.training.target != run.target
            or run.evaluation.target != run.target
            or run.training.test_row_positions != run.evaluation.holdout_positions
            or run.training.feature_columns
            != (
                run.features
                if run.features is not None
                else run.training.feature_columns
            )
        ):
            raise ValueError("cross result")
        if not all(isinstance(value, str) for value in run.warnings + run.limitations):
            raise ValueError("messages")
        return result
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception as error:
        raise ValueError("analysis run has invalid schema") from error


def _render(
    run: AnalysisRun,
    title: str,
    format: str,
    figures: list[tuple[PlotResult, Figure, str]],
    asset_name: str,
) -> str:
    sections = [
        (
            "Overview",
            "- Rows: "
            f"{run.summary.n_rows}\n- Columns: {run.summary.n_columns}\n"
            f"- Quality issues: {run.quality.issue_count}",
        ),
        ("Schema", _frame(run.summary.column_summary)),
        ("DataFrame Summary", _frame(run.summary.column_summary)),
        ("Data Quality", _quality(run)),
        ("Numeric Feature Analysis", _frame(run.numeric_analysis.summary)),
        ("Categorical Feature Analysis", _frame(run.categorical_analysis.summary)),
        ("Correlations", _frame(run.correlation_analysis.correlations)),
        ("Outliers", _frame(run.outlier_analysis.summary)),
        ("Feature Suggestions", _frame(_suggestion_frame(run))),
        (
            "Group Comparison",
            _frame(run.group_comparison.summary)
            if run.group_comparison
            else "Not requested.",
        ),
        (
            "Target Relationships",
            _frame(run.target_analysis.numeric_details)
            if run.target_analysis
            else "Not requested.",
        ),
        (
            "Model Training",
            "Not requested." if run.training is None else f"Task: {run.training.task}",
        ),
        (
            "Model Evaluation",
            "Not requested."
            if run.evaluation is None
            else _metrics(run.evaluation.metrics),
        ),
        ("Visualizations", _images(figures, asset_name, format)),
        (
            "Skipped Capabilities",
            _bullets(run.skipped, "No capabilities were skipped."),
        ),
        ("Warnings", _bullets(run.warnings, "No warnings.")),
        ("Limitations", _bullets(run.limitations, "No limitations.")),
    ]
    if format == "markdown":
        return (
            "\n\n".join(
                [f"# {title}"] + [f"## {name}\n\n{body}" for name, body in sections]
            ).rstrip("\n")
            + "\n"
        )
    content = "".join(
        f"<h2>{html.escape(name)}</h2>\n{_html_body(body)}\n" for name, body in sections
    )
    return (
        "<!doctype html>\n<html><body>\n"
        f"<h1>{html.escape(title)}</h1>\n{content}</body></html>\n"
    )


def _frame(frame: object) -> str:
    if not hasattr(frame, "empty") or frame.empty:
        return "No applicable results."
    columns = [str(column) for column in frame.columns]
    rows = [
        [_cell(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def _suggestion_frame(run: AnalysisRun):
    import pandas as pd

    return pd.DataFrame(
        [
            {"name": s.name, "type": s.feature_type, "reason": s.reason, "risk": s.risk}
            for s in run.feature_suggestions.suggestions
        ]
    )


def _quality(run: AnalysisRun) -> str:
    return (
        "No data quality issues detected."
        if not run.quality.issues
        else "\n".join(f"- {issue.message}" for issue in run.quality.issues)
    )


def _metrics(values: tuple[tuple[str, float], ...]) -> str:
    return "\n".join(f"- {name}: {value}" for name, value in values)


def _bullets(values: tuple[str, ...], empty: str) -> str:
    return empty if not values else "\n".join(f"- {value}" for value in values)


def _images(
    figures: list[tuple[PlotResult, Figure, str]], assets: str, format: str
) -> str:
    if not figures:
        return "No visualizations were generated."
    if format == "html":
        return "\n".join(
            f'<img src="{html.escape(assets + "/" + name)}" '
            f'alt="{html.escape(plot.title)}">'
            for plot, _, name in figures
        )
    return "\n".join(f"![{plot.title}]({assets}/{name})" for plot, _, name in figures)


def _html_body(body: str) -> str:
    if body.startswith("<img"):
        return body
    lines = body.splitlines()
    if len(lines) >= 2 and lines[0].startswith("|") and lines[1].startswith("|"):
        headers = _pipe_cells(lines[0])
        rows = [_pipe_cells(line) for line in lines[2:] if line.startswith("|")]
        head = "".join(f"<th>{html.escape(cell)}</th>" for cell in headers)
        data = "".join(
            "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
            for row in rows
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{data}</tbody></table>"
    return "<pre>" + html.escape(body) + "</pre>"


def _pipe_cells(line: str) -> list[str]:
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for character in line.strip()[1:-1]:
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())
    return cells


def _exact_results(run: AnalysisRun) -> None:
    required = (
        (run.schema, SchemaReport),
        (run.summary, DataFrameSummary),
        (run.quality, QualityReport),
        (run.numeric_analysis, NumericAnalysis),
        (run.categorical_analysis, CategoricalAnalysis),
        (run.correlation_analysis, CorrelationAnalysis),
        (run.outlier_analysis, OutlierAnalysis),
        (run.feature_suggestions, FeatureSuggestionReport),
    )
    if any(type(value) is not kind for value, kind in required):
        raise ValueError("nested result")
    _validate_table(run.summary.column_summary)
    _validate_table(run.numeric_analysis.summary)
    _validate_table(run.categorical_analysis.summary)
    _validate_table(run.categorical_analysis.top_categories)
    _validate_table(run.correlation_analysis.correlations)
    _validate_table(run.outlier_analysis.summary)
    _validate_table(run.outlier_analysis.outliers)
    for optional, kind in (
        (run.group_comparison, GroupComparison),
        (run.target_analysis, TargetAnalysis),
    ):
        if optional is not None and type(optional) is not kind:
            raise ValueError("optional result")
    if run.group_comparison is not None:
        _validate_table(run.group_comparison.summary)
    if run.target_analysis is not None:
        _validate_table(run.target_analysis.numeric_details)
        _validate_table(run.target_analysis.category_details)
        _validate_table(run.target_analysis.statistical_tests)


def _validate_table(table: object) -> None:
    """Reject structurally malformed reporting tables without recomputation."""
    import pandas as pd

    if type(table) is not pd.DataFrame or not table.columns.is_unique:
        raise ValueError("table")
    if not all(isinstance(column, str) for column in table.columns):
        raise ValueError("table")
    if not table.index.is_unique and not isinstance(table.index, pd.RangeIndex):
        raise ValueError("table")
    for dtype in table.dtypes:
        if str(dtype) == "object":
            continue
        if getattr(dtype, "kind", "") not in {"b", "i", "u", "f", "M", "m"}:
            raise ValueError("table")


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r\n", " ")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _restore(source: Path, destination: Path) -> None:
    if source.exists():
        source.replace(destination)


def _clean_staging(report: Path, assets: Path) -> None:
    """Remove this call's two staging paths in report-then-assets order."""
    _remove(report)
    _remove(assets)
