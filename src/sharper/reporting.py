"""Deterministic Markdown reporting for the Task 05 workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sharper.workflow import AnalysisRun


@dataclass(frozen=True)
class ReportArtifact:
    """Describe a generated Task 05 report file.

    Attributes
    ----------
    path
        Output path exactly as resolved by ``Path(output_path)``.
    format
        Report format, fixed to ``"markdown"`` in Task 05.
    title
        Cleaned title written to the report.
    """

    path: Path
    format: str
    title: str


def generate_analysis_report(
    run: AnalysisRun,
    output_path: str | Path,
    *,
    title: str = "Sharper Analysis Report",
    format: str = "markdown",
    overwrite: bool = True,
) -> ReportArtifact:
    """Render an analysis run as deterministic Markdown and write it to disk.

    Parameters
    ----------
    run
        Existing structured workflow result. No analysis is recomputed.
    output_path
        Destination file path. Missing parent directories are created.
    title
        Report heading. Newlines are replaced with spaces and surrounding
        whitespace is removed.
    format
        Must be ``"markdown"`` in Task 05.
    overwrite
        Whether an existing output file may be replaced.

    Returns
    -------
    ReportArtifact
        Immutable description of the written file.

    Raises
    ------
    ValueError
        If the format is unsupported or the output path is an existing directory.
    FileExistsError
        If the file exists and ``overwrite`` is ``False``.
    OSError
        If directory creation or UTF-8 file writing fails.

    Notes
    -----
    The report preserves the structured results' missing-value and ordering
    semantics. Its only side effects are creating parent directories and writing
    the requested file.

    Examples
    --------
    >>> import pandas as pd
    >>> from sharper import generate_analysis_report, run_analysis
    >>> run = run_analysis(pd.DataFrame({"value": [1, 2]}))
    >>> artifact = generate_analysis_report(run, "report.md")
    """
    if format != "markdown":
        raise ValueError("only markdown reports are supported in Task 05")

    path = Path(output_path)
    if path.is_dir():
        raise ValueError("output_path must be a file path")
    if path.exists() and not overwrite:
        raise FileExistsError(f"output file already exists: {path}")

    cleaned_title = " ".join(str(title).splitlines()).strip()
    if not cleaned_title:
        cleaned_title = "Sharper Analysis Report"

    content = _render_markdown(run, cleaned_title)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return ReportArtifact(path=path, format="markdown", title=cleaned_title)


def _render_markdown(run: AnalysisRun, title: str) -> str:
    sections = [
        f"# {title}",
        "## Overview\n\n"
        f"- Rows: {run.summary.n_rows}\n"
        f"- Columns: {run.summary.n_columns}\n"
        f"- Quality issues: {run.quality.issue_count}",
        "## Schema\n\n" + _schema_table(run),
        "## DataFrame Summary\n\n" + _summary_table(run),
        "## Data Quality\n\n" + _quality_content(run),
        "## Skipped Capabilities\n\n"
        + _bullet_content(run.skipped, "No capabilities were skipped."),
        "## Warnings\n\n" + _bullet_content(run.warnings, "No warnings."),
    ]
    return "\n\n".join(sections).rstrip("\n") + "\n"


def _schema_table(run: AnalysisRun) -> str:
    headers = [
        "Column",
        "Logical type",
        "Pandas dtype",
        "Missing rate",
        "Unique count",
        "ID-like",
    ]
    rows = [
        [
            column.name,
            column.logical_type,
            column.pandas_dtype,
            _format_value(column.missing_rate),
            str(column.unique_count),
            _format_value(column.is_id_like),
        ]
        for column in run.schema.columns
    ]
    return _pipe_table(headers, rows)


def _summary_table(run: AnalysisRun) -> str:
    columns = [
        "column",
        "logical_type",
        "missing_rate",
        "unique_count",
        "is_constant",
        "is_id_like",
    ]
    rows = [
        [_format_value(row[column]) for column in columns]
        for _, row in run.summary.column_summary.iterrows()
    ]
    return _pipe_table(columns, rows)


def _quality_content(run: AnalysisRun) -> str:
    if not run.quality.issues:
        return "No data quality issues detected."
    headers = ["Code", "Severity", "Scope", "Column", "Count", "Ratio", "Suggestion"]
    rows = [
        [
            issue.code,
            issue.severity,
            issue.scope,
            _format_value(issue.column),
            _format_value(issue.count),
            _format_value(issue.ratio),
            issue.suggestion,
        ]
        for issue in run.quality.issues
    ]
    return _pipe_table(headers, rows)


def _pipe_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _bullet_content(values: tuple[str, ...], empty_text: str) -> str:
    if not values:
        return empty_text
    return "\n".join(f"- {value}" for value in values)


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
