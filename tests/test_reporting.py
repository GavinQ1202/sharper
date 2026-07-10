"""Contract tests for deterministic Task 05 Markdown reports."""

from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import get_type_hints

import pandas as pd
import pytest

from sharper import (
    ReportArtifact,
    generate_analysis_report,
    run_analysis,
)


def test_report_artifact_frozen_fields_and_types(tmp_path: Path) -> None:
    assert [field.name for field in fields(ReportArtifact)] == [
        "path",
        "format",
        "title",
    ]
    assert get_type_hints(ReportArtifact) == {
        "path": Path,
        "format": str,
        "title": str,
    }
    artifact = generate_analysis_report(
        run_analysis(pd.DataFrame({"value": [1, 2]})),
        tmp_path / "report.md",
    )
    with pytest.raises(FrozenInstanceError):
        artifact.title = "changed"  # type: ignore[misc]


def test_markdown_fixed_sections_no_issues_and_single_newline(tmp_path: Path) -> None:
    run = run_analysis(
        pd.DataFrame({"value": [1, 1, 2, 2], "group": ["a", "b", "a", "b"]})
    )
    path = tmp_path / "nested" / "report.md"
    artifact = generate_analysis_report(run, path)
    content = path.read_text(encoding="utf-8")

    assert artifact == ReportArtifact(
        path=path,
        format="markdown",
        title="Sharper Analysis Report",
    )
    headings = [
        "# Sharper Analysis Report",
        "## Overview",
        "## Schema",
        "## DataFrame Summary",
        "## Data Quality",
        "## Skipped Capabilities",
        "## Warnings",
    ]
    assert [content.index(heading) for heading in headings] == sorted(
        content.index(heading) for heading in headings
    )
    assert "No data quality issues detected." in content
    assert "No warnings." in content
    assert content.endswith("\n")
    assert not content.endswith("\n\n")


def test_markdown_quality_table_skipped_warnings_and_values(tmp_path: Path) -> None:
    run = run_analysis(
        pd.DataFrame({"id": [1, 2], "constant": [None, None], "target": [0, 1]}),
        target="target",
        id_columns=["id"],
    )
    path = tmp_path / "report.md"
    generate_analysis_report(run, path)
    content = path.read_text(encoding="utf-8")

    assert (
        "| Code | Severity | Scope | Column | Count | Ratio | Suggestion |" in content
    )
    assert "all_missing_column" in content
    assert "| constant | unknown | object | 1.0000 | 0 | false |" in content
    assert "- modeling\n- visualization\n- feature_engineering" in content
    assert (
        "- target recorded but target analysis is not available in Task 05" in content
    )
    assert "- id_columns recorded but not applied in Task 05" in content


def test_title_newline_cleanup_and_empty_fallback(tmp_path: Path) -> None:
    run = run_analysis(pd.DataFrame({"value": [1, 2]}))
    artifact = generate_analysis_report(
        run, tmp_path / "custom.md", title="  First\nSecond  "
    )
    assert artifact.title == "First Second"
    assert (
        (tmp_path / "custom.md")
        .read_text(encoding="utf-8")
        .startswith("# First Second\n")
    )

    fallback = generate_analysis_report(run, tmp_path / "fallback.md", title="\n ")
    assert fallback.title == "Sharper Analysis Report"


def test_report_rejects_html_directory_and_existing_file(tmp_path: Path) -> None:
    run = run_analysis(pd.DataFrame({"value": [1, 2]}))
    with pytest.raises(ValueError, match="only markdown reports"):
        generate_analysis_report(run, tmp_path / "report.html", format="html")
    with pytest.raises(ValueError, match="output_path must be a file path"):
        generate_analysis_report(run, tmp_path)

    path = tmp_path / "existing.md"
    path.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="output file already exists"):
        generate_analysis_report(run, path, overwrite=False)
    assert path.read_text(encoding="utf-8") == "keep"


def test_report_preserves_native_write_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = run_analysis(pd.DataFrame({"value": [1, 2]}))

    def fail_write(*args: object, **kwargs: object) -> int:
        raise OSError("disk unavailable")

    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(OSError, match="disk unavailable"):
        generate_analysis_report(run, tmp_path / "report.md")
