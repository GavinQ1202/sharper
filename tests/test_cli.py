"""Integration tests for the minimal Task 05 Typer CLI."""

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from sharper import generate_analysis_report, load_csv, run_analysis
from sharper.cli import app

runner = CliRunner()


def test_root_and_analyze_help() -> None:
    root = runner.invoke(app, ["--help"])
    assert root.exit_code == 0
    assert "analyze" in root.stdout

    analyze = runner.invoke(app, ["analyze", "--help"])
    assert analyze.exit_code == 0
    for text in (
        "INPUT",
        "--output",
        "--target",
        "--task",
        "--id-column",
        "--exclude-column",
        "--random-state",
        "--format",
        "--overwrite",
        "--no-overwrite",
    ):
        assert text in analyze.stdout


def test_analyze_success_and_python_section_consistency(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "cli.md"
    python_path = tmp_path / "python.md"
    pd.DataFrame({"value": [1, 2], "group": ["a", "b"]}).to_csv(input_path, index=False)

    result = runner.invoke(
        app,
        ["analyze", str(input_path), "--output", str(output_path)],
    )
    assert result.exit_code == 0
    assert result.stdout == f"Report written to: {output_path}\n"
    assert result.stderr == ""

    generate_analysis_report(run_analysis(load_csv(input_path)), python_path)
    cli_headings = [
        line for line in output_path.read_text(encoding="utf-8").splitlines() if line
    ]
    python_headings = [
        line for line in python_path.read_text(encoding="utf-8").splitlines() if line
    ]
    assert [line for line in cli_headings if line.startswith("#")] == [
        line for line in python_headings if line.startswith("#")
    ]


def test_analyze_runtime_errors_use_stderr_and_exit_one(tmp_path: Path) -> None:
    missing = runner.invoke(
        app,
        ["analyze", str(tmp_path / "missing.csv"), "-o", str(tmp_path / "out.md")],
    )
    assert missing.exit_code == 1
    assert missing.stdout == ""
    assert "Could not read CSV file" in missing.stderr

    input_path = tmp_path / "input.csv"
    pd.DataFrame({"value": [1]}).to_csv(input_path, index=False)
    invalid_format = runner.invoke(
        app,
        [
            "analyze",
            str(input_path),
            "-o",
            str(tmp_path / "out.html"),
            "--format",
            "html",
        ],
    )
    assert invalid_format.exit_code == 1
    assert invalid_format.stdout == ""
    assert "only markdown reports are supported in Task 05" in invalid_format.stderr

    output_path = tmp_path / "existing.md"
    output_path.write_text("keep", encoding="utf-8")
    no_overwrite = runner.invoke(
        app,
        [
            "analyze",
            str(input_path),
            "-o",
            str(output_path),
            "--no-overwrite",
        ],
    )
    assert no_overwrite.exit_code == 1
    assert no_overwrite.stdout == ""
    assert "output file already exists" in no_overwrite.stderr


def test_usage_error_exit_two() -> None:
    result = runner.invoke(app, ["analyze"])
    assert result.exit_code == 2
