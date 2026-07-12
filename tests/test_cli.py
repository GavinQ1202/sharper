# ruff: noqa: E501

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from matplotlib import pyplot as plt
from typer.testing import CliRunner

from sharper import cli
from sharper.cli import app
from sharper.reporting import ReportArtifact
from sharper.workflow import run_analysis

runner = CliRunner()


def test_help_and_version() -> None:
    assert runner.invoke(app, ["--help"]).exit_code == 0
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout == "sharper 0.1.0\n"


def test_csv_success_and_html_bundle(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    output = tmp_path / "report.html"
    pd.DataFrame({"x": [1.0, 2.0, 3.0], "g": ["a", "b", "a"]}).to_csv(
        source, index=False
    )
    result = runner.invoke(
        app, ["analyze", str(source), "-o", str(output), "--format", "html"]
    )
    assert result.exit_code == 0
    assert result.stdout == f"Report written to: {output}\n"
    assert (tmp_path / "report_assets").is_dir()


def test_cli_errors_exit_one_and_usage_exit_two(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["analyze", str(tmp_path / "missing.csv"), "-o", str(tmp_path / "x.md")]
    )
    assert result.exit_code == 1 and result.stdout == ""
    assert runner.invoke(app, ["analyze"]).exit_code == 2


def test_analyze_help_lists_task13_options() -> None:
    result = runner.invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    for option in ("--feature", "--time-column", "--group-by", "--model", "--debug"):
        assert option in result.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ["--version"],
        ["--version", "--help"],
        ["--help", "--version"],
        ["--version", "--unknown"],
        ["--unknown", "--version"],
        ["--version", "analyze", "missing.csv"],
    ],
)
def test_module_root_version_is_eager(arguments: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "sharper.cli", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == "sharper 0.1.0\n"
    assert result.stderr == ""


@pytest.mark.parametrize(
    "arguments", [["analyze", "--version"], ["analyze", "missing.csv", "--version"]]
)
def test_module_subcommand_version_is_a_usage_error(arguments: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "sharper.cli", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("mode", ["pre", "post", "success"])
def test_cli_figure_ownership_is_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    run = run_analysis(pd.DataFrame({"x": [1.0, 2.0, 3.0], "g": ["a", "b", "a"]}))
    plots = [
        *run.distribution_plots.plots,
        run.missingness_plot,
        run.correlation_plot,
        *run.outlier_plots.plots,
    ]
    owned = [plot.figure for plot in plots]
    unrelated = plt.figure()
    original_close = plt.close
    closed: list[object] = []

    def close(value: object = None) -> None:
        closed.append(value)
        original_close(value)

    def loader(path: Path) -> pd.DataFrame:
        return pd.DataFrame({"ignored": [1]})

    def workflow_once(frame: pd.DataFrame, **kwargs: object):
        return run

    def report(run_: object, output: Path, **kwargs: object) -> ReportArtifact:
        if mode == "pre":
            raise FileExistsError("output file or asset directory already exists")
        for figure in owned:
            if figure.number in plt.get_fignums():
                plt.close(figure)
        if mode == "post":
            error = OSError("failed to write report output")
            error._sharper_figures_consumed = True  # type: ignore[attr-defined]
            raise error
        return ReportArtifact(output, "markdown", "Sharper Analysis Report")

    monkeypatch.setattr(cli, "load_csv", loader)
    monkeypatch.setattr(cli, "run_analysis", workflow_once)
    monkeypatch.setattr(cli, "generate_analysis_report", report)
    monkeypatch.setattr(plt, "close", close)
    result = runner.invoke(
        app, ["analyze", str(tmp_path / "in.csv"), "-o", str(tmp_path / "out.md")]
    )
    assert result.exit_code == (0 if mode == "success" else 1)
    assert [value for value in closed if value in owned] == owned
    assert unrelated.number in plt.get_fignums()
    original_close(unrelated)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_cli_only_loads_then_calls_workflow_and_reporting_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str
) -> None:
    calls: list[str] = []
    run = run_analysis(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))

    def loader(path: Path) -> pd.DataFrame:
        calls.append("loader")
        return pd.DataFrame({"x": [1.0]})

    def workflow_once(frame: pd.DataFrame, **kwargs: object):
        calls.append("workflow")
        assert kwargs["task"] == task and kwargs["include_model"] is True
        return run

    def report_once(run_: object, output: Path, **kwargs: object) -> ReportArtifact:
        calls.append("reporting")
        return ReportArtifact(output, "markdown", "Sharper Analysis Report")

    monkeypatch.setattr(cli, "load_csv", loader)
    monkeypatch.setattr(cli, "run_analysis", workflow_once)
    monkeypatch.setattr(cli, "generate_analysis_report", report_once)
    result = runner.invoke(
        app,
        [
            "analyze",
            str(tmp_path / "in.csv"),
            "-o",
            str(tmp_path / "out.md"),
            "--target",
            "y",
            "--task",
            task,
            "--model",
        ],
    )
    assert result.exit_code == 0
    assert calls == ["loader", "workflow", "reporting"]
