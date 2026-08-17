"""Task 20 Wave I3 static reporting contract tests."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

import sharper.v02_reporting as reporting
from sharper.risk_validation import BinaryRiskValidationResult
from sharper.v02_workflow import V02WorkflowResult


def _empty_instance(cls):
    instance = object.__new__(cls)
    for field in fields(cls):
        object.__setattr__(instance, field.name, None)
    return instance


def _empty_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _owner(cls, tables: dict[str, tuple[str, ...]]):
    instance = _empty_instance(cls)
    for name, columns in tables.items():
        object.__setattr__(instance, name, _empty_frame(columns))
    object.__setattr__(instance, "warnings", ())
    object.__setattr__(instance, "limitations", ())
    return instance


def _result(*, enabled: bool = False, score=None) -> V02WorkflowResult:
    owners = {
        "score_validation": score if enabled else None,
        "audit": None,
        "preloan": None,
        "postloan": None,
        "governance": None,
    }
    enabled_paths = ("score_validation",) if enabled else ()
    path_status = pd.DataFrame(
        {
            "path_key": list(owners),
            "enabled": [owners[key] is not None for key in owners],
            "status": [
                "enabled" if owners[key] is not None else "disabled" for key in owners
            ],
            "reason": [pd.NA] * len(owners),
        }
    )
    return V02WorkflowResult(
        "task20-integration-v1",
        enabled_paths,
        path_status,
        ("score_validation",) if enabled else (),
        owners["score_validation"],
        owners["audit"],
        owners["preloan"],
        owners["postloan"],
        owners["governance"],
        ("warning:synthetic",),
        ("limitation:synthetic",),
    )


def _empty_score() -> BinaryRiskValidationResult:
    return _owner(
        BinaryRiskValidationResult,
        {
            "metrics": reporting._METRICS_COLUMNS,
            "business_metrics": reporting._BUSINESS_COLUMNS,
            "gains": reporting._GAINS_PLOT_COLUMNS,
            "calibration": reporting._CALIBRATION_PLOT_COLUMNS,
            "threshold_analysis": reporting._THRESHOLD_PLOT_COLUMNS,
            "provenance": reporting._OWNER_PROVENANCE_COLUMNS,
        },
    )


def test_v02_report_sections_and_semantic_parity(tmp_path: Path) -> None:
    result = _result(enabled=True, score=_empty_score())
    markdown_path = tmp_path / "integration.md"
    html_path = tmp_path / "integration.html"

    markdown = reporting.generate_v02_report(result, markdown_path).path.read_text()
    html = reporting.generate_v02_report(
        result, html_path, format="html"
    ).path.read_text()

    titles = (
        "Run Context",
        "Path Status",
        "Score Validation",
        "Data Audit and Leakage",
        "Pre-loan Eligibility",
        "Post-loan Warning",
        "Governance",
        "Cross-path Comparison",
        "Reason and Override Trace",
        "Stability and Business Evidence",
        "Warnings and Limitations",
        "Provenance and Release Readiness",
    )
    assert [markdown.index(title) for title in titles] == sorted(
        markdown.index(title) for title in titles
    )
    assert [html.index(title) for title in titles] == sorted(
        html.index(title) for title in titles
    )
    assert "empty_result:score_validation" in markdown
    assert "empty_result:score_validation" in html
    assert "warning:synthetic" in markdown and "warning:synthetic" in html
    assert "limitation:synthetic" in markdown and "limitation:synthetic" in html
    assert "<script" not in markdown
    assert "<script" not in html
    assert (tmp_path / "integration_assets").is_dir()


def test_v02_report_consumes_result_only(tmp_path: Path, monkeypatch) -> None:
    result = _result()
    monkeypatch.setattr(
        reporting,
        "plot_binary_risk_validation",
        lambda *args, **kwargs: pytest.fail("disabled result-only plot was called"),
    )
    artifact = reporting.generate_v02_report(result, tmp_path / "disabled.md")
    content = artifact.path.read_text()
    assert content.count("not_requested") == 8

    with pytest.raises(ValueError, match=r"^sharper task20: report_result$"):
        reporting.generate_v02_report(pd.DataFrame(), tmp_path / "invalid.md")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"^sharper task20: report_title$"):
        reporting.generate_v02_report(result, tmp_path / "title.md", title=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"^sharper task20: report_format$"):
        reporting.generate_v02_report(result, tmp_path / "format.md", format="pdf")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"^sharper task20: report_path$"):
        reporting.generate_v02_report(result, tmp_path, format="markdown")
    with pytest.raises(ValueError, match=r"^sharper task20: report_overwrite$"):
        reporting.generate_v02_report(result, tmp_path / "overwrite.md", overwrite=1)  # type: ignore[arg-type]


def test_v02_report_asset_budget_is_checked_before_figure_creation(
    tmp_path: Path, monkeypatch
) -> None:
    score = _empty_score()
    object.__setattr__(
        score,
        "gains",
        pd.DataFrame(
            [
                {
                    "scope": "overall",
                    "fold_id": pd.NA,
                    "actual_fraction": 0.5,
                    "capture": 0.5,
                    "capture_status": "available",
                    "capture_reason": pd.NA,
                    "lift": pd.NA,
                    "lift_status": "undefined",
                    "lift_reason": "unavailable",
                }
            ],
            columns=reporting._GAINS_PLOT_COLUMNS,
        ),
    )
    monkeypatch.setattr(reporting, "_REPORT_FIGURE_LIMIT", 0)
    monkeypatch.setattr(
        reporting,
        "plot_binary_risk_validation",
        lambda *args, **kwargs: pytest.fail("budget gate acquired a Figure"),
    )
    with pytest.raises(ValueError, match=r"^sharper task20: report_asset_budget$"):
        reporting.generate_v02_report(
            _result(enabled=True, score=score), tmp_path / "budget.md"
        )


def test_v02_figure_close_asset_order_and_rollback(tmp_path: Path, monkeypatch) -> None:
    score = _empty_score()
    gains = pd.DataFrame(
        [
            {
                "scope": "overall",
                "fold_id": pd.NA,
                "actual_fraction": 0.5,
                "capture_status": "available",
                "capture_reason": pd.NA,
                "capture": 0.5,
                "lift_status": "undefined",
                "lift_reason": "unavailable",
                "lift": pd.NA,
            }
        ],
        columns=reporting._GAINS_PLOT_COLUMNS,
    )
    object.__setattr__(score, "gains", gains)
    result = _result(enabled=True, score=score)
    figure = Figure()
    calls: list[str] = []

    def plot(*args, **kwargs):
        calls.append(kwargs["kind"])
        return figure

    monkeypatch.setattr(reporting, "plot_binary_risk_validation", plot)
    closed: list[int] = []
    original_close = reporting.plt.close
    monkeypatch.setattr(
        reporting.plt,
        "close",
        lambda value=None: (closed.append(id(value)), original_close(value))[1],
    )

    def fail_savefig(self, path, *args, **kwargs):
        calls.append(Path(path).name)
        raise OSError("synthetic png failure")

    monkeypatch.setattr(Figure, "savefig", fail_savefig)
    output = tmp_path / "rollback.md"
    with pytest.raises(OSError, match="failed to write report output"):
        reporting.generate_v02_report(result, output)

    assert calls == ["gains", "plot-001.png"]
    assert closed == [id(figure)]
    assert not output.exists()
    assert not (tmp_path / "rollback_assets").exists()
    assert not (tmp_path / ".rollback.md.sharper-staging").exists()
    assert not (tmp_path / ".rollback_assets.sharper-staging").exists()
