"""Task 20 Wave I3 static reporting contract tests."""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import sharper.v02_reporting as reporting
from sharper.decision_strategy import DecisionStrategyConfig
from sharper.risk_validation import BinaryRiskValidationResult
from sharper.v02_workflow import (
    V02PreLoanRequest,
    V02WorkflowRequest,
    V02WorkflowResult,
    run_v02_workflow,
)


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


def _real_preloan_result() -> tuple[pd.DataFrame, V02WorkflowResult]:
    data = pd.DataFrame({"x": [1, 2]})
    config = DecisionStrategyConfig(
        "a3",
        "v1",
        datetime(2025, 1, 1),
        None,
        datetime(2025, 1, 2),
        (),
        "review",
        "review",
        (("review", "review"),),
    )
    result = run_v02_workflow(
        V02WorkflowRequest(data, preloan=V02PreLoanRequest(config))
    )
    return data, result


def test_v02_report_accepts_real_workflow_path_status(tmp_path: Path) -> None:
    data, result = _real_preloan_result()
    data_before = data.copy(deep=True)
    path_status_before = result.path_status.copy(deep=True)

    artifact = reporting.generate_v02_report(result, tmp_path / "real.md")

    assert artifact.path.is_file()
    assert "Path Status" in artifact.path.read_text()
    pd.testing.assert_frame_equal(data, data_before)
    pd.testing.assert_frame_equal(result.path_status, path_status_before)


@pytest.mark.parametrize("invalid", [0, 1.0, "true", None, pd.NA, np.int64(1)])
def test_v02_report_rejects_invalid_path_status_enabled_scalar(
    tmp_path: Path, invalid: object
) -> None:
    _, result = _real_preloan_result()
    path_status = result.path_status.copy(deep=True)
    path_status["enabled"] = pd.Series(
        [invalid, False, False, False, False], dtype="object"
    )
    invalid_result = replace(result, path_status=path_status)

    with pytest.raises(ValueError, match=r"^sharper task20: report_result$"):
        reporting.generate_v02_report(invalid_result, tmp_path / "invalid.md")


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


def _enabled_slots(count: int) -> tuple[reporting._Slot, ...]:
    owners = ("risk", "governance")
    return tuple(
        reporting._Slot(
            ordinal,
            f"synthetic_plot_{ordinal}",
            owners[(ordinal - 1) % len(owners)],
            "synthetic",
            "Score Validation" if ordinal == 1 else "Governance",
            True,
        )
        for ordinal in range(1, count + 1)
    )


def test_v02_report_figure_entry_budget_max_and_max_plus_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _result(enabled=True, score=_empty_score())
    figures: list[Figure] = []

    def plot(*args, **kwargs) -> Figure:
        figure = Figure()
        figures.append(figure)
        return figure

    monkeypatch.setattr(reporting, "plot_binary_risk_validation", plot)
    monkeypatch.setattr(reporting, "plot_model_governance", plot)
    monkeypatch.setattr(reporting, "_slot_plan", lambda *args: _enabled_slots(9))

    artifact = reporting.generate_v02_report(result, tmp_path / "nine.md")
    assert artifact.path.is_file()
    assert len(figures) == 9
    assert len(list((tmp_path / "nine_assets").glob("plot-*.png"))) == 9

    monkeypatch.setattr(reporting, "_slot_plan", lambda *args: _enabled_slots(10))
    monkeypatch.setattr(
        reporting,
        "plot_binary_risk_validation",
        lambda *args, **kwargs: pytest.fail("figure-entry budget acquired a Figure"),
    )
    monkeypatch.setattr(
        reporting,
        "plot_model_governance",
        lambda *args, **kwargs: pytest.fail("figure-entry budget acquired a Figure"),
    )
    with pytest.raises(ValueError, match=r"^sharper task20: report_asset_budget$"):
        reporting.generate_v02_report(result, tmp_path / "ten.md")
    assert not (tmp_path / "ten.md").exists()
    assert not (tmp_path / "ten_assets").exists()
    assert not (tmp_path / ".ten.md.sharper-staging").exists()
    assert not (tmp_path / ".ten_assets.sharper-staging").exists()


@pytest.mark.parametrize("observed_size", [64_000_000, 64_000_001])
def test_v02_report_png_bytes_budget_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observed_size: int,
) -> None:
    result = _result()
    slot = reporting._Slot(
        1, "synthetic_png", "risk", "synthetic", "Score Validation", True
    )
    figure = Figure()
    closed: list[int] = []
    monkeypatch.setattr(reporting, "_slot_plan", lambda *args: (slot,))
    monkeypatch.setattr(
        reporting, "plot_binary_risk_validation", lambda *args, **kwargs: figure
    )
    original_close = reporting.plt.close
    monkeypatch.setattr(
        reporting.plt,
        "close",
        lambda value=None: (closed.append(id(value)), original_close(value))[1],
    )
    original_stat = Path.stat
    observed = False

    def observed_stat(self: Path, *, follow_symlinks: bool = True):
        nonlocal observed
        if self.name == "plot-001.png" and not observed:
            observed = True
            return SimpleNamespace(st_size=observed_size)
        return original_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", observed_stat)
    output = tmp_path / f"png-{observed_size}.md"

    if observed_size == 64_000_000:
        artifact = reporting.generate_v02_report(result, output)
        assert artifact.path.is_file()
        assert (tmp_path / f"png-{observed_size}_assets" / "plot-001.png").is_file()
    else:
        with pytest.raises(ValueError, match=r"^sharper task20: report_asset_budget$"):
            reporting.generate_v02_report(result, output)
        assert not output.exists()
        assert not (tmp_path / f"png-{observed_size}_assets").exists()
        assert not (tmp_path / f".png-{observed_size}.md.sharper-staging").exists()
        assert not (tmp_path / f".png-{observed_size}_assets.sharper-staging").exists()
    assert closed == [id(figure)]


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
