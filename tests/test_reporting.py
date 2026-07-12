# ruff: noqa: E501, E701

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import get_type_hints

import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from sharper import ReportArtifact, generate_analysis_report, reporting, run_analysis


def _run():
    return run_analysis(
        pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "g": ["a", "a", "b", "b"]})
    )


def _paths(path: Path) -> tuple[Path, Path, Path, Path, Path]:
    assets = path.parent / f"{path.stem}_assets"
    return (
        assets,
        path.parent / f".{path.name}.sharper-staging",
        path.parent / f".{assets.name}.sharper-staging",
        path.parent / f".{path.name}.sharper-backup",
        path.parent / f".{assets.name}.sharper-backup",
    )


def _old_bundle(path: Path) -> tuple[Path, Path, Path, Path, Path]:
    assets, report_stage, assets_stage, report_backup, assets_backup = _paths(path)
    path.write_text("old report", encoding="utf-8")
    assets.mkdir()
    (assets / "old.txt").write_text("old assets", encoding="utf-8")
    return assets, report_stage, assets_stage, report_backup, assets_backup


def _path_state(path: Path, *, assets: bool = False) -> str:
    if not path.exists():
        return "absent"
    if assets:
        marker = path / "old.txt"
        return (
            "old"
            if marker.exists() and marker.read_text(encoding="utf-8") == "old assets"
            else "new"
        )
    content = path.read_text(encoding="utf-8")
    return "old" if content == "old report" else "new"


def _assert_states(
    path: Path,
    expected: tuple[str, str, str, str, str, str],
) -> None:
    assets, report_stage, assets_stage, report_backup, assets_backup = _paths(path)
    actual = (
        _path_state(path),
        _path_state(assets, assets=True),
        _path_state(report_stage),
        _path_state(assets_stage, assets=True),
        _path_state(report_backup),
        _path_state(assets_backup, assets=True),
    )
    assert actual == expected


def _unique_figures(run: object) -> list[Figure]:
    values = (
        run.distribution_plots,
        run.missingness_plot,
        run.correlation_plot,
        run.outlier_plots,
        run.group_plots,
        run.target_plots,
        run.evaluation_plots,
    )
    figures: list[Figure] = []
    for value in values:
        if value is None:
            continue
        plots = value.plots if hasattr(value, "plots") else (value,)
        for plot in plots:
            if plot.figure not in figures:
                figures.append(plot.figure)
    return figures


def _spy_figure_closure(
    monkeypatch: pytest.MonkeyPatch, run: object
) -> tuple[list[Figure], Figure, list[object]]:
    figures = _unique_figures(run)
    unrelated = plt.figure()
    original_close = reporting.plt.close
    calls: list[object] = []

    def close(value: object = None) -> None:
        calls.append(value)
        original_close(value)

    monkeypatch.setattr(reporting.plt, "close", close)
    return figures, unrelated, calls


def test_markdown_bundle_sections_assets_and_figure_closure(tmp_path: Path) -> None:
    run = run_analysis(pd.DataFrame({"x": [1.0, 2.0, 3.0], "g": ["a", "b", "a"]}))
    artifact = generate_analysis_report(run, tmp_path / "report.md")
    content = artifact.path.read_text(encoding="utf-8")
    assert artifact == ReportArtifact(
        tmp_path / "report.md", "markdown", "Sharper Analysis Report"
    )
    assert (
        content.startswith("# Sharper Analysis Report") and "## Limitations" in content
    )
    assert (tmp_path / "report_assets" / "plot-001.png").exists()


def test_html_bundle_and_preacquisition_errors_do_not_write(tmp_path: Path) -> None:
    run = run_analysis(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    artifact = generate_analysis_report(run, tmp_path / "report.html", format="html")
    assert "<h1>Sharper Analysis Report</h1>" in artifact.path.read_text(
        encoding="utf-8"
    )
    with pytest.raises(TypeError, match=r"^run must be an AnalysisRun$"):
        generate_analysis_report(None, tmp_path / "bad.md")  # type: ignore[arg-type]
    assert not (tmp_path / "bad.md").exists()


def test_malformed_exact_run_is_schema_error(tmp_path: Path) -> None:
    run = run_analysis(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    bad = replace(run, evaluation=object())
    with pytest.raises(ValueError, match=r"^analysis run has invalid schema$"):
        generate_analysis_report(bad, tmp_path / "bad.md")


def test_target_analysis_without_model_generates_a_report(tmp_path: Path) -> None:
    run = run_analysis(
        pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [0, 0, 1, 1]}),
        target="y",
        task="classification",
    )
    artifact = generate_analysis_report(run, tmp_path / "target.md")
    assert artifact.path.exists()
    assert "## Target Relationships" in artifact.path.read_text(encoding="utf-8")


def test_rendering_tampering_is_schema_error_before_figure_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenCell:
        def __str__(self) -> str:
            raise RuntimeError("render fault")

    run = _run()
    figures, unrelated, closes = _spy_figure_closure(monkeypatch, run)
    summary = replace(run.summary, column_summary=pd.DataFrame({"bad": [BrokenCell()]}))
    with pytest.raises(
        ValueError, match=r"^analysis run has invalid schema$"
    ) as caught:
        generate_analysis_report(replace(run, summary=summary), tmp_path / "bad.md")
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert not list(tmp_path.iterdir())
    assert closes == []
    assert all(figure.number in plt.get_fignums() for figure in figures)
    plt.close(unrelated)
    for figure in figures:
        plt.close(figure)


def test_overwrite_and_stale_conflicts(tmp_path: Path) -> None:
    run = run_analysis(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    path = tmp_path / "report.md"
    path.write_text("old", encoding="utf-8")
    with pytest.raises(
        FileExistsError, match="output file or asset directory already exists"
    ):
        generate_analysis_report(run, path, overwrite=False)
    stale = tmp_path / ".report.md.sharper-staging"
    stale.write_text("stale", encoding="utf-8")
    with pytest.raises(FileExistsError, match="staging or backup path already exists"):
        generate_analysis_report(run, path)
    assert stale.read_text(encoding="utf-8") == "stale"


def test_report_artifact_is_frozen_and_typed(tmp_path: Path) -> None:
    assert get_type_hints(ReportArtifact) == {
        "path": Path,
        "format": str,
        "title": str,
    }
    artifact = generate_analysis_report(
        run_analysis(pd.DataFrame({"x": [1.0, 2.0]})), tmp_path / "report.md"
    )
    with pytest.raises(FrozenInstanceError):
        artifact.title = "changed"  # type: ignore[misc]


def test_title_cleanup_and_output_directory_validation(tmp_path: Path) -> None:
    run = run_analysis(pd.DataFrame({"x": [1.0, 2.0]}))
    artifact = generate_analysis_report(run, tmp_path / "a.md", title=" A\nB ")
    assert artifact.title == "A B"
    with pytest.raises(ValueError, match="output_path must be a file path"):
        generate_analysis_report(run, tmp_path)


@pytest.mark.parametrize("value", [None, {}, object()])
def test_wrong_run_types_do_not_create_output(tmp_path: Path, value: object) -> None:
    with pytest.raises(TypeError, match=r"^run must be an AnalysisRun$"):
        generate_analysis_report(value, tmp_path / "bad.md")  # type: ignore[arg-type]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "replacement",
    [
        {"task": "invalid"},
        {"warnings": (1,)},
        {"numeric_analysis": object()},
        {"distribution_plots": object()},
    ],
)
def test_preflight_tampering_is_stable_and_has_no_output(
    tmp_path: Path, replacement: dict[str, object]
) -> None:
    run = run_analysis(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    with pytest.raises(ValueError, match=r"^analysis run has invalid schema$"):
        generate_analysis_report(replace(run, **replacement), tmp_path / "bad.md")
    assert not list(tmp_path.iterdir())


def test_closed_figure_is_preflight_error(tmp_path: Path) -> None:
    from matplotlib import pyplot as plt

    run = run_analysis(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    figure = run.missingness_plot.figure
    plt.close(figure)
    with pytest.raises(ValueError, match=r"^analysis run has invalid schema$"):
        generate_analysis_report(run, tmp_path / "bad.md")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "failure", ["mkdir", "png-first", "png-middle", "png-last", "write", "flush"]
)
def test_staging_faults_restore_clean_state_and_close_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    path = tmp_path / "report.md"
    assets, report_stage, assets_stage, report_backup, assets_backup = _old_bundle(path)
    run = _run()
    figures = [plot.figure for plot in run.distribution_plots.plots]
    figures += [run.missingness_plot.figure, run.correlation_plot.figure]
    original_mkdir, original_savefig, original_open = (
        Path.mkdir,
        Figure.savefig,
        Path.open,
    )
    calls = 0

    def mkdir(self: Path, *args: object, **kwargs: object) -> None:
        if failure == "mkdir" and self == assets_stage:
            raise OSError("mkdir fault")
        original_mkdir(self, *args, **kwargs)

    def savefig(self: Figure, *args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        target = {"png-first": 1, "png-middle": 2, "png-last": len(figures)}.get(
            failure
        )
        if calls == target:
            raise RuntimeError("png fault")
        original_savefig(self, *args, **kwargs)

    class Handle:
        def __init__(self, handle: object) -> None:
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args: object):
            return self.handle.__exit__(*args)

        def write(self, text: str) -> object:
            if failure == "write":
                raise OSError("write fault")
            return self.handle.write(text)

        def flush(self) -> object:
            if failure == "flush":
                raise OSError("flush fault")
            return self.handle.flush()

    def open_(self: Path, *args: object, **kwargs: object):
        return (
            Handle(original_open(self, *args, **kwargs))
            if self == report_stage
            else original_open(self, *args, **kwargs)
        )

    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(Figure, "savefig", savefig)
    monkeypatch.setattr(Path, "open", open_)
    with pytest.raises(OSError, match=r"^failed to write report output$") as caught:
        generate_analysis_report(run, path)
    assert caught.value.__cause__ is not None
    assert path.read_text(encoding="utf-8") == "old report"
    assert (assets / "old.txt").read_text(encoding="utf-8") == "old assets"
    assert not any(
        p.exists() for p in (report_stage, assets_stage, report_backup, assets_backup)
    )
    assert all(
        figure.number not in __import__("matplotlib.pyplot").pyplot.get_fignums()
        for figure in figures
    )


@pytest.mark.parametrize(
    "failure", ["assets-backup", "report-backup", "assets-commit", "report-commit"]
)
def test_backup_and_commit_faults_restore_old_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    path = tmp_path / "report.md"
    assets, report_stage, assets_stage, report_backup, assets_backup = _old_bundle(path)
    run = _run()
    original = Path.replace

    def replace_(self: Path, target: Path) -> Path:
        if (
            (failure == "assets-backup" and self == assets and target == assets_backup)
            or (failure == "report-backup" and self == path and target == report_backup)
            or (
                failure == "assets-commit" and self == assets_stage and target == assets
            )
            or (failure == "report-commit" and self == report_stage and target == path)
        ):
            raise OSError(f"{failure} fault")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", replace_)
    with pytest.raises(OSError, match=r"^failed to write report output$") as caught:
        generate_analysis_report(run, path)
    assert isinstance(caught.value.__cause__, OSError)
    assert path.read_text(encoding="utf-8") == "old report"
    assert (assets / "old.txt").read_text(encoding="utf-8") == "old assets"
    assert not any(
        p.exists() for p in (report_stage, assets_stage, report_backup, assets_backup)
    )


@pytest.mark.parametrize("failure", ["report-cleanup", "assets-cleanup"])
def test_cleanup_faults_keep_committed_bundle_and_residual_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    path = tmp_path / "report.md"
    assets, report_stage, assets_stage, report_backup, assets_backup = _old_bundle(path)
    run = _run()
    original_remove = reporting._remove
    calls: list[Path] = []

    def remove(path_: Path) -> None:
        calls.append(path_)
        if (failure == "report-cleanup" and path_ == report_backup) or (
            failure == "assets-cleanup" and path_ == assets_backup
        ):
            raise OSError(f"{failure} fault")
        original_remove(path_)

    monkeypatch.setattr(reporting, "_remove", remove)
    with pytest.raises(OSError, match=r"^failed to write report output$") as caught:
        generate_analysis_report(run, path)
    assert isinstance(caught.value.__cause__, OSError)
    assert path.exists() and assets.exists()
    assert not report_stage.exists() and not assets_stage.exists()
    if failure == "report-cleanup":
        assert report_backup.exists() and assets_backup.exists()
        assert calls.count(assets_backup) == 0
    else:
        assert not report_backup.exists() and assets_backup.exists()
    with pytest.raises(
        FileExistsError, match=r"^staging or backup path already exists$"
    ):
        generate_analysis_report(_run(), path)


@pytest.mark.parametrize("phase", ["backup-restore", "commit-remove"])
def test_compensation_failure_preserves_current_recoverable_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    path = tmp_path / "report.md"
    assets, report_stage, assets_stage, report_backup, assets_backup = _old_bundle(path)
    run = _run()
    original_replace = Path.replace
    original_remove = reporting._remove
    compensation = OSError(f"{phase} compensation fault")

    def replace_(self: Path, target: Path) -> Path:
        if phase == "backup-restore" and self == path and target == report_backup:
            raise OSError("report backup fault")
        if phase == "commit-remove" and self == report_stage and target == path:
            raise OSError("report commit fault")
        return original_replace(self, target)

    def remove(path_: Path) -> None:
        if phase == "commit-remove" and path_ == assets:
            raise compensation
        original_remove(path_)

    def restore(source: Path, destination: Path) -> None:
        if phase == "backup-restore":
            raise compensation
        source.replace(destination)

    monkeypatch.setattr(Path, "replace", replace_)
    monkeypatch.setattr(reporting, "_remove", remove)
    monkeypatch.setattr(reporting, "_restore", restore)
    with pytest.raises(OSError, match=r"^failed to write report output$") as caught:
        generate_analysis_report(run, path)
    assert caught.value.__cause__ is compensation
    assert any(
        p.exists()
        for p in (
            path,
            assets,
            report_stage,
            assets_stage,
            report_backup,
            assets_backup,
        )
    )


def test_commit_report_restore_failure_stops_before_assets_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "report.md"
    assets, report_stage, assets_stage, report_backup, assets_backup = _old_bundle(path)
    run = _run()
    figures, unrelated, closes = _spy_figure_closure(monkeypatch, run)
    original_replace = Path.replace
    original_restore = reporting._restore
    transaction = PermissionError("report commit fault")
    compensation = PermissionError("report restore fault")
    actions: list[str] = []

    def replace_(source: Path, destination: Path) -> Path:
        if source == report_stage and destination == path:
            raise transaction
        return original_replace(source, destination)

    def restore(source: Path, destination: Path) -> None:
        actions.append(f"restore:{source.name}")
        if source == report_backup:
            raise compensation
        original_restore(source, destination)

    monkeypatch.setattr(Path, "replace", replace_)
    monkeypatch.setattr(reporting, "_restore", restore)
    with pytest.raises(OSError, match=r"^failed to write report output$") as caught:
        generate_analysis_report(run, path)
    assert caught.value.__cause__ is compensation
    assert actions == [f"restore:{report_backup.name}"]
    _assert_states(path, ("absent", "absent", "new", "absent", "old", "old"))
    assert closes == figures and unrelated.number in plt.get_fignums()
    plt.close(unrelated)


def test_commit_assets_restore_failure_keeps_report_restored_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "report.md"
    assets, report_stage, assets_stage, report_backup, assets_backup = _old_bundle(path)
    run = _run()
    figures, unrelated, closes = _spy_figure_closure(monkeypatch, run)
    original_replace = Path.replace
    original_restore = reporting._restore
    transaction = PermissionError("report commit fault")
    compensation = PermissionError("assets restore fault")
    actions: list[str] = []

    def replace_(source: Path, destination: Path) -> Path:
        if source == report_stage and destination == path:
            raise transaction
        return original_replace(source, destination)

    def restore(source: Path, destination: Path) -> None:
        actions.append(f"restore:{source.name}")
        if source == assets_backup:
            raise compensation
        original_restore(source, destination)

    monkeypatch.setattr(Path, "replace", replace_)
    monkeypatch.setattr(reporting, "_restore", restore)
    with pytest.raises(OSError, match=r"^failed to write report output$") as caught:
        generate_analysis_report(run, path)
    assert caught.value.__cause__ is compensation
    assert actions == [f"restore:{report_backup.name}", f"restore:{assets_backup.name}"]
    _assert_states(path, ("old", "absent", "new", "absent", "absent", "old"))
    assert closes == figures and unrelated.number in plt.get_fignums()
    plt.close(unrelated)


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("new-assets", ("absent", "new", "new", "absent", "old", "old")),
        ("report-staging", ("old", "old", "new", "absent", "absent", "absent")),
        ("assets-staging", ("old", "old", "absent", "absent", "absent", "absent")),
    ],
)
def test_commit_rollback_remove_failures_have_independent_state_matrices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected: tuple[str, str, str, str, str, str],
) -> None:
    path = tmp_path / "report.md"
    assets, report_stage, assets_stage, report_backup, assets_backup = _old_bundle(path)
    run = _run()
    figures, unrelated, closes = _spy_figure_closure(monkeypatch, run)
    original_replace = Path.replace
    original_remove = reporting._remove
    transaction = OSError("report commit fault")
    compensation = OSError(f"{failure} remove fault")
    calls: list[Path] = []

    def replace_(source: Path, destination: Path) -> Path:
        if source == report_stage and destination == path:
            raise transaction
        return original_replace(source, destination)

    def remove(value: Path) -> None:
        calls.append(value)
        target = {
            "new-assets": assets,
            "report-staging": report_stage,
            "assets-staging": assets_stage,
        }[failure]
        if value == target:
            raise compensation
        original_remove(value)

    monkeypatch.setattr(Path, "replace", replace_)
    monkeypatch.setattr(reporting, "_remove", remove)
    with pytest.raises(OSError, match=r"^failed to write report output$") as caught:
        generate_analysis_report(run, path)
    assert caught.value.__cause__ is compensation
    _assert_states(path, expected)
    assert (
        calls[-1]
        == {
            "new-assets": assets,
            "report-staging": report_stage,
            "assets-staging": assets_stage,
        }[failure]
    )
    assert closes == figures and unrelated.number in plt.get_fignums()
    plt.close(unrelated)


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("report-staging", ("old", "old", "new", "new", "absent", "absent")),
        ("assets-staging", ("old", "old", "absent", "new", "absent", "absent")),
    ],
)
def test_staging_cleanup_remove_failures_preserve_the_failed_staging_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected: tuple[str, str, str, str, str, str],
) -> None:
    path = tmp_path / "report.md"
    assets, report_stage, assets_stage, report_backup, assets_backup = _old_bundle(path)
    run = _run()
    figures, unrelated, closes = _spy_figure_closure(monkeypatch, run)
    original_open = Path.open
    original_remove = reporting._remove
    transaction = OSError("write fault")
    compensation = OSError(f"{failure} remove fault")

    class FailingWrite:
        def __init__(self, handle: object) -> None:
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args: object):
            return self.handle.__exit__(*args)

        def write(self, value: str) -> None:
            raise transaction

        def flush(self) -> None:
            self.handle.flush()

    def open_(value: Path, *args: object, **kwargs: object):
        if value == report_stage and args and args[0] == "w":
            return FailingWrite(original_open(value, *args, **kwargs))
        return original_open(value, *args, **kwargs)

    def remove(value: Path) -> None:
        if (
            value
            == {"report-staging": report_stage, "assets-staging": assets_stage}[failure]
        ):
            raise compensation
        original_remove(value)

    monkeypatch.setattr(Path, "open", open_)
    monkeypatch.setattr(reporting, "_remove", remove)
    with pytest.raises(OSError, match=r"^failed to write report output$") as caught:
        generate_analysis_report(run, path)
    assert caught.value.__cause__ is compensation
    _assert_states(path, expected)
    assert closes == figures and unrelated.number in plt.get_fignums()
    plt.close(unrelated)


def test_backup_failure_then_successful_restore_then_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "report.md"
    assets, report_stage, assets_stage, report_backup, assets_backup = _old_bundle(path)
    run = _run()
    original_replace = Path.replace
    original_remove = reporting._remove
    transaction = OSError("report backup fault")
    compensation = OSError("report staging cleanup fault")
    calls: list[str] = []

    def replace_(source: Path, destination: Path) -> Path:
        if source == path and destination == report_backup:
            raise transaction
        return original_replace(source, destination)

    def remove(value: Path) -> None:
        calls.append(value.name)
        if value == report_stage:
            raise compensation
        original_remove(value)

    monkeypatch.setattr(Path, "replace", replace_)
    monkeypatch.setattr(reporting, "_remove", remove)
    with pytest.raises(OSError, match=r"^failed to write report output$") as caught:
        generate_analysis_report(run, path)
    assert caught.value.__cause__ is compensation
    assert calls == [report_stage.name]
    _assert_states(path, ("old", "old", "new", "new", "absent", "absent"))


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_transaction_does_not_wrap_base_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exception: type[BaseException]
) -> None:
    path = tmp_path / "report.md"
    _, report_stage, _, _, _ = _old_bundle(path)
    run = _run()
    figures, unrelated, closes = _spy_figure_closure(monkeypatch, run)
    original_replace = Path.replace

    def replace_(source: Path, destination: Path) -> Path:
        if source == report_stage and destination == path:
            raise exception()
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", replace_)
    with pytest.raises(exception):
        generate_analysis_report(run, path)
    assert closes == figures and unrelated.number in plt.get_fignums()
    plt.close(unrelated)


@pytest.mark.parametrize("format", ["markdown", "html"])
def test_reporting_never_recomputes_upstream_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, format: str
) -> None:
    """A valid run can be rendered after all computation entry points are poisoned."""
    import sharper.analysis as analysis
    import sharper.evaluation as evaluation
    import sharper.features as features
    import sharper.modeling as modeling
    import sharper.quality as quality
    import sharper.schema as schema
    import sharper.summary as summary
    import sharper.visualization as visualization

    run = _run()

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("reporting recomputed an upstream result")

    for module, names in (
        (schema, ("infer_schema",)),
        (summary, ("summarize_dataframe",)),
        (quality, ("check_data_quality",)),
        (
            analysis,
            (
                "analyze_numeric_features",
                "analyze_categorical_features",
                "compute_correlations",
                "detect_outliers",
                "compare_groups",
                "analyze_target_relationships",
            ),
        ),
        (features, ("suggest_feature_derivations", "derive_features")),
        (modeling, ("train_classifier", "train_regressor")),
        (evaluation, ("evaluate_model",)),
        (
            visualization,
            (
                "plot_distributions",
                "plot_missingness",
                "plot_correlations",
                "plot_outliers",
                "plot_group_comparison",
                "plot_target_relationships",
                "plot_classification_evaluation",
                "plot_regression_evaluation",
            ),
        ),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    artifact = generate_analysis_report(
        run, tmp_path / f"report.{format}", format=format
    )  # type: ignore[arg-type]
    assert artifact.path.exists()
