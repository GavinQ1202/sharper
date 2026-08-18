"""Permanent v0.1 compatibility manifest for the Task 20 transition."""

from __future__ import annotations

import inspect
from pathlib import Path

import sharper

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_V01_COMPATIBILITY_EXPORTS = (
    "__version__",
    "load_csv",
    "load_excel",
    "ColumnSchema",
    "TargetCandidate",
    "SchemaReport",
    "infer_schema",
    "DataFrameSummary",
    "summarize_dataframe",
    "QualityIssue",
    "QualityReport",
    "check_data_quality",
    "AnalysisRun",
    "run_analysis",
    "ReportArtifact",
    "generate_analysis_report",
    "NumericAnalysis",
    "CategoricalAnalysis",
    "CorrelationAnalysis",
    "OutlierAnalysis",
    "analyze_numeric_features",
    "analyze_categorical_features",
    "compute_correlations",
    "detect_outliers",
    "GroupComparison",
    "TargetAnalysis",
    "compare_groups",
    "analyze_target_relationships",
    "FeatureSuggestion",
    "FeatureSuggestionReport",
    "FeatureDerivationResult",
    "suggest_feature_derivations",
    "derive_features",
    "TrainingResult",
    "train_classifier",
    "RegressionTrainingResult",
    "train_regressor",
    "ClassificationEvaluation",
    "evaluate_classifier",
    "RegressionEvaluation",
    "evaluate_regressor",
    "evaluate_model",
    "PlotResult",
    "PlotCollection",
    "plot_distributions",
    "plot_missingness",
    "plot_correlations",
    "plot_outliers",
    "plot_group_comparison",
    "plot_target_relationships",
    "plot_classification_evaluation",
    "plot_regression_evaluation",
)


def test_v01_compatibility_manifest_unchanged() -> None:
    """The complete v0.1 export prefix remains unchanged and available."""
    assert tuple(sharper.__all__[: len(_V01_COMPATIBILITY_EXPORTS)]) == (
        _V01_COMPATIBILITY_EXPORTS
    )
    assert all(hasattr(sharper, name) for name in _V01_COMPATIBILITY_EXPORTS)
    assert (
        inspect.signature(sharper.run_analysis).parameters["include_model"].default
        is False
    )
    assert (
        inspect.signature(sharper.generate_analysis_report).parameters["format"].default
        == "markdown"
    )
    assert (
        inspect.signature(sharper.generate_analysis_report)
        .parameters["overwrite"]
        .default
        is True
    )


def test_ci_matrix_contains_v02_gates() -> None:
    ci = (_PROJECT_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert 'python-version: ["3.10", "3.11", "3.12", "3.13"]' in ci
    assert "python -m pytest" in ci
    assert "tests/test_distribution.py" in ci
    assert "python -m ruff check ." in ci
    assert "python -m ruff format --check ." in ci
    assert "v02-run --help" in ci
    assert "v02_score_validation.py" in ci
    assert "v02_preloan.py" in ci
    assert "v02_postloan.py" in ci
    assert "v02_combined_report.py" in ci
    assert "v02_cli_json.py" in ci
    assert "matrix.python-version == '3.12'" in ci
    assert "uv venv .venv" in ci
    assert "prepare-distribution-offline-cache.py" in ci
    assert "SHARPER_DISTRIBUTION_OFFLINE_CACHE_ROOT" in ci
    assert "tests/test_v02_compatibility.py" in ci
    assert "actions/upload-artifact" not in ci
    assert "pypa/gh-action-pypi-publish" not in ci
    assert "action-gh-release" not in ci
    assert "git tag" not in ci
    assert "git push" not in ci


def test_v02_release_terminal_state_is_not_released() -> None:
    readiness = (_PROJECT_ROOT / "docs/release-readiness.md").read_text(
        encoding="utf-8"
    )
    assert "Release Ready — Not Released" in readiness

    workflow_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (_PROJECT_ROOT / ".github/workflows").glob("*.yml")
    )
    for forbidden in (
        "actions/upload-artifact",
        "pypa/gh-action-pypi-publish",
        "action-gh-release",
        "git tag",
        "git push",
        "twine upload",
        "deployment",
    ):
        assert forbidden not in workflow_text
