"""Permanent v0.1 compatibility manifest for the Task 20 transition."""

from __future__ import annotations

import inspect

import sharper

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
