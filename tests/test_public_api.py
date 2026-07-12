"""Smoke tests for the public import contract."""

import inspect
from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Literal, get_type_hints

import pandas as pd
from sklearn.base import RegressorMixin

import sharper


def test_version_contract() -> None:
    """The package exposes its initial version."""
    assert sharper.__version__ == "0.1.0"


def test_console_entry_point_contract() -> None:
    """The sole frozen console entry point targets the public CLI app."""
    matches = [
        entry
        for entry in entry_points(group="console_scripts")
        if entry.name == "sharper"
    ]
    assert len(matches) == 1
    assert matches[0].value == "sharper.cli:app"


def test_all_contains_only_implemented_public_api() -> None:
    """The package exports only public APIs implemented through Task 11."""
    assert sharper.__all__ == [
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
    ]
    assert all(not name.startswith("_types") for name in sharper.__all__)


def test_load_csv_public_contract() -> None:
    """The public CSV loader has the documented signature and typing."""
    signature = inspect.signature(sharper.load_csv)
    assert list(signature.parameters) == ["path", "read_options"]
    assert signature.parameters["read_options"].kind is inspect.Parameter.VAR_KEYWORD

    hints = get_type_hints(sharper.load_csv)
    assert hints == {
        "path": str | Path,
        "read_options": Any,
        "return": pd.DataFrame,
    }
    assert sharper.load_csv.__doc__


def test_load_excel_public_contract() -> None:
    """The public Excel loader has the documented signature and typing."""
    signature = inspect.signature(sharper.load_excel)
    assert list(signature.parameters) == ["path", "sheet_name", "read_options"]
    assert signature.parameters["sheet_name"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["sheet_name"].default == 0
    assert signature.parameters["read_options"].kind is inspect.Parameter.VAR_KEYWORD

    hints = get_type_hints(sharper.load_excel)
    assert hints == {
        "path": str | Path,
        "sheet_name": str | int,
        "read_options": Any,
        "return": pd.DataFrame,
    }
    assert sharper.load_excel.__doc__


def test_task03_function_signatures_and_typing() -> None:
    """Schema and summary functions expose their frozen signatures."""
    schema_signature = inspect.signature(sharper.infer_schema)
    assert list(schema_signature.parameters) == ["df", "target", "id_threshold"]
    assert schema_signature.parameters["target"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        schema_signature.parameters["id_threshold"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert get_type_hints(sharper.infer_schema) == {
        "df": pd.DataFrame,
        "target": str | None,
        "id_threshold": float,
        "return": sharper.SchemaReport,
    }

    summary_signature = inspect.signature(sharper.summarize_dataframe)
    assert list(summary_signature.parameters) == ["df", "schema"]
    assert summary_signature.parameters["schema"].kind is inspect.Parameter.KEYWORD_ONLY
    assert get_type_hints(sharper.summarize_dataframe) == {
        "df": pd.DataFrame,
        "schema": sharper.SchemaReport | None,
        "return": sharper.DataFrameSummary,
    }
    assert sharper.infer_schema.__doc__
    assert sharper.summarize_dataframe.__doc__


def test_task03_dataclass_fields_are_frozen() -> None:
    """Public result dataclasses contain exactly the approved fields."""
    contracts = {
        sharper.ColumnSchema: [
            "name",
            "pandas_dtype",
            "logical_type",
            "nullable",
            "missing_count",
            "missing_rate",
            "unique_count",
            "unique_rate",
            "is_constant",
            "is_id_like",
            "confidence",
            "reasons",
        ],
        sharper.TargetCandidate: [
            "name",
            "suggested_task_type",
            "confidence",
            "reasons",
        ],
        sharper.SchemaReport: [
            "n_rows",
            "n_columns",
            "columns",
            "logical_type_counts",
            "target_candidates",
        ],
        sharper.DataFrameSummary: [
            "n_rows",
            "n_columns",
            "memory_usage_bytes",
            "total_missing_cells",
            "total_missing_rate",
            "schema",
            "column_summary",
        ],
    }

    for result_type, expected_fields in contracts.items():
        assert is_dataclass(result_type)
        assert result_type.__dataclass_params__.frozen is True
        assert [field.name for field in fields(result_type)] == expected_fields
        assert result_type.__doc__


def test_task04_function_signature_and_typing() -> None:
    """The quality function exposes its frozen signature and type hints."""
    signature = inspect.signature(sharper.check_data_quality)
    assert list(signature.parameters) == ["df", "schema", "missing_threshold"]
    assert signature.parameters["schema"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        signature.parameters["missing_threshold"].kind is inspect.Parameter.KEYWORD_ONLY
    )
    assert signature.parameters["schema"].default is None
    assert signature.parameters["missing_threshold"].default == 0.40
    assert get_type_hints(sharper.check_data_quality) == {
        "df": pd.DataFrame,
        "schema": sharper.SchemaReport | None,
        "missing_threshold": float,
        "return": sharper.QualityReport,
    }
    assert sharper.check_data_quality.__doc__


def test_task04_dataclass_fields_are_frozen() -> None:
    """Quality result dataclasses contain exactly the approved fields."""
    contracts = {
        sharper.QualityIssue: [
            "code",
            "severity",
            "scope",
            "column",
            "count",
            "ratio",
            "threshold",
            "message",
            "suggestion",
        ],
        sharper.QualityReport: [
            "n_rows",
            "n_columns",
            "issue_count",
            "severity_counts",
            "issues",
        ],
    }

    for result_type, expected_fields in contracts.items():
        assert is_dataclass(result_type)
        assert result_type.__dataclass_params__.frozen is True
        assert [field.name for field in fields(result_type)] == expected_fields
        assert result_type.__doc__


def test_task13_function_signatures_and_typing() -> None:
    """Workflow and reporting functions expose their frozen signatures."""
    run_signature = inspect.signature(sharper.run_analysis)
    assert list(run_signature.parameters) == [
        "df",
        "target",
        "task",
        "include_model",
        "id_columns",
        "exclude_columns",
        "features",
        "time_column",
        "group_by",
        "reference_date",
        "max_suggestions",
        "test_size",
        "random_state",
    ]
    assert get_type_hints(sharper.run_analysis)["return"] is sharper.AnalysisRun

    report_signature = inspect.signature(sharper.generate_analysis_report)
    assert list(report_signature.parameters) == [
        "run",
        "output_path",
        "title",
        "format",
        "overwrite",
    ]
    assert get_type_hints(sharper.generate_analysis_report) == {
        "run": sharper.AnalysisRun,
        "output_path": str | Path,
        "title": str,
        "format": Literal["markdown", "html"],
        "overwrite": bool,
        "return": sharper.ReportArtifact,
    }
    assert sharper.run_analysis.__doc__
    assert sharper.generate_analysis_report.__doc__


def test_task07_function_signatures_and_typing() -> None:
    """Non-target analysis functions expose their frozen signatures and hints."""
    contracts = {
        sharper.analyze_numeric_features: (
            ["df", "columns"],
            {"columns": None},
            {
                "df": pd.DataFrame,
                "columns": Sequence[str] | None,
                "return": sharper.NumericAnalysis,
            },
        ),
        sharper.analyze_categorical_features: (
            ["df", "columns", "top_n"],
            {"columns": None, "top_n": 10},
            {
                "df": pd.DataFrame,
                "columns": Sequence[str] | None,
                "top_n": int,
                "return": sharper.CategoricalAnalysis,
            },
        ),
        sharper.compute_correlations: (
            ["df", "columns", "method", "max_columns", "min_periods"],
            {
                "columns": None,
                "method": "pearson",
                "max_columns": 50,
                "min_periods": 2,
            },
            {
                "df": pd.DataFrame,
                "columns": Sequence[str] | None,
                "method": str,
                "max_columns": int,
                "min_periods": int,
                "return": sharper.CorrelationAnalysis,
            },
        ),
        sharper.detect_outliers: (
            ["df", "columns", "method", "threshold"],
            {"columns": None, "method": "iqr", "threshold": 1.5},
            {
                "df": pd.DataFrame,
                "columns": Sequence[str] | None,
                "method": str,
                "threshold": float,
                "return": sharper.OutlierAnalysis,
            },
        ),
    }

    for function, (names, defaults, expected_hints) in contracts.items():
        signature = inspect.signature(function)
        assert list(signature.parameters) == names
        assert (
            signature.parameters["df"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        for parameter, default in defaults.items():
            assert (
                signature.parameters[parameter].kind is inspect.Parameter.KEYWORD_ONLY
            )
            assert signature.parameters[parameter].default == default
        hints = get_type_hints(function)
        assert hints == expected_hints
        assert function.__doc__


def test_task07_dataclass_fields_are_frozen() -> None:
    """Task 07 result dataclasses contain exactly the frozen fields."""
    common_hints = {
        "n_rows": int,
        "requested_columns": tuple[str, ...] | None,
        "analyzed_columns": tuple[str, ...],
        "skipped_columns": tuple[str, ...],
        "skipped_reasons": dict[str, str],
    }
    contracts = {
        sharper.NumericAnalysis: {
            **common_hints,
            "summary": pd.DataFrame,
        },
        sharper.CategoricalAnalysis: {
            **common_hints,
            "top_n": int,
            "summary": pd.DataFrame,
            "top_categories": pd.DataFrame,
        },
        sharper.CorrelationAnalysis: {
            **common_hints,
            "method": str,
            "max_columns": int,
            "min_periods": int,
            "truncated": bool,
            "correlations": pd.DataFrame,
        },
        sharper.OutlierAnalysis: {
            **common_hints,
            "method": str,
            "threshold": float,
            "summary": pd.DataFrame,
            "outliers": pd.DataFrame,
        },
    }

    for result_type, expected_hints in contracts.items():
        assert is_dataclass(result_type)
        assert result_type.__dataclass_params__.frozen is True
        assert [field.name for field in fields(result_type)] == list(expected_hints)
        assert get_type_hints(result_type) == expected_hints
        assert result_type.__doc__


def test_task08_function_signatures_and_typing() -> None:
    """Group and target APIs expose exactly the frozen Task 08 signatures."""
    group_signature = inspect.signature(sharper.compare_groups)
    assert list(group_signature.parameters) == [
        "df",
        "group_by",
        "values",
        "max_groups",
    ]
    assert (
        group_signature.parameters["group_by"].kind
        is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    assert group_signature.parameters["values"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        group_signature.parameters["max_groups"].kind is inspect.Parameter.KEYWORD_ONLY
    )
    assert group_signature.parameters["values"].default is None
    assert group_signature.parameters["max_groups"].default == 20
    assert get_type_hints(sharper.compare_groups) == {
        "df": pd.DataFrame,
        "group_by": str,
        "values": Sequence[str] | None,
        "max_groups": int,
        "return": sharper.GroupComparison,
    }

    target_signature = inspect.signature(sharper.analyze_target_relationships)
    assert list(target_signature.parameters) == ["df", "target", "task", "features"]
    assert (
        target_signature.parameters["target"].kind
        is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    assert target_signature.parameters["task"].kind is inspect.Parameter.KEYWORD_ONLY
    assert target_signature.parameters["task"].default is inspect.Parameter.empty
    assert (
        target_signature.parameters["features"].kind is inspect.Parameter.KEYWORD_ONLY
    )
    assert target_signature.parameters["features"].default is None
    assert get_type_hints(sharper.analyze_target_relationships) == {
        "df": pd.DataFrame,
        "target": str,
        "task": Literal["classification", "regression"],
        "features": Sequence[str] | None,
        "return": sharper.TargetAnalysis,
    }
    assert sharper.compare_groups.__doc__
    assert sharper.analyze_target_relationships.__doc__


def test_task08_dataclass_fields_are_frozen() -> None:
    """Task 08 result dataclasses contain exactly the frozen fields and hints."""
    contracts = {
        sharper.GroupComparison: {
            "n_rows": int,
            "group_by": str,
            "requested_values": tuple[str, ...] | None,
            "analyzed_values": tuple[str, ...],
            "skipped_values": tuple[str, ...],
            "skipped_reasons": dict[str, str],
            "max_groups": int,
            "available_group_count": int,
            "displayed_group_count": int,
            "missing_group_count": int,
            "truncated": bool,
            "truncation_reason": str | None,
            "summary": pd.DataFrame,
        },
        sharper.TargetAnalysis: {
            "n_rows": int,
            "target": str,
            "task": str,
            "requested_features": tuple[str, ...] | None,
            "analyzed_features": tuple[str, ...],
            "skipped_features": tuple[str, ...],
            "skipped_reasons": dict[str, str],
            "max_features": int,
            "max_categories": int,
            "available_feature_count": int,
            "truncated": bool,
            "truncation_reason": str | None,
            "numeric_details": pd.DataFrame,
            "category_details": pd.DataFrame,
            "statistical_tests": pd.DataFrame,
            "limitations": tuple[str, ...],
        },
    }
    for result_type, expected_hints in contracts.items():
        assert is_dataclass(result_type)
        assert result_type.__dataclass_params__.frozen is True
        assert [field.name for field in fields(result_type)] == list(expected_hints)
        assert get_type_hints(result_type) == expected_hints
        assert result_type.__doc__


def test_task09_function_signatures_and_typing() -> None:
    """Feature APIs expose exactly the frozen Task 09 signatures."""
    suggest_signature = inspect.signature(sharper.suggest_feature_derivations)
    assert list(suggest_signature.parameters) == [
        "df",
        "schema",
        "target",
        "exclude_columns",
        "reference_date",
        "max_suggestions",
    ]
    for name in list(suggest_signature.parameters)[1:]:
        assert suggest_signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert get_type_hints(sharper.suggest_feature_derivations) == {
        "df": pd.DataFrame,
        "schema": sharper.SchemaReport | None,
        "target": str | None,
        "exclude_columns": Sequence[str],
        "reference_date": str | date | datetime | pd.Timestamp | None,
        "max_suggestions": int,
        "return": sharper.FeatureSuggestionReport,
    }

    derive_signature = inspect.signature(sharper.derive_features)
    assert list(derive_signature.parameters) == ["df", "suggestions", "copy"]
    assert derive_signature.parameters["copy"].kind is inspect.Parameter.KEYWORD_ONLY
    assert get_type_hints(sharper.derive_features) == {
        "df": pd.DataFrame,
        "suggestions": Sequence[sharper.FeatureSuggestion],
        "copy": bool,
        "return": sharper.FeatureDerivationResult,
    }
    assert sharper.suggest_feature_derivations.__doc__
    assert sharper.derive_features.__doc__


def test_task09_dataclass_fields_are_frozen() -> None:
    """Task 09 result dataclasses contain exactly the frozen fields and hints."""
    contracts = {
        sharper.FeatureSuggestion: {
            "name": str,
            "feature_type": str,
            "source_columns": tuple[str, ...],
            "formula": str | None,
            "parameters": tuple[tuple[str, str], ...],
            "reason": str,
            "risk": str,
            "requires_fit": bool,
            "priority": int,
        },
        sharper.FeatureSuggestionReport: {
            "n_rows": int,
            "requested_target": str | None,
            "requested_exclusions": tuple[str, ...],
            "reference_date": str | None,
            "eligible_columns": tuple[str, ...],
            "excluded_columns": tuple[str, ...],
            "skipped_columns": tuple[str, ...],
            "skipped_reasons": dict[str, str],
            "max_suggestions": int,
            "type_budgets": dict[str, int],
            "available_counts": dict[str, int],
            "available_suggestion_count": int,
            "truncated": bool,
            "truncation_reason": str | None,
            "suggestions": tuple[sharper.FeatureSuggestion, ...],
        },
        sharper.FeatureDerivationResult: {
            "data": pd.DataFrame,
            "applied_suggestions": tuple[str, ...],
            "skipped_suggestions": tuple[str, ...],
            "skipped_reasons": dict[str, str],
            "copy": bool,
        },
    }
    for result_type, expected_hints in contracts.items():
        assert is_dataclass(result_type)
        assert result_type.__dataclass_params__.frozen is True
        assert [field.name for field in fields(result_type)] == list(expected_hints)
        assert get_type_hints(result_type) == expected_hints
        assert result_type.__doc__


def test_task11_public_signatures_and_frozen_fields() -> None:
    """Task 11 exposes only its frozen classification contract."""
    training = inspect.signature(sharper.train_classifier)
    assert list(training.parameters) == [
        "df",
        "target",
        "features",
        "exclude_columns",
        "time_column",
        "estimator",
        "test_size",
        "random_state",
    ]
    assert all(
        training.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in list(training.parameters)[2:]
    )
    assert training.parameters["features"].default is None
    assert training.parameters["exclude_columns"].default == ()
    assert training.parameters["time_column"].default is None
    assert training.parameters["estimator"].default is None
    assert training.parameters["test_size"].default == 0.20
    assert training.parameters["random_state"].default == 42
    for function in (
        sharper.train_classifier,
        sharper.evaluate_classifier,
        sharper.evaluate_model,
        sharper.plot_classification_evaluation,
    ):
        assert function.__doc__
    assert [field.name for field in fields(sharper.TrainingResult)] == [
        "task",
        "target",
        "feature_columns",
        "excluded_columns",
        "time_column",
        "schema",
        "pipeline",
        "estimator",
        "classes",
        "train_row_positions",
        "test_row_positions",
        "X_test",
        "y_test",
        "test_size",
        "random_state",
        "warnings",
        "limitations",
    ]
    assert [field.name for field in fields(sharper.ClassificationEvaluation)] == [
        "task",
        "target",
        "holdout_positions",
        "classes",
        "y_true",
        "y_pred",
        "score_kind",
        "positive_label",
        "scores",
        "roc_curve",
        "metrics",
        "confusion_matrix",
        "roc_auc",
        "limitations",
    ]
    assert sharper.TrainingResult.__dataclass_params__.frozen is True
    assert sharper.ClassificationEvaluation.__dataclass_params__.frozen is True


def test_task12_public_signatures_and_frozen_fields() -> None:
    """Task 12 exposes its independent frozen regression contract."""
    signature = inspect.signature(sharper.train_regressor)
    assert list(signature.parameters) == [
        "df",
        "target",
        "features",
        "exclude_columns",
        "time_column",
        "estimator",
        "test_size",
        "random_state",
    ]
    assert all(
        signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in list(signature.parameters)[2:]
    )
    assert signature.parameters["features"].default is None
    assert signature.parameters["exclude_columns"].default == ()
    assert signature.parameters["time_column"].default is None
    assert signature.parameters["estimator"].default is None
    assert signature.parameters["test_size"].default == 0.20
    assert signature.parameters["random_state"].default == 42
    assert get_type_hints(sharper.train_regressor) == {
        "df": pd.DataFrame,
        "target": str,
        "features": Sequence[str] | None,
        "exclude_columns": Sequence[str],
        "time_column": str | None,
        "estimator": RegressorMixin | None,
        "test_size": float,
        "random_state": int | None,
        "return": sharper.RegressionTrainingResult,
    }
    assert [field.name for field in fields(sharper.RegressionTrainingResult)] == [
        "task",
        "target",
        "feature_columns",
        "excluded_columns",
        "time_column",
        "schema",
        "pipeline",
        "estimator",
        "train_row_positions",
        "test_row_positions",
        "X_test",
        "y_test",
        "test_size",
        "random_state",
        "warnings",
        "limitations",
    ]
    assert [field.name for field in fields(sharper.RegressionEvaluation)] == [
        "task",
        "target",
        "holdout_positions",
        "predictions",
        "metrics",
        "limitations",
    ]
    assert sharper.RegressionTrainingResult.__dataclass_params__.frozen is True
    assert sharper.RegressionEvaluation.__dataclass_params__.frozen is True
    for function in (
        sharper.train_regressor,
        sharper.evaluate_regressor,
        sharper.evaluate_model,
        sharper.plot_regression_evaluation,
    ):
        assert function.__doc__
