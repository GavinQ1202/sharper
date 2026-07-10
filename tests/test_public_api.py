"""Smoke tests for the public import contract."""

import inspect
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import pandas as pd

import sharper


def test_version_contract() -> None:
    """The package exposes its initial version."""
    assert sharper.__version__ == "0.1.0"


def test_all_contains_only_implemented_public_api() -> None:
    """The package exports only APIs implemented through Task 05."""
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


def test_task05_function_signatures_and_typing() -> None:
    """Workflow and reporting functions expose their frozen signatures."""
    run_signature = inspect.signature(sharper.run_analysis)
    assert list(run_signature.parameters) == [
        "df",
        "target",
        "task",
        "include_model",
        "id_columns",
        "exclude_columns",
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
        "format": str,
        "overwrite": bool,
        "return": sharper.ReportArtifact,
    }
    assert sharper.run_analysis.__doc__
    assert sharper.generate_analysis_report.__doc__
