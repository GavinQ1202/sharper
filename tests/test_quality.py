"""Contract and rule tests for minimal data-quality reporting."""

from dataclasses import FrozenInstanceError, fields, is_dataclass

import numpy as np
import pandas as pd
import pytest

from sharper import (
    QualityIssue,
    QualityReport,
    check_data_quality,
    infer_schema,
)
from sharper.schema import SchemaReport


def _issues_by_code(report: QualityReport, code: str) -> list[QualityIssue]:
    return [issue for issue in report.issues if issue.code == code]


def test_quality_result_contracts_are_frozen() -> None:
    """Public quality results expose exactly the frozen Task 04 fields."""
    assert is_dataclass(QualityIssue)
    assert QualityIssue.__dataclass_params__.frozen is True
    assert [field.name for field in fields(QualityIssue)] == [
        "code",
        "severity",
        "scope",
        "column",
        "count",
        "ratio",
        "threshold",
        "message",
        "suggestion",
    ]
    assert [field.type for field in fields(QualityIssue)] == [
        "str",
        "str",
        "str",
        "str | None",
        "int | None",
        "float | None",
        "float | None",
        "str",
        "str",
    ]

    assert is_dataclass(QualityReport)
    assert QualityReport.__dataclass_params__.frozen is True
    assert [field.name for field in fields(QualityReport)] == [
        "n_rows",
        "n_columns",
        "issue_count",
        "severity_counts",
        "issues",
    ]
    assert [field.type for field in fields(QualityReport)] == [
        "int",
        "int",
        "int",
        "dict[str, int]",
        "list[QualityIssue]",
    ]

    report = check_data_quality(pd.DataFrame({"value": [1, 2]}))
    with pytest.raises(FrozenInstanceError):
        report.issue_count = 1  # type: ignore[misc]


def test_no_issue_report_has_all_zero_severity_keys() -> None:
    """A clean frame returns the complete deterministic empty report shape."""
    report = check_data_quality(pd.DataFrame({"value": [1.0, 2.0, 3.0]}))

    assert report.issues == []
    assert report.issue_count == 0
    assert report.severity_counts == {"info": 0, "warning": 0, "error": 0}
    assert list(report.severity_counts) == ["info", "warning", "error"]


@pytest.mark.parametrize(
    ("frame", "severity", "message", "n_columns"),
    [
        (
            pd.DataFrame(),
            "error",
            "DataFrame has no rows or columns",
            0,
        ),
        (
            pd.DataFrame({"value": pd.Series(dtype="float64")}),
            "warning",
            "DataFrame has no rows",
            1,
        ),
    ],
)
def test_empty_frames_only_report_empty_dataframe(
    frame: pd.DataFrame,
    severity: str,
    message: str,
    n_columns: int,
) -> None:
    """Both frozen zero-row cases produce only their table-level issue."""
    report = check_data_quality(frame)

    assert report.n_rows == 0
    assert report.n_columns == n_columns
    assert report.issues == [
        QualityIssue(
            code="empty_dataframe",
            severity=severity,
            scope="table",
            column=None,
            count=0,
            ratio=None,
            threshold=None,
            message=message,
            suggestion="Provide data rows before running analysis",
        )
    ]


def test_duplicate_rows_use_keep_false_and_pandas_nan_semantics() -> None:
    """All members of duplicate groups count, including rows containing NaN."""
    frame = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b", "c"],
            "value": [1.0, 1.0, np.nan, np.nan, 3.0],
        }
    )
    report = check_data_quality(frame)

    assert _issues_by_code(report, "duplicate_rows") == [
        QualityIssue(
            code="duplicate_rows",
            severity="warning",
            scope="table",
            column=None,
            count=4,
            ratio=0.8,
            threshold=None,
            message="Duplicate rows detected",
            suggestion=(
                "Review duplicate rows and decide whether they should be removed "
                "or consolidated"
            ),
        )
    ]


def test_all_missing_and_high_missing_are_mutually_exclusive() -> None:
    """All-missing columns do not also emit the threshold-based missing issue."""
    frame = pd.DataFrame(
        {
            "all_missing": [None, None, None, None],
            "high_missing": [1.0, None, None, 2.0],
            "complete": [1, 2, 3, 4],
        }
    )
    report = check_data_quality(frame, missing_threshold=0.50)

    all_missing = _issues_by_code(report, "all_missing_column")
    assert [
        (issue.column, issue.count, issue.ratio, issue.threshold)
        for issue in all_missing
    ] == [("all_missing", 4, 1.0, None)]
    high_missing = _issues_by_code(report, "high_missing_column")
    assert [
        (issue.column, issue.count, issue.ratio, issue.threshold)
        for issue in high_missing
    ] == [("high_missing", 2, 0.5, 0.5)]


def test_missing_threshold_is_inclusive() -> None:
    """A missing rate exactly at the configured threshold emits an issue."""
    frame = pd.DataFrame({"value": [1.0, None, 2.0, None]})

    at_boundary = check_data_quality(frame, missing_threshold=0.50)
    above_boundary = check_data_quality(frame, missing_threshold=0.51)

    assert len(_issues_by_code(at_boundary, "high_missing_column")) == 1
    assert _issues_by_code(above_boundary, "high_missing_column") == []


def test_constant_and_near_constant_are_mutually_exclusive() -> None:
    """Constant columns never also emit near-constant issues."""
    frame = pd.DataFrame(
        {
            "constant": [7] * 20,
            "near": ["common"] * 19 + ["rare"],
        }
    )
    report = check_data_quality(frame)

    constant = _issues_by_code(report, "constant_column")
    assert [(issue.column, issue.count, issue.ratio) for issue in constant] == [
        ("constant", 1, 1.0)
    ]
    near = _issues_by_code(report, "near_constant_column")
    assert [
        (issue.column, issue.count, issue.ratio, issue.threshold) for issue in near
    ] == [("near", 19, 0.95, 0.95)]


def test_all_missing_is_not_constant() -> None:
    """Task 03 all-missing semantics prevent a constant issue."""
    report = check_data_quality(pd.DataFrame({"value": [None, None]}))

    assert len(_issues_by_code(report, "all_missing_column")) == 1
    assert _issues_by_code(report, "constant_column") == []


def test_high_cardinality_only_applies_to_categorical_schema() -> None:
    """A categorical dtype can cross both frozen high-cardinality thresholds."""
    values = [f"category-{index}" for index in range(60)]
    frame = pd.DataFrame({"category": pd.Categorical(values)})
    report = check_data_quality(frame)

    issues = _issues_by_code(report, "high_cardinality_categorical")
    assert [
        (issue.column, issue.count, issue.ratio, issue.threshold) for issue in issues
    ] == [("category", 60, 1.0, 0.50)]

    text_report = check_data_quality(pd.DataFrame({"text": values}))
    assert _issues_by_code(text_report, "high_cardinality_categorical") == []


def test_identifier_is_reported_without_high_cardinality_categorical() -> None:
    """Identifier-like columns emit only their approved identity issue."""
    frame = pd.DataFrame({"record_id": [f"id-{index}" for index in range(60)]})
    report = check_data_quality(frame)

    identifier = _issues_by_code(report, "identifier_like_column")
    assert [
        (issue.column, issue.count, issue.ratio, issue.threshold)
        for issue in identifier
    ] == [("record_id", 60, 1.0, None)]
    assert _issues_by_code(report, "high_cardinality_categorical") == []


def test_numeric_infinite_values_exclude_boolean() -> None:
    """Positive and negative infinity count without broadening to outliers."""
    frame = pd.DataFrame(
        {
            "numeric": [1.0, np.inf, -np.inf, np.nan],
            "boolean": [True, False, True, False],
        }
    )
    report = check_data_quality(frame)

    issues = _issues_by_code(report, "infinite_values")
    assert [
        (issue.column, issue.count, issue.ratio, issue.threshold) for issue in issues
    ] == [("numeric", 2, 0.5, None)]


def test_mixed_python_types_use_task03_schema_reason() -> None:
    """Mixed object reporting is tied to the frozen schema reason code."""
    frame = pd.DataFrame({"mixed": pd.Series(["text", 2, None], dtype="object")})
    report = check_data_quality(frame)

    assert _issues_by_code(report, "mixed_python_types") == [
        QualityIssue(
            code="mixed_python_types",
            severity="warning",
            scope="column",
            column="mixed",
            count=2,
            ratio=pytest.approx(2 / 3),
            threshold=None,
            message="Column contains mixed Python value types",
            suggestion="Standardize the column values before analysis",
        )
    ]


@pytest.mark.parametrize(
    "series",
    [
        pd.Series(["2025-01-01", "not-a-date", "still-not-a-date"], dtype="object"),
        pd.Series(["2025-01-01", "not-a-date", "still-not-a-date"], dtype="string"),
        pd.Series(pd.Categorical(["2025-01-01", "not-a-date", "still-not-a-date"])),
    ],
)
def test_partial_datetime_parse_failures_for_supported_dtypes(
    series: pd.Series,
) -> None:
    """Object, string, and category columns report partial parse failures."""
    report = check_data_quality(pd.DataFrame({"date_value": series}))

    issues = _issues_by_code(report, "datetime_parse_failures")
    assert [
        (issue.column, issue.count, issue.ratio, issue.threshold) for issue in issues
    ] == [("date_value", 2, pytest.approx(2 / 3), None)]


def test_schema_datetime_columns_do_not_report_parse_failures() -> None:
    """Meeting Task 03's datetime threshold suppresses parse-failure reporting."""
    frame = pd.DataFrame(
        {
            "date_value": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "2025-01-04",
                "bad",
            ]
        }
    )
    report = check_data_quality(frame)

    assert infer_schema(frame).columns[0].logical_type == "datetime"
    assert _issues_by_code(report, "datetime_parse_failures") == []


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.01])
def test_invalid_missing_threshold_raises(threshold: float) -> None:
    """The public threshold is restricted to the frozen interval."""
    with pytest.raises(ValueError, match="missing_threshold must be > 0 and <= 1"):
        check_data_quality(pd.DataFrame({"value": [1]}), missing_threshold=threshold)


@pytest.mark.parametrize("threshold", [0.01, 1.0])
def test_valid_missing_threshold_boundaries_are_accepted(threshold: float) -> None:
    """Positive thresholds through one are valid."""
    report = check_data_quality(
        pd.DataFrame({"value": [1.0, None]}),
        missing_threshold=threshold,
    )
    assert isinstance(report, QualityReport)


def test_invalid_column_names_raise() -> None:
    """Task 04 inherits Task 03's non-string and duplicate-name validation."""
    with pytest.raises(
        ValueError,
        match="DataFrame column names must all be strings",
    ):
        check_data_quality(pd.DataFrame([[1]], columns=[1]))

    with pytest.raises(ValueError, match="DataFrame column names must be unique"):
        check_data_quality(pd.DataFrame([[1, 2]], columns=["value", "value"]))


@pytest.mark.parametrize(
    "schema",
    [
        SchemaReport(1, 1, [], {}, []),
        infer_schema(pd.DataFrame({"other": [1]})),
        infer_schema(pd.DataFrame({"value": [1, 2]})),
    ],
)
def test_mismatched_schema_raises(schema: SchemaReport) -> None:
    """Shape, names, and ordered schema columns must all match."""
    with pytest.raises(ValueError, match="schema does not match DataFrame"):
        check_data_quality(pd.DataFrame({"value": [1]}), schema=schema)


def test_explicit_matching_schema_is_accepted() -> None:
    """A matching Task 03 schema can be reused directly."""
    frame = pd.DataFrame({"value": [1, 2, 3]})
    schema = infer_schema(frame)

    report = check_data_quality(frame, schema=schema)

    assert report.n_rows == 3
    assert report.n_columns == 1


def test_input_dataframe_is_not_mutated() -> None:
    """Quality rules perform no cleaning, conversion, or other mutation."""
    frame = pd.DataFrame(
        {
            "date": ["2025-01-01", "bad", "also-bad"],
            "numeric": [np.inf, 1.0, np.nan],
        }
    )
    before = frame.copy(deep=True)
    dtypes_before = frame.dtypes.copy()

    check_data_quality(frame)

    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_series_equal(frame.dtypes, dtypes_before)


def test_issue_order_is_table_then_code_then_original_column() -> None:
    """Issues follow the frozen code order and preserve input column order."""
    frame = pd.DataFrame(
        {
            "second": [None, None, None, None],
            "first": [None, None, None, None],
            "constant": [1, 1, 1, 1],
        }
    )
    report = check_data_quality(frame)

    assert [(issue.code, issue.column) for issue in report.issues] == [
        ("duplicate_rows", None),
        ("all_missing_column", "second"),
        ("all_missing_column", "first"),
        ("constant_column", "constant"),
    ]
    assert report.issue_count == len(report.issues)
    assert report.severity_counts == {"info": 0, "warning": 4, "error": 0}


def test_issue_messages_and_suggestions_are_stable() -> None:
    """A representative issue uses the exact frozen user-facing text."""
    report = check_data_quality(pd.DataFrame({"value": [None, 1.0, None]}))
    issue = _issues_by_code(report, "high_missing_column")[0]

    assert issue.message == "Column has a high missing rate"
    assert issue.suggestion == (
        "Review missingness before using this column for analysis or modeling"
    )
