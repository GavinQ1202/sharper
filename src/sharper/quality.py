"""Deterministic minimal data-quality reporting for pandas DataFrames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from sharper.schema import (
    SchemaReport,
    _validate_dataframe_columns,
    infer_schema,
)

_SEVERITIES = ("info", "warning", "error")
_NEAR_CONSTANT_THRESHOLD = 0.95


@dataclass(frozen=True)
class QualityIssue:
    """Describe one deterministic table-level or column-level quality issue.

    Attributes
    ----------
    code
        One of the eleven issue codes frozen by the Task 04 contract.
    severity
        ``"info"``, ``"warning"``, or ``"error"``.
    scope
        ``"table"`` or ``"column"``.
    column
        Original column name for a column issue, otherwise ``None``.
    count
        Number of values or rows triggering the issue, when applicable.
    ratio
        Trigger count divided by the rule's frozen denominator, when applicable.
    threshold
        Threshold used by the rule, when applicable.
    message
        Stable description of the observed issue.
    suggestion
        Stable, non-executing suggestion for reviewing the issue.
    """

    code: str
    severity: str
    scope: str
    column: str | None
    count: int | None
    ratio: float | None
    threshold: float | None
    message: str
    suggestion: str


@dataclass(frozen=True)
class QualityReport:
    """Contain deterministic quality issues for a DataFrame.

    Attributes
    ----------
    n_rows
        Number of input rows.
    n_columns
        Number of input columns.
    issue_count
        Number of entries in ``issues``.
    severity_counts
        Counts for ``info``, ``warning``, and ``error`` in that order.
    issues
        Issues in frozen table/code/original-column order.
    """

    n_rows: int
    n_columns: int
    issue_count: int
    severity_counts: dict[str, int]
    issues: list[QualityIssue]


def check_data_quality(
    df: pd.DataFrame,
    *,
    schema: SchemaReport | None = None,
    missing_threshold: float = 0.40,
) -> QualityReport:
    """Report the fixed Task 04 quality rules without modifying the input.

    Parameters
    ----------
    df
        DataFrame with unique string column names.
    schema
        Optional schema inferred for the same shape, column names, and order.
        When omitted, :func:`sharper.schema.infer_schema` is called.
    missing_threshold
        Inclusive high-missing threshold in ``(0.0, 1.0]``.

    Returns
    -------
    QualityReport
        Stable table-level and column-level issues with counts, ratios, thresholds,
        messages, and non-executing suggestions.

    Raises
    ------
    ValueError
        If ``df`` is not a DataFrame, column names are duplicated or non-string,
        ``missing_threshold`` is outside ``(0.0, 1.0]``, or a supplied schema does
        not match the DataFrame shape, names, and order.

    Notes
    -----
    Missing values use pandas ``isna`` semantics. Duplicate rows use
    ``duplicated(keep=False)``. No issue repairs, conversions, or other input
    mutations are performed.

    Examples
    --------
    >>> import pandas as pd
    >>> from sharper import check_data_quality
    >>> report = check_data_quality(pd.DataFrame({"value": [1, 1]}))
    >>> [issue.code for issue in report.issues]
    ['duplicate_rows', 'constant_column']
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    _validate_dataframe_columns(df)
    if not 0.0 < missing_threshold <= 1.0:
        raise ValueError("missing_threshold must be > 0 and <= 1")

    resolved_schema = infer_schema(df) if schema is None else schema
    _validate_schema_matches(df, resolved_schema)

    n_rows, n_columns = df.shape
    if n_rows == 0:
        severity = "error" if n_columns == 0 else "warning"
        message = (
            "DataFrame has no rows or columns"
            if n_columns == 0
            else "DataFrame has no rows"
        )
        return _build_report(
            n_rows,
            n_columns,
            [
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
            ],
        )

    issues: list[QualityIssue] = []
    _append_duplicate_rows(df, issues)
    _append_all_missing(resolved_schema, issues)
    _append_high_missing(resolved_schema, missing_threshold, issues)
    _append_constant(resolved_schema, issues)
    _append_near_constant(df, resolved_schema, issues)
    _append_high_cardinality(resolved_schema, issues)
    _append_identifiers(resolved_schema, issues)
    _append_infinite_values(df, issues)
    _append_mixed_python_types(resolved_schema, issues)
    _append_datetime_parse_failures(df, resolved_schema, issues)
    return _build_report(n_rows, n_columns, issues)


def _validate_schema_matches(df: pd.DataFrame, schema: SchemaReport) -> None:
    schema_names = [column.name for column in schema.columns]
    if (
        schema.n_rows != len(df)
        or schema.n_columns != len(df.columns)
        or schema_names != list(df.columns)
    ):
        raise ValueError("schema does not match DataFrame")


def _append_duplicate_rows(
    df: pd.DataFrame,
    issues: list[QualityIssue],
) -> None:
    count = int(df.duplicated(keep=False).sum())
    if count == 0:
        return
    issues.append(
        QualityIssue(
            code="duplicate_rows",
            severity="warning",
            scope="table",
            column=None,
            count=count,
            ratio=float(count / len(df)),
            threshold=None,
            message="Duplicate rows detected",
            suggestion=(
                "Review duplicate rows and decide whether they should be removed "
                "or consolidated"
            ),
        )
    )


def _append_all_missing(
    schema: SchemaReport,
    issues: list[QualityIssue],
) -> None:
    for column in schema.columns:
        if column.missing_count != schema.n_rows:
            continue
        issues.append(
            QualityIssue(
                code="all_missing_column",
                severity="warning",
                scope="column",
                column=column.name,
                count=column.missing_count,
                ratio=1.0,
                threshold=None,
                message="Column contains only missing values",
                suggestion=(
                    "Consider dropping the column or investigating the data source"
                ),
            )
        )


def _append_high_missing(
    schema: SchemaReport,
    missing_threshold: float,
    issues: list[QualityIssue],
) -> None:
    for column in schema.columns:
        if (
            0 < column.missing_count < schema.n_rows
            and column.missing_rate >= missing_threshold
        ):
            issues.append(
                QualityIssue(
                    code="high_missing_column",
                    severity="warning",
                    scope="column",
                    column=column.name,
                    count=column.missing_count,
                    ratio=column.missing_rate,
                    threshold=missing_threshold,
                    message="Column has a high missing rate",
                    suggestion=(
                        "Review missingness before using this column for analysis "
                        "or modeling"
                    ),
                )
            )


def _append_constant(
    schema: SchemaReport,
    issues: list[QualityIssue],
) -> None:
    for column in schema.columns:
        if not column.is_constant:
            continue
        issues.append(
            QualityIssue(
                code="constant_column",
                severity="warning",
                scope="column",
                column=column.name,
                count=1,
                ratio=1.0,
                threshold=None,
                message="Column has a constant non-missing value",
                suggestion=(
                    "Consider excluding constant columns from analysis or modeling"
                ),
            )
        )


def _append_near_constant(
    df: pd.DataFrame,
    schema: SchemaReport,
    issues: list[QualityIssue],
) -> None:
    for column in schema.columns:
        if column.is_constant:
            continue
        non_null = df[column.name].dropna()
        if non_null.empty:
            continue
        top_count = int(non_null.value_counts(dropna=True).iloc[0])
        ratio = float(top_count / len(non_null))
        if ratio < _NEAR_CONSTANT_THRESHOLD:
            continue
        issues.append(
            QualityIssue(
                code="near_constant_column",
                severity="info",
                scope="column",
                column=column.name,
                count=top_count,
                ratio=ratio,
                threshold=_NEAR_CONSTANT_THRESHOLD,
                message="Column is near constant",
                suggestion=(
                    "Check whether the column adds useful variation before analysis "
                    "or modeling"
                ),
            )
        )


def _append_high_cardinality(
    schema: SchemaReport,
    issues: list[QualityIssue],
) -> None:
    for column in schema.columns:
        if (
            column.logical_type != "categorical"
            or column.is_id_like
            or column.unique_count <= 50
            or column.unique_rate <= 0.50
        ):
            continue
        issues.append(
            QualityIssue(
                code="high_cardinality_categorical",
                severity="info",
                scope="column",
                column=column.name,
                count=column.unique_count,
                ratio=column.unique_rate,
                threshold=0.50,
                message="Categorical column has high cardinality",
                suggestion=(
                    "Consider grouping rare categories or using appropriate encoding "
                    "strategies"
                ),
            )
        )


def _append_identifiers(
    schema: SchemaReport,
    issues: list[QualityIssue],
) -> None:
    for column in schema.columns:
        if column.logical_type != "identifier" and not column.is_id_like:
            continue
        issues.append(
            QualityIssue(
                code="identifier_like_column",
                severity="info",
                scope="column",
                column=column.name,
                count=column.unique_count,
                ratio=column.unique_rate,
                threshold=None,
                message="Column appears to be an identifier",
                suggestion=(
                    "Avoid treating identifier-like columns as ordinary predictive "
                    "features"
                ),
            )
        )


def _append_infinite_values(
    df: pd.DataFrame,
    issues: list[QualityIssue],
) -> None:
    for name in df.columns:
        series = df[name]
        if not is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype):
            continue
        count = int(np.isinf(series).sum())
        if count == 0:
            continue
        issues.append(
            QualityIssue(
                code="infinite_values",
                severity="warning",
                scope="column",
                column=name,
                count=count,
                ratio=float(count / len(df)),
                threshold=None,
                message="Numeric column contains infinite values",
                suggestion=(
                    "Replace or remove infinite values before analysis or modeling"
                ),
            )
        )


def _append_mixed_python_types(
    schema: SchemaReport,
    issues: list[QualityIssue],
) -> None:
    for column in schema.columns:
        if (
            column.logical_type != "unknown"
            or "mixed_object_unknown" not in column.reasons
        ):
            continue
        non_null_count = schema.n_rows - column.missing_count
        issues.append(
            QualityIssue(
                code="mixed_python_types",
                severity="warning",
                scope="column",
                column=column.name,
                count=non_null_count,
                ratio=float(non_null_count / schema.n_rows),
                threshold=None,
                message="Column contains mixed Python value types",
                suggestion="Standardize the column values before analysis",
            )
        )


def _append_datetime_parse_failures(
    df: pd.DataFrame,
    schema: SchemaReport,
    issues: list[QualityIssue],
) -> None:
    schema_by_name = {column.name: column for column in schema.columns}
    for name in df.columns:
        series = df[name]
        column = schema_by_name[name]
        if column.logical_type == "datetime" or not _supports_datetime_parse(series):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
        parse_success_count = int(parsed.notna().sum())
        failure_count = int(parsed.isna().sum())
        if parse_success_count == 0 or failure_count == 0:
            continue
        issues.append(
            QualityIssue(
                code="datetime_parse_failures",
                severity="info",
                scope="column",
                column=name,
                count=failure_count,
                ratio=float(failure_count / len(non_null)),
                threshold=None,
                message="Column contains partial datetime-like values",
                suggestion=(
                    "Review datetime parsing before using this column as a date or "
                    "time feature"
                ),
            )
        )


def _supports_datetime_parse(series: pd.Series) -> bool:
    dtype = series.dtype
    return (
        is_object_dtype(dtype)
        or is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def _build_report(
    n_rows: int,
    n_columns: int,
    issues: list[QualityIssue],
) -> QualityReport:
    return QualityReport(
        n_rows=n_rows,
        n_columns=n_columns,
        issue_count=len(issues),
        severity_counts={
            severity: sum(issue.severity == severity for issue in issues)
            for severity in _SEVERITIES
        },
        issues=issues,
    )
