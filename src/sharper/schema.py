"""Deterministic schema inference for pandas DataFrames."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

_LOGICAL_TYPES = (
    "numeric",
    "categorical",
    "datetime",
    "boolean",
    "text",
    "identifier",
    "unknown",
)
_ID_NAME_TOKENS = frozenset({"id", "key", "uuid"})
_BOOLEAN_TYPES = (bool, np.bool_)
_DATETIME_TYPES = (date, datetime, np.datetime64, pd.Timestamp)


@dataclass(frozen=True)
class ColumnSchema:
    """Describe the observed physical and logical schema of one column.

    Attributes
    ----------
    name
        Original string column name.
    pandas_dtype
        String representation of the pandas dtype.
    logical_type
        One of the seven logical types frozen by the Task 03 contract.
    nullable
        Whether at least one missing value was observed.
    missing_count
        Number of missing cells.
    missing_rate
        Missing cells divided by row count, or zero for an empty-row frame.
    unique_count
        Number of distinct non-missing values.
    unique_rate
        Distinct values divided by non-missing values, or zero when none exist.
    is_constant
        Whether the column has exactly one distinct non-missing value.
    is_id_like
        Whether the identifier rule matched.
    confidence
        Fixed confidence value assigned by the inference rule.
    reasons
        Ordered reason codes from the frozen Task 03 vocabulary.
    """

    name: str
    pandas_dtype: str
    logical_type: str
    nullable: bool
    missing_count: int
    missing_rate: float
    unique_count: int
    unique_rate: float
    is_constant: bool
    is_id_like: bool
    confidence: float
    reasons: list[str]


@dataclass(frozen=True)
class TargetCandidate:
    """Describe a non-binding target-column suggestion.

    Attributes
    ----------
    name
        Original string column name.
    suggested_task_type
        ``"classification"``, ``"regression"``, or ``"unknown"``.
    confidence
        Fixed target-candidate confidence; v0.1 emits only ``0.9`` or ``0.75``.
    reasons
        Ordered reason codes from the frozen Task 03 vocabulary.
    """

    name: str
    suggested_task_type: str
    confidence: float
    reasons: list[str]


@dataclass(frozen=True)
class SchemaReport:
    """Contain deterministic schema inference for a DataFrame.

    Attributes
    ----------
    n_rows
        Number of input rows.
    n_columns
        Number of input columns.
    columns
        Column schemas in original column order.
    logical_type_counts
        Counts for all seven logical types in their frozen order.
    target_candidates
        Non-binding suggestions in input order, with an explicit target first.
    """

    n_rows: int
    n_columns: int
    columns: list[ColumnSchema]
    logical_type_counts: dict[str, int]
    target_candidates: list[TargetCandidate]


def infer_schema(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    id_threshold: float = 0.98,
) -> SchemaReport:
    """Infer a deterministic logical schema without modifying the input.

    Parameters
    ----------
    df
        DataFrame whose string-named columns will be inspected.
    target
        Optional explicit target suggestion. It is validated and placed first in
        ``target_candidates`` but is not confirmed or analyzed.
    id_threshold
        Minimum non-missing unique rate for identifier inference. Must be in
        ``(0.0, 1.0]``.

    Returns
    -------
    SchemaReport
        Dataset shape, ordered column schemas, logical-type counts, and target
        suggestions.

    Raises
    ------
    ValueError
        If column names are duplicated or non-string, ``id_threshold`` is invalid,
        or an explicit target is absent.

    Notes
    -----
    Missing values are excluded from unique counts and inference samples. Empty-row
    and all-missing columns are ``"unknown"``. Date strings are parsed only in a
    temporary value sequence; neither values nor dtypes in ``df`` are changed.

    Examples
    --------
    >>> import pandas as pd
    >>> from sharper import infer_schema
    >>> report = infer_schema(pd.DataFrame({"score": [1.0, 2.0]}))
    >>> report.columns[0].logical_type
    'numeric'
    """
    _validate_dataframe_columns(df)
    if not 0.0 < id_threshold <= 1.0:
        raise ValueError("id_threshold must be in the interval (0.0, 1.0]")
    if target is not None and target not in df.columns:
        raise ValueError(f"target column not found: {target!r}")

    columns = [
        _infer_column(df[name], name=name, n_rows=len(df), id_threshold=id_threshold)
        for name in df.columns
    ]
    logical_type_counts = {
        logical_type: sum(column.logical_type == logical_type for column in columns)
        for logical_type in _LOGICAL_TYPES
    }
    target_candidates = _infer_target_candidates(columns, target=target)
    return SchemaReport(
        n_rows=len(df),
        n_columns=len(df.columns),
        columns=columns,
        logical_type_counts=logical_type_counts,
        target_candidates=target_candidates,
    )


def _validate_dataframe_columns(df: pd.DataFrame) -> None:
    if not all(isinstance(name, str) for name in df.columns):
        raise ValueError("DataFrame column names must all be strings")
    if df.columns.has_duplicates:
        raise ValueError("DataFrame column names must be unique")


def _infer_column(
    series: pd.Series,
    *,
    name: str,
    n_rows: int,
    id_threshold: float,
) -> ColumnSchema:
    missing_count = int(series.isna().sum())
    non_null = series.dropna()
    non_null_count = int(non_null.size)
    unique_count = int(non_null.nunique())
    missing_rate = missing_count / n_rows if n_rows else 0.0
    unique_rate = unique_count / non_null_count if non_null_count else 0.0
    is_constant = non_null_count > 0 and unique_count == 1

    logical_type, confidence, reasons, is_id_like = _infer_logical_type(
        series,
        non_null=non_null,
        n_rows=n_rows,
        unique_rate=unique_rate,
        id_threshold=id_threshold,
        name=name,
    )
    return ColumnSchema(
        name=name,
        pandas_dtype=str(series.dtype),
        logical_type=logical_type,
        nullable=missing_count > 0,
        missing_count=missing_count,
        missing_rate=float(missing_rate),
        unique_count=unique_count,
        unique_rate=float(unique_rate),
        is_constant=is_constant,
        is_id_like=is_id_like,
        confidence=confidence,
        reasons=reasons,
    )


def _infer_logical_type(
    series: pd.Series,
    *,
    non_null: pd.Series,
    n_rows: int,
    unique_rate: float,
    id_threshold: float,
    name: str,
) -> tuple[str, float, list[str], bool]:
    if n_rows == 0:
        return "unknown", 0.5, ["empty_dataframe"], False
    if non_null.empty:
        return "unknown", 0.5, ["all_missing"], False

    dtype = series.dtype
    if is_bool_dtype(dtype) and not _is_category_dtype(dtype):
        return "boolean", 1.0, ["pandas_boolean_dtype"], False
    if _supports_value_inference(dtype) and _all_boolean_tokens(non_null):
        return "boolean", 0.8, ["boolean_values_only"], False
    if is_datetime64_any_dtype(dtype):
        return "datetime", 1.0, ["pandas_datetime_dtype"], False
    if _is_string_values(non_null) and not _is_category_dtype(dtype):
        parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
        if float(parsed.notna().mean()) >= 0.80:
            return "datetime", 0.85, ["string_datetime_parse_rate_met"], False

    if is_object_dtype(dtype) and _has_mixed_type_families(non_null):
        return "unknown", 0.5, ["mixed_object_unknown"], False

    id_name_match = bool(_ID_NAME_TOKENS.intersection(_name_tokens(name)))
    all_unique = unique_rate == 1.0
    id_value_dtype = (
        is_integer_dtype(dtype) or is_object_dtype(dtype) or is_string_dtype(dtype)
    )
    identifier_evidence = id_name_match or (all_unique and id_value_dtype)
    if unique_rate >= id_threshold and identifier_evidence:
        reasons = ["identifier_high_unique_rate"]
        if id_name_match:
            reasons.insert(0, "identifier_name_pattern")
        return "identifier", 0.9, reasons, True

    if is_numeric_dtype(dtype):
        return "numeric", 1.0, ["pandas_numeric_dtype"], False
    if _is_category_dtype(dtype):
        return "categorical", 1.0, ["pandas_category_dtype"], False
    if _is_string_values(non_null):
        if unique_rate <= 0.50 or int(non_null.nunique()) <= 50:
            return "categorical", 0.8, ["categorical_unique_threshold_met"], False
        return "text", 0.8, ["text_high_unique_rate"], False
    return "unknown", 0.5, ["fallback_unknown"], False


def _supports_value_inference(dtype: object) -> bool:
    return is_object_dtype(dtype) or is_string_dtype(dtype) or _is_category_dtype(dtype)


def _all_boolean_tokens(series: pd.Series) -> bool:
    return all(
        str(value).strip().casefold() in {"true", "false"} for value in series.array
    )


def _is_string_values(series: pd.Series) -> bool:
    return all(isinstance(value, str) for value in series.array)


def _is_category_dtype(dtype: object) -> bool:
    return isinstance(dtype, pd.CategoricalDtype)


def _name_tokens(name: str) -> set[str]:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return set(re.findall(r"[a-z0-9]+", separated.lower()))


def _has_mixed_type_families(series: pd.Series) -> bool:
    return len({_type_family(value) for value in series.array}) > 1


def _type_family(value: object) -> str:
    if isinstance(value, _BOOLEAN_TYPES):
        return "boolean"
    if isinstance(value, _DATETIME_TYPES):
        return "datetime-like"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float, complex, np.number)):
        return "numeric"
    return "other"


def _infer_target_candidates(
    columns: list[ColumnSchema],
    *,
    target: str | None,
) -> list[TargetCandidate]:
    candidates: list[TargetCandidate] = []
    for column in columns:
        explicit = column.name == target
        name_reasons, name_confidence = _target_name_signal(column.name)
        if not explicit and not name_reasons:
            continue
        if column.logical_type == "identifier" and not explicit:
            continue

        task_type, task_reason = _suggest_task_type(column)
        explicit_confidence = 0.0
        reasons: list[str] = []
        if explicit:
            reasons.append("explicit_target")
            explicit_confidence = 0.9 if task_type != "unknown" else 0.75
        reasons.extend(name_reasons)
        if task_reason is not None:
            reasons.append(task_reason)
        candidates.append(
            TargetCandidate(
                name=column.name,
                suggested_task_type=task_type,
                confidence=max(explicit_confidence, name_confidence),
                reasons=reasons,
            )
        )

    if target is not None:
        candidates.sort(key=lambda candidate: candidate.name != target)
    return candidates


def _target_name_signal(name: str) -> tuple[list[str], float]:
    lowered = name.lower()
    if lowered == "target":
        return ["target_name_exact"], 0.9
    if lowered == "y":
        return ["target_name_is_y"], 0.9

    reasons: list[str] = []
    exact_confidence = 0.0
    if "target" in lowered:
        reasons.append("target_name_contains_target")
    if "label" in lowered:
        reasons.append("target_name_contains_label")
        if lowered == "label":
            exact_confidence = 0.9
    if "outcome" in lowered:
        reasons.append("target_name_contains_outcome")
        if lowered == "outcome":
            exact_confidence = 0.9
    if not reasons:
        return [], 0.0
    return reasons, max(0.75, exact_confidence)


def _suggest_task_type(column: ColumnSchema) -> tuple[str, str | None]:
    if column.logical_type == "boolean":
        return "classification", "classification_boolean"
    if column.logical_type == "categorical":
        return "classification", "classification_categorical"
    if column.logical_type == "numeric":
        if column.unique_count <= 20:
            return "classification", "classification_low_cardinality"
        return "regression", "regression_numeric_high_cardinality"
    return "unknown", None
