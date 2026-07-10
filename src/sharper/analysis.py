"""Deterministic non-target feature analysis for pandas DataFrames."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from numbers import Real

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

_NUMERIC_SUMMARY_DTYPES = {
    "column": "object",
    "count": "int64",
    "missing_count": "int64",
    "missing_rate": "float64",
    "mean": "float64",
    "std": "float64",
    "min": "float64",
    "q25": "float64",
    "median": "float64",
    "q75": "float64",
    "max": "float64",
    "skew": "float64",
    "zero_count": "int64",
    "zero_rate": "float64",
}

_CATEGORICAL_SUMMARY_DTYPES = {
    "column": "object",
    "count": "int64",
    "missing_count": "int64",
    "missing_rate": "float64",
    "unique_count": "int64",
    "unique_rate": "float64",
    "top": "object",
    "top_count": "int64",
    "top_rate": "float64",
}

_TOP_CATEGORIES_DTYPES = {
    "column": "object",
    "category": "object",
    "count": "int64",
    "rate": "float64",
    "rank": "int64",
}

_CORRELATION_DTYPES = {
    "column_a": "object",
    "column_b": "object",
    "method": "object",
    "correlation": "float64",
    "n_pairs": "int64",
}

_OUTLIER_SUMMARY_DTYPES = {
    "column": "object",
    "method": "object",
    "threshold": "float64",
    "lower_bound": "float64",
    "upper_bound": "float64",
    "outlier_count": "int64",
    "outlier_rate": "float64",
}

_OUTLIER_DETAILS_DTYPES = {
    "column": "object",
    "row_index": "object",
    "value": "float64",
    "lower_bound": "float64",
    "upper_bound": "float64",
}


@dataclass(frozen=True)
class NumericAnalysis:
    """Contain ordered numeric feature statistics and skipped-column metadata.

    Attributes
    ----------
    n_rows
        Number of input rows.
    requested_columns
        Explicit requested columns in caller order, or ``None`` for auto-selection.
    analyzed_columns
        Numeric non-boolean, non-all-missing columns in analysis order.
    skipped_columns
        Skipped columns in request or DataFrame order.
    skipped_reasons
        One frozen Task 07 reason code for each skipped column.
    summary
        Numeric statistics using the fixed Task 07 table schema.
    """

    n_rows: int
    requested_columns: tuple[str, ...] | None
    analyzed_columns: tuple[str, ...]
    skipped_columns: tuple[str, ...]
    skipped_reasons: dict[str, str]
    summary: pd.DataFrame


@dataclass(frozen=True)
class CategoricalAnalysis:
    """Contain ordered categorical summaries and top-category frequencies.

    Attributes
    ----------
    n_rows
        Number of input rows.
    requested_columns
        Explicit requested columns in caller order, or ``None`` for auto-selection.
    analyzed_columns
        Categorical columns with at least one non-missing value.
    skipped_columns
        Skipped columns in request or DataFrame order.
    skipped_reasons
        One frozen Task 07 reason code for each skipped column.
    top_n
        Requested maximum displayed categories per analyzed column.
    summary
        Per-column categorical statistics with the fixed schema.
    top_categories
        Frequency rows ordered by count and first appearance.
    """

    n_rows: int
    requested_columns: tuple[str, ...] | None
    analyzed_columns: tuple[str, ...]
    skipped_columns: tuple[str, ...]
    skipped_reasons: dict[str, str]
    top_n: int
    summary: pd.DataFrame
    top_categories: pd.DataFrame


@dataclass(frozen=True)
class CorrelationAnalysis:
    """Contain budgeted long-form numeric pairwise correlations.

    Attributes
    ----------
    n_rows
        Number of input rows.
    requested_columns
        Explicit requested columns in caller order, or ``None`` for auto-selection.
    analyzed_columns
        Eligible columns retained after validation, skips, and the column budget.
    skipped_columns
        Skipped columns in request or DataFrame order.
    skipped_reasons
        One frozen Task 07 reason code for each skipped column.
    method
        Correlation method, ``"pearson"`` or ``"spearman"``.
    max_columns
        Maximum eligible columns retained for pairwise computation.
    min_periods
        Minimum non-missing observations required for a column and pair.
    truncated
        Whether any otherwise eligible column exceeded ``max_columns``.
    correlations
        Long-form pairs in analyzed-column order with no diagonal.
    """

    n_rows: int
    requested_columns: tuple[str, ...] | None
    analyzed_columns: tuple[str, ...]
    skipped_columns: tuple[str, ...]
    skipped_reasons: dict[str, str]
    method: str
    max_columns: int
    min_periods: int
    truncated: bool
    correlations: pd.DataFrame


@dataclass(frozen=True)
class OutlierAnalysis:
    """Contain deterministic IQR bounds, rates, and original-row outliers.

    Attributes
    ----------
    n_rows
        Number of input rows.
    requested_columns
        Explicit requested columns in caller order, or ``None`` for auto-selection.
    analyzed_columns
        Eligible finite, non-constant columns with at least two observations.
    skipped_columns
        Skipped columns in request or DataFrame order.
    skipped_reasons
        One frozen Task 07 reason code for each skipped column.
    method
        Outlier method; Task 07 supports only ``"iqr"``.
    threshold
        Positive IQR multiplier stored as a float.
    summary
        Per-column bounds and outlier rates with the fixed schema.
    outliers
        Outlying values with original index labels and row order.
    """

    n_rows: int
    requested_columns: tuple[str, ...] | None
    analyzed_columns: tuple[str, ...]
    skipped_columns: tuple[str, ...]
    skipped_reasons: dict[str, str]
    method: str
    threshold: float
    summary: pd.DataFrame
    outliers: pd.DataFrame


def analyze_numeric_features(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
) -> NumericAnalysis:
    """Analyze numeric non-boolean features without modifying the input.

    Parameters
    ----------
    df
        DataFrame with unique string column names.
    columns
        Optional ordered columns. When omitted, numeric non-boolean columns are
        selected in DataFrame order.

    Returns
    -------
    NumericAnalysis
        Ordered statistics plus explicit skipped columns and reasons.

    Raises
    ------
    ValueError
        If ``df`` or its column names are invalid, or requested columns contain
        non-strings, duplicates, or missing names.

    Notes
    -----
    Missing values are excluded from statistics and rate denominators as stated in
    the result schema. Positive and negative infinity are not cleaned or replaced.
    ``df`` is never mutated.

    Examples
    --------
    >>> import pandas as pd
    >>> from sharper import analyze_numeric_features
    >>> result = analyze_numeric_features(pd.DataFrame({"x": [0.0, 2.0]}))
    >>> result.summary.loc[0, "zero_rate"]
    0.5
    """
    requested, candidates = _validate_inputs(df, columns)
    analyzed, skipped_reasons = _select_basic_columns(
        df,
        candidates,
        eligible=_is_numeric_non_bool,
        ineligible_reason="not_numeric",
        record_ineligible=requested is not None,
    )

    rows: list[dict[str, object]] = []
    final_analyzed: list[str] = []
    for column in analyzed:
        series = df[column]
        count = int(series.count())
        if count == 0:
            skipped_reasons[column] = "all_missing"
            continue
        missing_count = len(df) - count
        zero_count = int(series.eq(0).sum())
        rows.append(
            {
                "column": column,
                "count": count,
                "missing_count": missing_count,
                "missing_rate": missing_count / len(df) if len(df) else 0.0,
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "q25": series.quantile(0.25),
                "median": series.quantile(0.50),
                "q75": series.quantile(0.75),
                "max": series.max(),
                "skew": series.skew(),
                "zero_count": zero_count,
                "zero_rate": zero_count / count,
            }
        )
        final_analyzed.append(column)

    return NumericAnalysis(
        n_rows=len(df),
        requested_columns=requested,
        analyzed_columns=tuple(final_analyzed),
        skipped_columns=_ordered_skipped(candidates, skipped_reasons),
        skipped_reasons=skipped_reasons,
        summary=_typed_frame(rows, _NUMERIC_SUMMARY_DTYPES),
    )


def analyze_categorical_features(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    top_n: int = 10,
) -> CategoricalAnalysis:
    """Analyze categorical features and display budgeted category frequencies.

    Parameters
    ----------
    df
        DataFrame with unique string column names.
    columns
        Optional ordered columns. When omitted, object, string, category, and
        boolean columns are selected in DataFrame order.
    top_n
        Positive maximum number of categories returned for each analyzed column.

    Returns
    -------
    CategoricalAnalysis
        Ordered categorical summaries, top-category rows, and skipped metadata.

    Raises
    ------
    ValueError
        If ``df``, its names, requested columns, or ``top_n`` are invalid.

    Notes
    -----
    Missing values do not participate in unique counts or category frequencies.
    Ties are resolved by first appearance in the original column. The function
    does not mutate ``df``.

    Examples
    --------
    >>> import pandas as pd
    >>> from sharper import analyze_categorical_features
    >>> result = analyze_categorical_features(pd.DataFrame({"x": ["b", "a", "b"]}))
    >>> result.summary.loc[0, "top"]
    'b'
    """
    requested, candidates = _validate_inputs(df, columns)
    if not _is_strict_int(top_n) or top_n < 1:
        raise ValueError("top_n must be a positive integer")

    analyzed, skipped_reasons = _select_basic_columns(
        df,
        candidates,
        eligible=_is_categorical,
        ineligible_reason="not_categorical",
        record_ineligible=requested is not None,
    )
    summary_rows: list[dict[str, object]] = []
    category_rows: list[dict[str, object]] = []
    final_analyzed: list[str] = []
    for column in analyzed:
        series = df[column]
        non_missing = series.dropna()
        count = int(non_missing.size)
        if count == 0:
            skipped_reasons[column] = "all_missing"
            continue

        frequencies = _ordered_category_frequencies(non_missing)
        top_value, top_count = frequencies[0]
        unique_count = len(frequencies)
        missing_count = len(df) - count
        summary_rows.append(
            {
                "column": column,
                "count": count,
                "missing_count": missing_count,
                "missing_rate": missing_count / len(df) if len(df) else 0.0,
                "unique_count": unique_count,
                "unique_rate": unique_count / count,
                "top": top_value,
                "top_count": top_count,
                "top_rate": top_count / count,
            }
        )
        for rank, (category, category_count) in enumerate(frequencies[:top_n], start=1):
            category_rows.append(
                {
                    "column": column,
                    "category": category,
                    "count": category_count,
                    "rate": category_count / count,
                    "rank": rank,
                }
            )
        final_analyzed.append(column)

    return CategoricalAnalysis(
        n_rows=len(df),
        requested_columns=requested,
        analyzed_columns=tuple(final_analyzed),
        skipped_columns=_ordered_skipped(candidates, skipped_reasons),
        skipped_reasons=skipped_reasons,
        top_n=top_n,
        summary=_typed_frame(summary_rows, _CATEGORICAL_SUMMARY_DTYPES),
        top_categories=_typed_frame(category_rows, _TOP_CATEGORIES_DTYPES),
    )


def compute_correlations(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    method: str = "pearson",
    max_columns: int = 50,
    min_periods: int = 2,
) -> CorrelationAnalysis:
    """Compute budgeted long-form Pearson or Spearman numeric correlations.

    Parameters
    ----------
    df
        DataFrame with unique string column names.
    columns
        Optional ordered columns; otherwise numeric non-boolean columns are used.
    method
        ``"pearson"`` or ``"spearman"``.
    max_columns
        Integer at least two limiting eligible analyzed columns.
    min_periods
        Integer at least two required for each column and emitted pair.

    Returns
    -------
    CorrelationAnalysis
        Long-form pairs, effective pair counts, skips, and truncation metadata.

    Raises
    ------
    ValueError
        If inputs, names, requested columns, method, or budgets are invalid.

    Notes
    -----
    Missing values are excluded pairwise. Pairs below ``min_periods`` and pairs for
    which pandas returns ``NaN`` are omitted. No p-values or plots are produced,
    and ``df`` is not mutated.

    Examples
    --------
    >>> import pandas as pd
    >>> from sharper import compute_correlations
    >>> result = compute_correlations(pd.DataFrame({"x": [1, 2], "y": [2, 4]}))
    >>> result.correlations.loc[0, "correlation"]
    0.9999999999999999
    """
    requested, candidates = _validate_inputs(df, columns)
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be pearson or spearman")
    if not _is_strict_int(max_columns) or max_columns < 2:
        raise ValueError("max_columns must be an integer >= 2")
    if not _is_strict_int(min_periods) or min_periods < 2:
        raise ValueError("min_periods must be an integer >= 2")

    eligible: list[str] = []
    skipped_reasons: dict[str, str] = {}
    for column in candidates:
        series = df[column]
        if not _is_numeric_non_bool(series.dtype):
            if requested is not None:
                skipped_reasons[column] = "not_numeric"
            continue
        non_missing = series.dropna()
        if non_missing.empty:
            skipped_reasons[column] = "all_missing"
        elif len(non_missing) < min_periods:
            skipped_reasons[column] = "insufficient_non_missing"
        elif non_missing.nunique() == 1:
            skipped_reasons[column] = "constant"
        else:
            eligible.append(column)

    analyzed = eligible[:max_columns]
    for column in eligible[max_columns:]:
        skipped_reasons[column] = "exceeds_max_columns"

    pair_rows: list[dict[str, object]] = []
    for column_a, column_b in combinations(analyzed, 2):
        pair_mask = df[column_a].notna() & df[column_b].notna()
        n_pairs = int(pair_mask.sum())
        if n_pairs < min_periods:
            continue
        coefficient = df[column_a].corr(df[column_b], method=method)
        if pd.isna(coefficient):
            continue
        pair_rows.append(
            {
                "column_a": column_a,
                "column_b": column_b,
                "method": method,
                "correlation": coefficient,
                "n_pairs": n_pairs,
            }
        )

    return CorrelationAnalysis(
        n_rows=len(df),
        requested_columns=requested,
        analyzed_columns=tuple(analyzed),
        skipped_columns=_ordered_skipped(candidates, skipped_reasons),
        skipped_reasons=skipped_reasons,
        method=method,
        max_columns=max_columns,
        min_periods=min_periods,
        truncated=bool(eligible[max_columns:]),
        correlations=_typed_frame(pair_rows, _CORRELATION_DTYPES),
    )


def detect_outliers(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> OutlierAnalysis:
    """Detect numeric outliers with deterministic IQR bounds.

    Parameters
    ----------
    df
        DataFrame with unique string column names.
    columns
        Optional ordered columns; otherwise numeric non-boolean columns are used.
    method
        Outlier method. Task 07 supports only ``"iqr"``.
    threshold
        Positive real-valued multiplier for the interquartile range.

    Returns
    -------
    OutlierAnalysis
        Per-column bounds and rates plus original-index outlier details.

    Raises
    ------
    ValueError
        If inputs, names, requested columns, method, or threshold are invalid.

    Notes
    -----
    Missing values are excluded. A column containing either infinity is skipped in
    full. Detected rows preserve the original index label and DataFrame row order.
    Values are reported, never cleaned, deleted, or changed in ``df``.

    Examples
    --------
    >>> import pandas as pd
    >>> from sharper import detect_outliers
    >>> result = detect_outliers(pd.DataFrame({"x": [0, 0, 0, 10]}))
    >>> result.outliers["value"].tolist()
    [10.0]
    """
    requested, candidates = _validate_inputs(df, columns)
    if method != "iqr":
        raise ValueError("method must be iqr")
    if (
        isinstance(threshold, (bool, np.bool_))
        or not isinstance(threshold, Real)
        or not threshold > 0
    ):
        raise ValueError("threshold must be a positive number")
    threshold_value = float(threshold)

    analyzed: list[str] = []
    skipped_reasons: dict[str, str] = {}
    summary_rows: list[dict[str, object]] = []
    outlier_rows: list[dict[str, object]] = []
    for column in candidates:
        series = df[column]
        if not _is_numeric_non_bool(series.dtype):
            if requested is not None:
                skipped_reasons[column] = "not_numeric"
            continue
        non_missing = series.dropna()
        if non_missing.empty:
            skipped_reasons[column] = "all_missing"
            continue
        if bool(np.isinf(non_missing).any()):
            skipped_reasons[column] = "non_finite_values"
            continue
        if len(non_missing) < 2:
            skipped_reasons[column] = "insufficient_non_missing"
            continue
        if non_missing.nunique() == 1:
            skipped_reasons[column] = "constant"
            continue

        q1 = float(non_missing.quantile(0.25))
        q3 = float(non_missing.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - threshold_value * iqr
        upper_bound = q3 + threshold_value * iqr
        outlier_mask = series.lt(lower_bound) | series.gt(upper_bound)
        positions = np.flatnonzero(outlier_mask.fillna(False).to_numpy(dtype=bool))
        outlier_count = int(len(positions))
        summary_rows.append(
            {
                "column": column,
                "method": method,
                "threshold": threshold_value,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
                "outlier_rate": outlier_count / len(non_missing),
            }
        )
        for position in positions:
            outlier_rows.append(
                {
                    "column": column,
                    "row_index": df.index[position],
                    "value": series.iloc[position],
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                }
            )
        analyzed.append(column)

    return OutlierAnalysis(
        n_rows=len(df),
        requested_columns=requested,
        analyzed_columns=tuple(analyzed),
        skipped_columns=_ordered_skipped(candidates, skipped_reasons),
        skipped_reasons=skipped_reasons,
        method=method,
        threshold=threshold_value,
        summary=_typed_frame(summary_rows, _OUTLIER_SUMMARY_DTYPES),
        outliers=_typed_frame(outlier_rows, _OUTLIER_DETAILS_DTYPES),
    )


def _validate_inputs(
    df: pd.DataFrame,
    columns: Sequence[str] | None,
) -> tuple[tuple[str, ...] | None, list[str]]:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if not all(isinstance(column, str) for column in df.columns):
        raise ValueError("DataFrame column names must all be strings")
    if df.columns.has_duplicates:
        raise ValueError("duplicate DataFrame column names are not supported")
    if columns is None:
        return None, list(df.columns)

    requested = tuple(columns)
    if not all(isinstance(column, str) for column in requested):
        raise ValueError("columns must contain only strings")
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate column parameter")
    for column in requested:
        if column not in df.columns:
            raise ValueError(f"column not found: {column!r}")
    return requested, list(requested)


def _select_basic_columns(
    df: pd.DataFrame,
    candidates: Sequence[str],
    *,
    eligible: Callable[[object], bool],
    ineligible_reason: str,
    record_ineligible: bool,
) -> tuple[list[str], dict[str, str]]:
    analyzed: list[str] = []
    skipped_reasons: dict[str, str] = {}
    for column in candidates:
        if eligible(df[column].dtype):
            analyzed.append(column)
        elif record_ineligible:
            skipped_reasons[column] = ineligible_reason
    return analyzed, skipped_reasons


def _is_numeric_non_bool(dtype: object) -> bool:
    return bool(is_numeric_dtype(dtype) and not is_bool_dtype(dtype))


def _is_categorical(dtype: object) -> bool:
    return bool(
        is_object_dtype(dtype)
        or is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
        or is_bool_dtype(dtype)
    )


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _ordered_category_frequencies(series: pd.Series) -> list[tuple[object, int]]:
    first_seen = list(pd.unique(series))
    counts = series.value_counts(sort=False, dropna=False)
    count_by_value = {value: int(count) for value, count in counts.items()}
    frequencies = [(value, count_by_value[value]) for value in first_seen]
    return sorted(frequencies, key=lambda item: -item[1])


def _ordered_skipped(
    candidates: Sequence[str],
    skipped_reasons: dict[str, str],
) -> tuple[str, ...]:
    return tuple(column for column in candidates if column in skipped_reasons)


def _typed_frame(
    rows: list[dict[str, object]],
    dtypes: dict[str, str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: pd.Series([row[column] for row in rows], dtype=dtype)
            for column, dtype in dtypes.items()
        }
    )
