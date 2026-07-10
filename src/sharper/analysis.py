"""Deterministic feature, group, and target analysis for pandas DataFrames."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from scipy import stats

TASK08_MIN_GROUP_SIZE = 2
TASK08_MAX_FEATURES = 50
TASK08_MAX_CATEGORIES = 20
TASK08_MAX_TARGET_CLASSES = 20

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

_GROUP_SUMMARY_DTYPES = {
    "value": "object",
    "group": "object",
    "group_count": "int64",
    "count": "int64",
    "missing_count": "int64",
    "mean": "float64",
    "q25": "float64",
    "median": "float64",
    "q75": "float64",
}

_TARGET_NUMERIC_DETAILS_DTYPES = {
    "feature": "object",
    "target_category": "object",
    "group_count": "int64",
    "count": "int64",
    "missing_count": "int64",
    "mean": "float64",
    "q25": "float64",
    "median": "float64",
    "q75": "float64",
}

_TARGET_CATEGORY_DETAILS_DTYPES = {
    "feature": "object",
    "feature_category": "object",
    "target_category": "object",
    "count": "int64",
    "rate": "float64",
    "target_mean": "float64",
    "target_median": "float64",
}

_TARGET_TEST_DTYPES = {
    "feature": "object",
    "feature_kind": "object",
    "analysis": "object",
    "n_obs": "int64",
    "group_count": "int64",
    "statistic": "float64",
    "p_value": "float64",
    "effect_size": "float64",
    "effect_size_name": "object",
    "limitation": "object",
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


@dataclass(frozen=True)
class GroupComparison:
    """Contain a budgeted single-key comparison of numeric values.

    Missing group keys are excluded and counted separately. ``summary`` uses the
    fixed Task 08 long-form schema, while skipped values retain one deterministic
    reason each. The result has no side effects or generated artifacts.
    """

    n_rows: int
    group_by: str
    requested_values: tuple[str, ...] | None
    analyzed_values: tuple[str, ...]
    skipped_values: tuple[str, ...]
    skipped_reasons: dict[str, str]
    max_groups: int
    available_group_count: int
    displayed_group_count: int
    missing_group_count: int
    truncated: bool
    truncation_reason: str | None
    summary: pd.DataFrame


@dataclass(frozen=True)
class TargetAnalysis:
    """Contain explicit classification or regression target relationships.

    The detail and statistical-test tables use the fixed Task 08 schemas. Feature
    budgets, skips, effective samples, and exploratory limitations are recorded
    deterministically. The result never contains plots, models, paths, or files.
    """

    n_rows: int
    target: str
    task: str
    requested_features: tuple[str, ...] | None
    analyzed_features: tuple[str, ...]
    skipped_features: tuple[str, ...]
    skipped_reasons: dict[str, str]
    max_features: int
    max_categories: int
    available_feature_count: int
    truncated: bool
    truncation_reason: str | None
    numeric_details: pd.DataFrame
    category_details: pd.DataFrame
    statistical_tests: pd.DataFrame
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class _TargetFeatureResult:
    feature: str
    numeric_rows: tuple[dict[str, object], ...]
    category_rows: tuple[dict[str, object], ...]
    test_row: dict[str, object]
    chi_square_expected_counts_small: bool = False


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


def compare_groups(
    df: pd.DataFrame,
    group_by: str,
    *,
    values: Sequence[str] | None = None,
    max_groups: int = 20,
) -> GroupComparison:
    """Compare numeric values across one categorical group key.

    Parameters
    ----------
    df
        DataFrame with unique string column names.
    group_by
        Positional-or-keyword categorical column used as the single group key.
    values
        Optional real numeric non-boolean columns in caller order. When omitted,
        all eligible columns are selected in DataFrame order; complex columns are
        excluded.
    max_groups
        Positive non-boolean integer limiting displayed groups.

    Returns
    -------
    GroupComparison
        Fixed-schema summaries, group-budget metadata, and skipped value reasons.

    Raises
    ------
    ValueError
        If the DataFrame, group key, explicit values, or group budget are invalid.

    Notes
    -----
    Missing group keys are excluded. Missing numeric values are counted but not
    used in statistics. A value containing infinity is skipped in full. The input
    DataFrame is never modified and the function has no external side effects.

    Examples
    --------
    >>> import pandas as pd
    >>> from sharper import compare_groups
    >>> result = compare_groups(pd.DataFrame({"g": ["a", "b"], "x": [1, 2]}), "g")
    >>> result.summary["mean"].tolist()
    [1.0, 2.0]
    """
    _validate_task08_dataframe(df)
    if not isinstance(group_by, str):
        raise ValueError("group_by must be a string")
    if group_by not in df.columns:
        raise ValueError(f"group column not found: {group_by!r}")
    if not _is_categorical(df[group_by].dtype):
        raise ValueError("group_by must be categorical")
    if not _is_strict_int(max_groups) or max_groups < 1:
        raise ValueError("max_groups must be an integer >= 1")

    non_missing_groups = df[group_by].dropna()
    if non_missing_groups.empty:
        raise ValueError("group_by must contain at least one non-missing value")
    group_frequencies = _ordered_category_frequencies(non_missing_groups)
    displayed_groups = [group for group, _ in group_frequencies[:max_groups]]
    available_group_count = len(group_frequencies)

    requested_values, candidates = _validate_task08_columns(
        df,
        values,
        member_error="values must contain only strings",
        duplicate_error="duplicate value column parameter",
        missing_label="value column not found",
    )
    if requested_values is None:
        candidates = [
            column
            for column in df.columns
            if column != group_by and _is_task08_real_numeric_non_bool(df[column].dtype)
        ]
    else:
        if group_by in requested_values:
            raise ValueError("group_by must not appear in values")
        if any(
            not _is_task08_real_numeric_non_bool(df[column].dtype)
            for column in candidates
        ):
            raise ValueError("values must contain only numeric columns")

    displayed_mask = df[group_by].isin(displayed_groups)
    rows: list[dict[str, object]] = []
    analyzed: list[str] = []
    skipped_reasons: dict[str, str] = {}
    for value in candidates:
        displayed_non_missing = df.loc[displayed_mask, value].dropna()
        if displayed_non_missing.empty:
            skipped_reasons[value] = "all_missing"
            continue
        if _contains_infinity(df[value].dropna()):
            skipped_reasons[value] = "non_finite_values"
            continue
        if len(displayed_non_missing) < 2:
            skipped_reasons[value] = "insufficient_non_missing"
            continue
        if displayed_non_missing.nunique() == 1:
            skipped_reasons[value] = "constant"
            continue

        for group in displayed_groups:
            group_mask = _value_mask(df[group_by], group)
            group_values = df.loc[group_mask, value]
            group_count = int(group_mask.sum())
            finite_values = group_values.dropna()
            count = int(len(finite_values))
            rows.append(
                {
                    "value": value,
                    "group": group,
                    "group_count": group_count,
                    "count": count,
                    "missing_count": group_count - count,
                    "mean": finite_values.mean() if count else np.nan,
                    "q25": finite_values.quantile(0.25) if count else np.nan,
                    "median": finite_values.quantile(0.50) if count else np.nan,
                    "q75": finite_values.quantile(0.75) if count else np.nan,
                }
            )
        analyzed.append(value)

    truncated = available_group_count > max_groups
    return GroupComparison(
        n_rows=len(df),
        group_by=group_by,
        requested_values=requested_values,
        analyzed_values=tuple(analyzed),
        skipped_values=_ordered_skipped(candidates, skipped_reasons),
        skipped_reasons=skipped_reasons,
        max_groups=max_groups,
        available_group_count=available_group_count,
        displayed_group_count=len(displayed_groups),
        missing_group_count=int(df[group_by].isna().sum()),
        truncated=truncated,
        truncation_reason="exceeds_max_groups" if truncated else None,
        summary=_typed_frame(rows, _GROUP_SUMMARY_DTYPES),
    )


def analyze_target_relationships(
    df: pd.DataFrame,
    target: str,
    *,
    task: Literal["classification", "regression"],
    features: Sequence[str] | None = None,
) -> TargetAnalysis:
    """Analyze explicit classification or regression target relationships.

    Parameters
    ----------
    df
        DataFrame with unique string column names.
    target
        Positional-or-keyword target column. Missing target rows are excluded.
    task
        Required keyword-only task, ``"classification"`` or ``"regression"``.
    features
        Optional features in caller order. When omitted, eligible numeric and
        categorical columns are selected in DataFrame order.

    Returns
    -------
    TargetAnalysis
        Fixed-schema details and statistical tests with budgets and limitations.

    Raises
    ------
    ValueError
        If the DataFrame, target, task, requested features, or target values violate
        the frozen Task 08 input contract.

    Notes
    -----
    Classification uses Kruskal-Wallis for numeric features and Chi-square with
    Cramer's V for categorical features. Regression uses Pearson correlation for
    numeric features and Kruskal-Wallis for categorical features. Results are
    exploratory, p-values are unadjusted, and the input is never modified.

    Examples
    --------
    >>> import pandas as pd
    >>> from sharper import analyze_target_relationships
    >>> frame = pd.DataFrame({"x": [1, 2, 3, 4], "y": ["a", "a", "b", "b"]})
    >>> result = analyze_target_relationships(frame, "y", task="classification")
    >>> result.statistical_tests["analysis"].tolist()
    ['kruskal_wallis']
    """
    _validate_task08_dataframe(df)
    if not isinstance(target, str):
        raise ValueError("target must be a string")
    if target not in df.columns:
        raise ValueError(f"target column not found: {target!r}")
    if not isinstance(task, str) or task not in {"classification", "regression"}:
        raise ValueError("task must be classification or regression")

    requested_features, candidates = _validate_task08_columns(
        df,
        features,
        member_error="features must contain only strings",
        duplicate_error="duplicate feature column parameter",
        missing_label="feature column not found",
    )
    if requested_features is not None and target in requested_features:
        raise ValueError("target must not appear in features")
    if requested_features is None:
        candidates = [
            column
            for column in df.columns
            if column != target
            and (
                _is_task08_real_numeric_non_bool(df[column].dtype)
                or _is_categorical(df[column].dtype)
            )
        ]

    if task == "classification":
        _validate_classification_target(df[target])
    else:
        _validate_regression_target(df[target])

    skipped_reasons: dict[str, str] = {}
    successful: list[_TargetFeatureResult] = []
    available_feature_count = 0
    for feature in candidates:
        result, reason, available = _analyze_target_feature(
            df,
            target=target,
            feature=feature,
            task=task,
            explicit=requested_features is not None,
        )
        if available:
            available_feature_count += 1
        if reason is not None:
            skipped_reasons[feature] = reason
        elif result is not None:
            successful.append(result)

    retained = successful[:TASK08_MAX_FEATURES]
    for result in successful[TASK08_MAX_FEATURES:]:
        skipped_reasons[result.feature] = "exceeds_max_features"

    numeric_rows = [row for result in retained for row in result.numeric_rows]
    category_rows = [row for result in retained for row in result.category_rows]
    test_rows = [result.test_row for result in retained]
    truncated = bool(successful[TASK08_MAX_FEATURES:])
    limitations: tuple[str, ...]
    if test_rows:
        limitation_values = ["exploratory_unadjusted_p_values"]
        if any(result.chi_square_expected_counts_small for result in retained):
            limitation_values.append("chi_square_expected_counts_may_be_small")
        limitations = tuple(limitation_values)
    else:
        limitations = ()

    return TargetAnalysis(
        n_rows=len(df),
        target=target,
        task=task,
        requested_features=requested_features,
        analyzed_features=tuple(result.feature for result in retained),
        skipped_features=_ordered_skipped(candidates, skipped_reasons),
        skipped_reasons=skipped_reasons,
        max_features=TASK08_MAX_FEATURES,
        max_categories=TASK08_MAX_CATEGORIES,
        available_feature_count=available_feature_count,
        truncated=truncated,
        truncation_reason="exceeds_max_features" if truncated else None,
        numeric_details=_typed_frame(numeric_rows, _TARGET_NUMERIC_DETAILS_DTYPES),
        category_details=_typed_frame(category_rows, _TARGET_CATEGORY_DETAILS_DTYPES),
        statistical_tests=_typed_frame(test_rows, _TARGET_TEST_DTYPES),
        limitations=limitations,
    )


def _validate_task08_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if not all(isinstance(column, str) for column in df.columns):
        raise ValueError("DataFrame column names must all be strings")
    if df.columns.has_duplicates:
        raise ValueError("duplicate DataFrame column names are not supported")


def _validate_task08_columns(
    df: pd.DataFrame,
    columns: Sequence[str] | None,
    *,
    member_error: str,
    duplicate_error: str,
    missing_label: str,
) -> tuple[tuple[str, ...] | None, list[str]]:
    if columns is None:
        return None, list(df.columns)
    requested = tuple(columns)
    if not all(isinstance(column, str) for column in requested):
        raise ValueError(member_error)
    if len(set(requested)) != len(requested):
        raise ValueError(duplicate_error)
    for column in requested:
        if column not in df.columns:
            raise ValueError(f"{missing_label}: {column!r}")
    return requested, list(requested)


def _validate_classification_target(target: pd.Series) -> None:
    dtype = target.dtype
    is_supported = _is_categorical(dtype) or _is_task08_real_numeric_non_bool(dtype)
    if not is_supported:
        raise ValueError(
            "classification target must be categorical or low-cardinality numeric"
        )
    non_missing = target.dropna()
    if _is_task08_real_numeric_non_bool(dtype) and _contains_infinity(non_missing):
        raise ValueError("classification target must contain only finite values")
    if non_missing.empty:
        raise ValueError("classification target must contain non-missing values")
    class_count = int(non_missing.nunique())
    if class_count < 2:
        raise ValueError("classification target must contain at least two classes")
    if class_count > TASK08_MAX_TARGET_CLASSES:
        raise ValueError("classification target must contain at most 20 classes")


def _validate_regression_target(target: pd.Series) -> None:
    if not _is_task08_real_numeric_non_bool(target.dtype):
        raise ValueError("regression target must be numeric")
    non_missing = target.dropna()
    if _contains_infinity(non_missing):
        raise ValueError(
            "regression target must contain only finite non-missing values"
        )
    if len(non_missing) < 3:
        raise ValueError("regression target must contain at least three finite values")
    if non_missing.nunique() == 1:
        raise ValueError("regression target must not be constant")


def _analyze_target_feature(
    df: pd.DataFrame,
    *,
    target: str,
    feature: str,
    task: str,
    explicit: bool,
) -> tuple[_TargetFeatureResult | None, str | None, bool]:
    dtype = df[feature].dtype
    if _is_task08_real_numeric_non_bool(dtype):
        if task == "classification":
            return _classification_numeric(df, target=target, feature=feature)
        return _regression_numeric(df, target=target, feature=feature)
    if _is_categorical(dtype):
        if task == "classification":
            return _classification_categorical(df, target=target, feature=feature)
        return _regression_categorical(df, target=target, feature=feature)
    if explicit:
        return None, "unsupported_dtype", False
    return None, None, False


def _classification_numeric(
    df: pd.DataFrame,
    *,
    target: str,
    feature: str,
) -> tuple[_TargetFeatureResult | None, str | None, bool]:
    target_present = df[target].notna()
    feature_present = df[feature].notna()
    complete_mask = target_present & feature_present
    if not bool(complete_mask.any()):
        return None, "all_missing", False
    if _contains_infinity(df[feature].dropna()):
        return None, "non_finite_values", False
    if int(complete_mask.sum()) < 4:
        return None, "insufficient_non_missing", False
    complete_values = df.loc[complete_mask, feature]
    if complete_values.nunique() == 1:
        return None, "constant", False

    target_order = _ordered_values(df.loc[target_present, target])
    retained: list[tuple[object, pd.Series]] = []
    for category in target_order:
        mask = complete_mask & _value_mask(df[target], category)
        values = df.loc[mask, feature]
        if len(values) >= TASK08_MIN_GROUP_SIZE:
            retained.append((category, values))
    if len(retained) < 2:
        return None, "insufficient_groups", False

    available = True
    retained_values = pd.concat([values for _, values in retained], ignore_index=True)
    if retained_values.nunique() == 1:
        return None, "statistical_test_not_applicable", available
    statistic, p_value = stats.kruskal(*(values.to_numpy() for _, values in retained))
    n_obs = sum(len(values) for _, values in retained)
    group_count = len(retained)
    effect_size = _epsilon_squared(statistic, n_obs, group_count)
    if not _all_finite(statistic, p_value, effect_size):
        return None, "statistical_test_not_applicable", available

    numeric_rows: list[dict[str, object]] = []
    for category, values in retained:
        target_group_mask = df[target].notna() & _value_mask(df[target], category)
        group_count_rows = int(target_group_mask.sum())
        count = int(len(values))
        numeric_rows.append(
            {
                "feature": feature,
                "target_category": category,
                "group_count": group_count_rows,
                "count": count,
                "missing_count": group_count_rows - count,
                "mean": values.mean(),
                "q25": values.quantile(0.25),
                "median": values.quantile(0.50),
                "q75": values.quantile(0.75),
            }
        )
    return (
        _TargetFeatureResult(
            feature=feature,
            numeric_rows=tuple(numeric_rows),
            category_rows=(),
            test_row=_test_row(
                feature=feature,
                feature_kind="numeric",
                analysis="kruskal_wallis",
                n_obs=n_obs,
                group_count=group_count,
                statistic=statistic,
                p_value=p_value,
                effect_size=effect_size,
                effect_size_name="epsilon_squared",
            ),
        ),
        None,
        available,
    )


def _classification_categorical(
    df: pd.DataFrame,
    *,
    target: str,
    feature: str,
) -> tuple[_TargetFeatureResult | None, str | None, bool]:
    complete_mask = df[target].notna() & df[feature].notna()
    if not bool(complete_mask.any()):
        return None, "all_missing", False
    if int(complete_mask.sum()) < 4:
        return None, "insufficient_non_missing", False
    complete_feature = df.loc[complete_mask, feature]
    complete_target = df.loc[complete_mask, target]
    feature_order = _ordered_values(complete_feature)
    target_order = [
        category
        for category in _ordered_values(df[target].dropna())
        if bool(_value_mask(complete_target, category).any())
    ]
    if len(feature_order) == 1:
        return None, "constant", False
    insufficient_groups = len(target_order) < 2
    if len(feature_order) > TASK08_MAX_CATEGORIES:
        return None, "exceeds_max_categories", not insufficient_groups
    if insufficient_groups:
        return None, "insufficient_groups", False

    available = True

    table = np.array(
        [
            [
                int(
                    (
                        _value_mask(complete_feature, feature_category)
                        & _value_mask(complete_target, target_category)
                    ).sum()
                )
                for target_category in target_order
            ]
            for feature_category in feature_order
        ],
        dtype="int64",
    )
    statistic, p_value, _, expected = stats.chi2_contingency(table)
    n_obs = int(table.sum())
    denominator = n_obs * min(table.shape[0] - 1, table.shape[1] - 1)
    if denominator == 0:
        return None, "insufficient_groups", False
    effect_size = _cramers_v(statistic, denominator)
    if not _all_finite(statistic, p_value, effect_size):
        return None, "statistical_test_not_applicable", available

    expected_small = bool((expected < 5).any())
    limitation = "exploratory_unadjusted_p_value"
    if expected_small:
        limitation += "; chi_square_expected_counts_may_be_small"
    category_rows: list[dict[str, object]] = []
    for feature_position, feature_category in enumerate(feature_order):
        row_total = int(table[feature_position].sum())
        for target_position, target_category in enumerate(target_order):
            count = int(table[feature_position, target_position])
            category_rows.append(
                {
                    "feature": feature,
                    "feature_category": feature_category,
                    "target_category": target_category,
                    "count": count,
                    "rate": count / row_total,
                    "target_mean": np.nan,
                    "target_median": np.nan,
                }
            )
    return (
        _TargetFeatureResult(
            feature=feature,
            numeric_rows=(),
            category_rows=tuple(category_rows),
            test_row=_test_row(
                feature=feature,
                feature_kind="categorical",
                analysis="chi_square",
                n_obs=n_obs,
                group_count=len(feature_order),
                statistic=statistic,
                p_value=p_value,
                effect_size=effect_size,
                effect_size_name="cramers_v",
                limitation=limitation,
            ),
            chi_square_expected_counts_small=expected_small,
        ),
        None,
        available,
    )


def _regression_numeric(
    df: pd.DataFrame,
    *,
    target: str,
    feature: str,
) -> tuple[_TargetFeatureResult | None, str | None, bool]:
    complete_mask = df[target].notna() & df[feature].notna()
    if not bool(complete_mask.any()):
        return None, "all_missing", False
    if _contains_infinity(df[feature].dropna()):
        return None, "non_finite_values", False
    if int(complete_mask.sum()) < 3:
        return None, "insufficient_non_missing", False
    target_values = df.loc[complete_mask, target]
    feature_values = df.loc[complete_mask, feature]
    if target_values.nunique() == 1 or feature_values.nunique() == 1:
        return None, "constant", False

    available = True
    statistic, p_value = stats.pearsonr(feature_values, target_values)
    effect_size = abs(float(statistic))
    if not _all_finite(statistic, p_value, effect_size):
        return None, "statistical_test_not_applicable", available
    return (
        _TargetFeatureResult(
            feature=feature,
            numeric_rows=(),
            category_rows=(),
            test_row=_test_row(
                feature=feature,
                feature_kind="numeric",
                analysis="pearson",
                n_obs=len(feature_values),
                group_count=0,
                statistic=statistic,
                p_value=p_value,
                effect_size=effect_size,
                effect_size_name="absolute_pearson_r",
            ),
        ),
        None,
        available,
    )


def _regression_categorical(
    df: pd.DataFrame,
    *,
    target: str,
    feature: str,
) -> tuple[_TargetFeatureResult | None, str | None, bool]:
    complete_mask = df[target].notna() & df[feature].notna()
    if not bool(complete_mask.any()):
        return None, "all_missing", False
    if int(complete_mask.sum()) < 4:
        return None, "insufficient_non_missing", False
    complete_feature = df.loc[complete_mask, feature]
    complete_target = df.loc[complete_mask, target]
    feature_order = _ordered_values(complete_feature)
    if len(feature_order) == 1:
        return None, "constant", False

    retained: list[tuple[object, pd.Series]] = []
    for category in feature_order:
        values = complete_target.loc[_value_mask(complete_feature, category)]
        if len(values) >= TASK08_MIN_GROUP_SIZE:
            retained.append((category, values))
    insufficient_groups = len(retained) < 2
    if len(feature_order) > TASK08_MAX_CATEGORIES:
        return None, "exceeds_max_categories", not insufficient_groups
    if insufficient_groups:
        return None, "insufficient_groups", False

    available = True
    retained_values = pd.concat([values for _, values in retained], ignore_index=True)
    if retained_values.nunique() == 1:
        return None, "statistical_test_not_applicable", available
    statistic, p_value = stats.kruskal(*(values.to_numpy() for _, values in retained))
    n_obs = sum(len(values) for _, values in retained)
    group_count = len(retained)
    effect_size = _epsilon_squared(statistic, n_obs, group_count)
    if not _all_finite(statistic, p_value, effect_size):
        return None, "statistical_test_not_applicable", available

    category_rows: list[dict[str, object]] = []
    for category, values in retained:
        category_rows.append(
            {
                "feature": feature,
                "feature_category": category,
                "target_category": None,
                "count": len(values),
                "rate": len(values) / n_obs,
                "target_mean": values.mean(),
                "target_median": values.median(),
            }
        )
    return (
        _TargetFeatureResult(
            feature=feature,
            numeric_rows=(),
            category_rows=tuple(category_rows),
            test_row=_test_row(
                feature=feature,
                feature_kind="categorical",
                analysis="kruskal_wallis",
                n_obs=n_obs,
                group_count=group_count,
                statistic=statistic,
                p_value=p_value,
                effect_size=effect_size,
                effect_size_name="epsilon_squared",
            ),
        ),
        None,
        available,
    )


def _test_row(
    *,
    feature: str,
    feature_kind: str,
    analysis: str,
    n_obs: int,
    group_count: int,
    statistic: object,
    p_value: object,
    effect_size: object,
    effect_size_name: str,
    limitation: str = "exploratory_unadjusted_p_value",
) -> dict[str, object]:
    return {
        "feature": feature,
        "feature_kind": feature_kind,
        "analysis": analysis,
        "n_obs": n_obs,
        "group_count": group_count,
        "statistic": statistic,
        "p_value": p_value,
        "effect_size": effect_size,
        "effect_size_name": effect_size_name,
        "limitation": limitation,
    }


def _ordered_values(series: pd.Series) -> list[object]:
    return list(pd.unique(series))


def _value_mask(series: pd.Series, value: object) -> pd.Series:
    matches: list[bool] = []
    for item in series:
        missing = pd.isna(item)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            matches.append(False)
            continue
        equal = item == value
        matches.append(bool(equal) if isinstance(equal, (bool, np.bool_)) else False)
    return pd.Series(matches, index=series.index, dtype="bool")


def _contains_infinity(series: pd.Series) -> bool:
    if not len(series):
        return False
    values = np.asarray(series, dtype=np.complex128)
    return bool(np.isinf(values).any())


def _all_finite(*values: object) -> bool:
    return all(bool(np.isfinite(value)) for value in values)


def _epsilon_squared(statistic: object, n_obs: int, group_count: int) -> float:
    return max(
        0.0,
        (float(statistic) - group_count + 1) / (n_obs - group_count),
    )


def _cramers_v(statistic: object, denominator: int) -> float:
    return float(np.sqrt(float(statistic) / denominator))


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


def _is_task08_real_numeric_non_bool(dtype: object) -> bool:
    return bool(
        is_numeric_dtype(dtype)
        and not is_bool_dtype(dtype)
        and not is_complex_dtype(dtype)
    )


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
