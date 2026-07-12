"""Deterministic, static analytical figures for frozen Sharper result types."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from sharper.analysis import (
    CorrelationAnalysis,
    GroupComparison,
    OutlierAnalysis,
    TargetAnalysis,
)
from sharper.evaluation import (
    ClassificationEvaluation,
    RegressionEvaluation,
    _validate_classification_evaluation,
    _validate_regression_evaluation,
)

_BLUE = "#4C78A8"
_PALETTE = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
)
_CORR_COLUMNS = ("column_a", "column_b", "method", "correlation", "n_pairs")
_OUTLIER_COLUMNS = (
    "column",
    "method",
    "threshold",
    "lower_bound",
    "upper_bound",
    "outlier_count",
    "outlier_rate",
)
_OUTLIER_DETAIL_COLUMNS = ("column", "row_index", "value", "lower_bound", "upper_bound")
_GROUP_COLUMNS = (
    "value",
    "group",
    "group_count",
    "count",
    "missing_count",
    "mean",
    "q25",
    "median",
    "q75",
)
_NUMERIC_DETAIL_COLUMNS = (
    "feature",
    "target_category",
    "group_count",
    "count",
    "missing_count",
    "mean",
    "q25",
    "median",
    "q75",
)
_CATEGORY_DETAIL_COLUMNS = (
    "feature",
    "feature_category",
    "target_category",
    "count",
    "rate",
    "target_mean",
    "target_median",
)
_TEST_COLUMNS = (
    "feature",
    "feature_kind",
    "analysis",
    "n_obs",
    "group_count",
    "statistic",
    "p_value",
    "effect_size",
    "effect_size_name",
    "limitation",
)
_CORR_DTYPES = ("object", "object", "object", "float64", "int64")
_OUTLIER_DTYPES = (
    "object",
    "object",
    "float64",
    "float64",
    "float64",
    "int64",
    "float64",
)
_OUTLIER_DETAIL_DTYPES = ("object", "object", "float64", "float64", "float64")
_GROUP_DTYPES = (
    "object",
    "object",
    "int64",
    "int64",
    "int64",
    "float64",
    "float64",
    "float64",
    "float64",
)
_NUMERIC_DETAIL_DTYPES = (
    "object",
    "object",
    "int64",
    "int64",
    "int64",
    "float64",
    "float64",
    "float64",
    "float64",
)
_CATEGORY_DETAIL_DTYPES = (
    "object",
    "object",
    "object",
    "int64",
    "float64",
    "float64",
    "float64",
)
_TEST_DTYPES = (
    "object",
    "object",
    "object",
    "int64",
    "int64",
    "float64",
    "float64",
    "float64",
    "object",
    "object",
)


@dataclass(frozen=True)
class PlotResult:
    """A caller-owned matplotlib Figure and its frozen analytical metadata."""

    figure: Figure
    chart_type: str
    title: str
    source: str
    item: str | None
    metadata: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PlotCollection:
    """A deterministic bounded collection of analytical PlotResult objects."""

    requested_count: int
    available_count: int
    actual_count: int
    truncated: bool
    truncation_reason: str | None
    plots: tuple[PlotResult, ...]


def _text(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".12g")
    return str(value)


def _json(values: list[object]) -> str:
    return json.dumps(
        [str(value) for value in values], ensure_ascii=False, separators=(",", ":")
    )


def _meta(**values: object) -> tuple[tuple[str, str], ...]:
    return tuple((key, _text(value)) for key, value in values.items())


def _validate_df(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if not all(isinstance(column, str) for column in df.columns):
        raise ValueError("DataFrame column names must all be strings")
    if df.columns.duplicated().any():
        raise ValueError("duplicate DataFrame column names are not supported")


def _validate_int(value: int, name: str, maximum: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise ValueError(f"{name} must be an integer from 1 to {maximum}")


def _collection(
    requested: int, available: int, plots: list[PlotResult]
) -> PlotCollection:
    return PlotCollection(
        requested,
        available,
        len(plots),
        available > requested,
        "max_plots" if available > requested else None,
        tuple(plots),
    )


def _classification_collection(plots: list[PlotResult]) -> PlotCollection:
    """Build Task 11's fixed two-slot evaluation collection."""
    return PlotCollection(
        requested_count=2,
        available_count=len(plots),
        actual_count=len(plots),
        truncated=False,
        truncation_reason=None,
        plots=tuple(plots),
    )


def _axes(title: str, xlabel: str, ylabel: str) -> tuple[Figure, object]:
    figure, axes = plt.subplots()
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    return figure, axes


def _bar_grid(axes: object) -> None:
    axes.grid(axis="y", linestyle="--", alpha=0.3)


def plot_distributions(
    df: pd.DataFrame, *, max_plots: int = 20, sample_size: int = 10_000
) -> PlotCollection:
    """Plot one bounded histogram or category-frequency Figure per supported column."""
    _validate_df(df)
    _validate_int(max_plots, "max_plots", 20)
    _validate_int(sample_size, "sample_size", 10_000)
    candidates: list[tuple[str, str, pd.Series]] = []
    for column in df.columns:
        series = df[column]
        if (
            is_numeric_dtype(series.dtype)
            and not is_bool_dtype(series.dtype)
            and not is_complex_dtype(series.dtype)
        ):
            finite = series.dropna().astype(float)
            finite = finite[np.isfinite(finite)]
            if not finite.empty:
                candidates.append((column, "numeric", series))
        elif (
            is_object_dtype(series.dtype)
            or is_string_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or is_bool_dtype(series.dtype)
        ):
            non_missing = series.dropna()
            if not non_missing.empty:
                try:
                    list(pd.unique(non_missing))
                except TypeError as error:
                    raise ValueError(
                        "categorical column contains unhashable values"
                    ) from error
                candidates.append((column, "category", series))
    plots: list[PlotResult] = []
    for column, kind, series in candidates[:max_plots]:
        if kind == "numeric":
            values = series.dropna().astype(float)
            finite = values[np.isfinite(values)]
            sampled = finite.iloc[:sample_size]
            if len(sampled) < 2 or sampled.nunique() == 1:
                bins = 1
            else:
                iqr = sampled.quantile(0.75) - sampled.quantile(0.25)
                bins = (
                    ceil(
                        (sampled.max() - sampled.min())
                        / (2 * iqr / len(sampled) ** (1 / 3))
                    )
                    if iqr > 0
                    else ceil(sqrt(len(sampled)))
                )
                bins = min(50, max(1, bins))
            title = f"{column} distribution"
            figure, axes = _axes(title, column, "Count")
            sns.histplot(x=sampled, bins=bins, kde=False, color=_BLUE, ax=axes)
            _bar_grid(axes)
            metadata = _meta(
                column=column,
                dtype=str(series.dtype),
                finite_count=len(finite),
                missing_count=series.isna().sum(),
                non_finite_count=len(values) - len(finite),
                sample_size_requested=sample_size,
                sample_size_actual=len(sampled),
                bins=bins,
            )
            plots.append(
                PlotResult(
                    figure,
                    "distribution_histogram",
                    title,
                    "dataframe",
                    column,
                    metadata,
                )
            )
        else:
            non_missing = series.dropna()
            first: dict[object, int] = {}
            for index, value in enumerate(non_missing):
                first.setdefault(value, index)
            counts = non_missing.value_counts(dropna=True)
            ordered = sorted(
                counts.index.tolist(),
                key=lambda value: (-int(counts[value]), first[value]),
            )
            shown = ordered[:20]
            title = f"{column} category frequency"
            figure, axes = _axes(title, column, "Count")
            positions = np.arange(len(shown))
            axes.bar(positions, [int(counts[value]) for value in shown], color=_BLUE)
            axes.set_xticks(positions, [str(value) for value in shown])
            _bar_grid(axes)
            metadata = _meta(
                column=column,
                dtype=str(series.dtype),
                non_missing_count=len(non_missing),
                missing_count=series.isna().sum(),
                available_categories=len(ordered),
                displayed_categories=_json(shown),
                truncated_categories=len(ordered) > 20,
                category_limit=20,
            )
            plots.append(
                PlotResult(
                    figure,
                    "distribution_categories",
                    title,
                    "dataframe",
                    column,
                    metadata,
                )
            )
    return _collection(max_plots, len(candidates), plots)


def plot_missingness(df: pd.DataFrame, *, max_columns: int = 50) -> PlotResult:
    """Plot missing rates for the first bounded set of DataFrame columns."""
    _validate_df(df)
    _validate_int(max_columns, "max_columns", 50)
    columns = list(df.columns[:max_columns])
    rates = [
        float(df[column].isna().sum() / len(df)) if len(df) else 0.0
        for column in columns
    ]
    title = "Missingness by column"
    figure, axes = _axes(title, "Column", "Missing rate")
    positions = np.arange(len(columns))
    axes.bar(positions, rates, color=_BLUE)
    axes.set_xticks(positions, columns)
    axes.set_ylim(0.0, 1.0)
    _bar_grid(axes)
    return PlotResult(
        figure,
        "missingness_rate",
        title,
        "dataframe",
        None,
        _meta(
            n_rows=len(df),
            requested_columns=max_columns,
            available_columns=len(df.columns),
            analyzed_columns=_json(columns),
            truncated_columns=len(df.columns) > max_columns,
            truncation_reason="max_columns"
            if len(df.columns) > max_columns
            else "none",
        ),
    )


def _expect(result: object, kind: type, message: str) -> None:
    if not isinstance(result, kind):
        raise ValueError(message)


def _schema(
    frame: pd.DataFrame, columns: tuple[str, ...], dtypes: tuple[str, ...], message: str
) -> None:
    if tuple(frame.columns) != columns or tuple(map(str, frame.dtypes)) != dtypes:
        raise ValueError(message)


def _validate_correlation_metadata(result: CorrelationAnalysis) -> tuple[str, ...]:
    """Reject malformed frozen CorrelationAnalysis metadata before plotting."""
    message = "correlation result has invalid schema"
    columns = result.analyzed_columns
    if (
        not isinstance(columns, tuple)
        or not all(isinstance(column, str) for column in columns)
        or len(set(columns)) != len(columns)
        or isinstance(result.max_columns, bool)
        or not isinstance(result.max_columns, int)
        or not 2 <= result.max_columns <= 50
        or len(columns) > result.max_columns
        or isinstance(result.min_periods, bool)
        or not isinstance(result.min_periods, int)
        or result.min_periods < 2
        or result.method not in {"pearson", "spearman"}
        or not isinstance(result.truncated, bool)
        or not isinstance(result.skipped_columns, tuple)
        or not all(isinstance(column, str) for column in result.skipped_columns)
        or len(set(result.skipped_columns)) != len(result.skipped_columns)
        or not isinstance(result.skipped_reasons, dict)
        or set(result.skipped_reasons) != set(result.skipped_columns)
        or any(
            reason
            not in {
                "not_numeric",
                "all_missing",
                "constant",
                "insufficient_non_missing",
                "exceeds_max_columns",
            }
            for reason in result.skipped_reasons.values()
        )
    ):
        raise ValueError(message)
    has_excess = any(
        reason == "exceeds_max_columns" for reason in result.skipped_reasons.values()
    )
    if (
        result.truncated and (len(columns) != result.max_columns or not has_excess)
    ) or (not result.truncated and has_excess):
        raise ValueError(message)
    return columns


def plot_correlations(result: CorrelationAnalysis) -> PlotResult:
    """Render the supplied long-form correlation result without recomputation."""
    _expect(result, CorrelationAnalysis, "result must be a CorrelationAnalysis")
    _schema(
        result.correlations,
        _CORR_COLUMNS,
        _CORR_DTYPES,
        "correlation result has invalid schema",
    )
    cols = list(_validate_correlation_metadata(result))
    seen: set[frozenset[str]] = set()
    matrix = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    for col in cols:
        matrix.loc[col, col] = 1.0
    for row in result.correlations.itertuples(index=False):
        if (
            row.column_a not in cols
            or row.column_b not in cols
            or row.column_a == row.column_b
            or row.method != result.method
            or not np.isfinite(row.correlation)
        ):
            raise ValueError("correlation result has invalid schema")
        key = frozenset((row.column_a, row.column_b))
        if key in seen:
            raise ValueError("correlation result has invalid schema")
        seen.add(key)
        matrix.loc[row.column_a, row.column_b] = row.correlation
        matrix.loc[row.column_b, row.column_a] = row.correlation
    title = f"Correlation heatmap ({result.method})"
    figure, axes = _axes(title, "Feature", "Feature")
    if cols:
        sns.heatmap(
            matrix,
            mask=matrix.isna(),
            cmap="vlag",
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Correlation"},
            ax=axes,
        )
        axes.set_title(title)
        axes.set_xlabel("Feature")
        axes.set_ylabel("Feature")
    missing = [
        [a, b]
        for i, a in enumerate(cols)
        for b in cols[i + 1 :]
        if frozenset((a, b)) not in seen
    ]
    return PlotResult(
        figure,
        "correlation_heatmap",
        title,
        "correlation_analysis",
        None,
        _meta(
            method=result.method,
            analyzed_columns=_json(cols),
            pair_rows=len(result.correlations),
            missing_pairs=json.dumps(
                missing, ensure_ascii=False, separators=(",", ":")
            ),
            input_max_columns=result.max_columns,
            input_truncated=result.truncated,
            annotation_format=".2f",
        ),
    )


def plot_outliers(result: OutlierAnalysis, *, max_plots: int = 20) -> PlotCollection:
    """Plot supplied IQR outlier rates without reading raw observations."""
    _expect(result, OutlierAnalysis, "result must be an OutlierAnalysis")
    _validate_int(max_plots, "max_plots", 20)
    _schema(
        result.summary,
        _OUTLIER_COLUMNS,
        _OUTLIER_DTYPES,
        "outlier result has invalid schema",
    )
    _schema(
        result.outliers,
        _OUTLIER_DETAIL_COLUMNS,
        _OUTLIER_DETAIL_DTYPES,
        "outlier result has invalid schema",
    )
    if result.summary["column"].duplicated().any() or set(
        result.summary["column"]
    ) != set(result.analyzed_columns):
        raise ValueError("outlier result has invalid schema")
    if not set(result.outliers["column"]).issubset(result.analyzed_columns):
        raise ValueError("outlier result has invalid schema")
    summary = result.summary.set_index("column").loc[list(result.analyzed_columns)]
    for column, row in summary.iterrows():
        outlier_count = row["outlier_count"]
        details = result.outliers[result.outliers["column"] == column]
        lower_matches = details["lower_bound"].eq(row["lower_bound"]) | (
            details["lower_bound"].isna() & pd.isna(row["lower_bound"])
        )
        upper_matches = details["upper_bound"].eq(row["upper_bound"]) | (
            details["upper_bound"].isna() & pd.isna(row["upper_bound"])
        )
        if (
            result.method != "iqr"
            or row["method"] != result.method
            or isinstance(result.threshold, (bool, np.bool_))
            or not isinstance(result.threshold, (int, float, np.integer, np.floating))
            or not np.isfinite(float(result.threshold))
            or float(result.threshold) <= 0
            or not np.isfinite(row["threshold"])
            or float(row["threshold"]) != float(result.threshold)
            or not np.isfinite(row["lower_bound"])
            or not np.isfinite(row["upper_bound"])
            or row["lower_bound"] > row["upper_bound"]
            or isinstance(outlier_count, (bool, np.bool_))
            or not isinstance(outlier_count, (int, np.integer))
            or outlier_count < 0
            or len(details) != outlier_count
            or not bool(lower_matches.all())
            or not bool(upper_matches.all())
        ):
            raise ValueError("outlier result has invalid schema")
    displayed = list(result.analyzed_columns[:max_plots])
    plots: list[PlotResult] = []
    for column in displayed:
        row = summary.loc[column]
        title = f"{column} outlier rate (IQR)"
        figure, axes = _axes(title, column, "Outlier rate")
        axes.bar([0], [row.outlier_rate], color=_BLUE)
        axes.set_xticks([0], [column])
        axes.set_ylim(0.0, 1.0)
        _bar_grid(axes)
        plots.append(
            PlotResult(
                figure,
                "outlier_rate",
                title,
                "outlier_analysis",
                column,
                _meta(
                    displayed_features=_json(displayed),
                    truncated_features=len(result.analyzed_columns) > max_plots,
                    outlier_count=row.outlier_count,
                    outlier_rate=row.outlier_rate,
                    lower_bound=row.lower_bound,
                    upper_bound=row.upper_bound,
                    threshold=result.threshold,
                ),
            )
        )
    return _collection(max_plots, len(result.analyzed_columns), plots)


def plot_group_comparison(result: GroupComparison) -> PlotCollection:
    """Plot supplied group medians and IQR error bars without regrouping data."""
    _expect(result, GroupComparison, "result must be a GroupComparison")
    _schema(
        result.summary, _GROUP_COLUMNS, _GROUP_DTYPES, "group result has invalid schema"
    )
    if result.summary.duplicated(["value", "group"]).any() or not set(
        result.summary["value"]
    ).issubset(result.analyzed_values):
        raise ValueError("group result has invalid schema")
    counts = (result.available_group_count, result.displayed_group_count)
    if (
        any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in counts
        )
        or isinstance(result.max_groups, bool)
        or not isinstance(result.max_groups, int)
        or result.max_groups < 1
        or not isinstance(result.truncated, bool)
        or result.displayed_group_count > result.available_group_count
        or result.displayed_group_count
        != min(result.available_group_count, result.max_groups)
        or (
            not result.truncated
            and (
                result.displayed_group_count != result.available_group_count
                or result.truncation_reason is not None
            )
        )
        or (
            result.truncated
            and (
                result.displayed_group_count >= result.available_group_count
                or result.truncation_reason != "exceeds_max_groups"
            )
        )
    ):
        raise ValueError("group result has invalid schema")
    if any(
        (result.summary["value"] == value).sum() != result.displayed_group_count
        for value in result.analyzed_values
    ):
        raise ValueError("group result has invalid schema")
    plots: list[PlotResult] = []
    for value in result.analyzed_values[:20]:
        rows = result.summary[result.summary["value"] == value]
        positions = np.arange(len(rows))
        medians = rows["median"].to_numpy(float)
        lower = medians - rows["q25"].to_numpy(float)
        upper = rows["q75"].to_numpy(float) - medians
        title = f"{value} by {result.group_by}"
        figure, axes = _axes(title, result.group_by, f"{value} median")
        axes.bar(positions, medians, color=_BLUE)
        finite = np.isfinite(medians) & np.isfinite(lower) & np.isfinite(upper)
        if finite.any():
            axes.errorbar(
                positions[finite],
                medians[finite],
                yerr=np.vstack((lower[finite], upper[finite])),
                fmt="none",
                color="black",
            )
        axes.set_xticks(positions, [str(group) for group in rows["group"]])
        _bar_grid(axes)
        plots.append(
            PlotResult(
                figure,
                "group_median",
                title,
                "group_comparison",
                value,
                _meta(
                    value=value,
                    displayed_groups=_json(rows["group"].tolist()),
                    finite_medians=int(np.isfinite(medians).sum()),
                    metric="median",
                    error_bars="q25_q75",
                ),
            )
        )
    return _collection(20, len(result.analyzed_values), plots)


def _validate_classification_categorical_block(
    rows: pd.DataFrame, test_row: pd.Series
) -> None:
    """Validate one complete Task 08 categorical classification detail block."""
    message = "target result has invalid schema"
    categories = list(pd.unique(rows["feature_category"]))
    targets = list(pd.unique(rows["target_category"]))
    if (
        test_row["analysis"] != "chi_square"
        or test_row["group_count"] != len(categories)
        or not categories
        or not targets
        or len(rows) != len(categories) * len(targets)
        or not rows["target_mean"].isna().all()
        or not rows["target_median"].isna().all()
    ):
        raise ValueError(message)
    counts = rows["count"].to_numpy()
    if (
        any(
            isinstance(count, (bool, np.bool_))
            or not isinstance(count, (int, np.integer))
            or count < 0
            for count in counts
        )
        or int(counts.sum()) != test_row["n_obs"]
    ):
        raise ValueError(message)
    for category in categories:
        category_rows = rows[rows["feature_category"] == category]
        denominator = int(category_rows["count"].sum())
        if denominator == 0:
            raise ValueError(message)
        for row in category_rows.itertuples(index=False):
            if (
                not np.isfinite(row.rate)
                or row.rate < 0.0
                or row.rate > 1.0
                or row.rate != row.count / denominator
            ):
                raise ValueError(message)
    for target in targets:
        if int(rows.loc[rows["target_category"] == target, "count"].sum()) == 0:
            raise ValueError(message)


def plot_target_relationships(result: TargetAnalysis) -> PlotCollection:
    """Plot only the approved Task 08 detail-table target relationships."""
    _expect(result, TargetAnalysis, "result must be a TargetAnalysis")
    _schema(
        result.numeric_details,
        _NUMERIC_DETAIL_COLUMNS,
        _NUMERIC_DETAIL_DTYPES,
        "target result has invalid schema",
    )
    _schema(
        result.category_details,
        _CATEGORY_DETAIL_COLUMNS,
        _CATEGORY_DETAIL_DTYPES,
        "target result has invalid schema",
    )
    _schema(
        result.statistical_tests,
        _TEST_COLUMNS,
        _TEST_DTYPES,
        "target result has invalid schema",
    )
    features = set(result.analyzed_features)
    if (
        not set(result.numeric_details["feature"]).issubset(features)
        or not set(result.category_details["feature"]).issubset(features)
        or set(result.statistical_tests["feature"]) != features
        or result.numeric_details.duplicated(["feature", "target_category"]).any()
        or result.statistical_tests["feature"].duplicated().any()
    ):
        raise ValueError("target result has invalid schema")
    category_keys = (
        ["feature", "feature_category"]
        if result.task == "regression"
        else ["feature", "feature_category", "target_category"]
    )
    if result.category_details.duplicated(category_keys).any() or result.task not in (
        "classification",
        "regression",
    ):
        raise ValueError("target result has invalid schema")
    if result.task == "regression" and not result.numeric_details.empty:
        raise ValueError("target result has invalid schema")
    for feature in result.analyzed_features:
        feature_test = result.statistical_tests[
            result.statistical_tests["feature"] == feature
        ]
        if len(feature_test) != 1:
            raise ValueError("target result has invalid schema")
        test_row = feature_test.iloc[0]
        feature_kind = test_row["feature_kind"]
        numeric_rows = result.numeric_details[
            result.numeric_details["feature"] == feature
        ]
        category_rows = result.category_details[
            result.category_details["feature"] == feature
        ]
        if (
            not all(
                np.isfinite(test_row[field])
                for field in ("statistic", "p_value", "effect_size")
            )
            or not 0.0 <= test_row["p_value"] <= 1.0
            or test_row["effect_size"] < 0.0
        ):
            raise ValueError("target result has invalid schema")
        if feature_kind == "numeric":
            if result.task == "classification":
                if (
                    test_row["analysis"] != "kruskal_wallis"
                    or test_row["effect_size_name"] != "epsilon_squared"
                    or test_row["limitation"] != "exploratory_unadjusted_p_value"
                    or numeric_rows.empty
                    or not category_rows.empty
                    or test_row["group_count"] != len(numeric_rows)
                    or test_row["n_obs"] != int(numeric_rows["count"].sum())
                ):
                    raise ValueError("target result has invalid schema")
            elif (
                test_row["analysis"] != "pearson"
                or test_row["effect_size_name"] != "absolute_pearson_r"
                or test_row["limitation"] != "exploratory_unadjusted_p_value"
                or test_row["group_count"] != 0
                or not numeric_rows.empty
                or not category_rows.empty
            ):
                raise ValueError("target result has invalid schema")
        elif feature_kind == "categorical":
            if not numeric_rows.empty or category_rows.empty:
                raise ValueError("target result has invalid schema")
            if result.task == "classification":
                if test_row["effect_size_name"] != "cramers_v" or test_row[
                    "limitation"
                ] not in {
                    "exploratory_unadjusted_p_value",
                    "exploratory_unadjusted_p_value; "
                    "chi_square_expected_counts_may_be_small",
                }:
                    raise ValueError("target result has invalid schema")
                _validate_classification_categorical_block(category_rows, test_row)
            elif (
                test_row["analysis"] != "kruskal_wallis"
                or test_row["effect_size_name"] != "epsilon_squared"
                or test_row["limitation"] != "exploratory_unadjusted_p_value"
                or test_row["group_count"]
                != len(pd.unique(category_rows["feature_category"]))
                or test_row["n_obs"] != int(category_rows["count"].sum())
            ):
                raise ValueError("target result has invalid schema")
        else:
            raise ValueError("target result has invalid schema")
    candidates: list[tuple[str, str, pd.DataFrame]] = []
    if result.task == "classification":
        for feature in result.analyzed_features:
            rows = result.numeric_details[result.numeric_details["feature"] == feature]
            if not rows.empty:
                candidates.append(("numeric", feature, rows.copy()))
            rows = result.category_details[
                result.category_details["feature"] == feature
            ]
            if not rows.empty:
                candidates.append(("classification_category", feature, rows.copy()))
    else:
        for feature in result.analyzed_features:
            rows = result.category_details[
                result.category_details["feature"] == feature
            ]
            if not rows.empty:
                candidates.append(("regression_category", feature, rows.copy()))
    selected = candidates[:20]
    plots: list[PlotResult] = []
    for kind, feature, rows in selected:
        if kind == "numeric":
            pos = np.arange(len(rows))
            med = rows["median"].to_numpy(float)
            title = f"{feature} by {result.target}"
            fig, ax = _axes(title, result.target, f"{feature} median")
            ax.bar(pos, med, color=_BLUE)
            lower = med - rows["q25"].to_numpy(float)
            upper = rows["q75"].to_numpy(float) - med
            finite = np.isfinite(med) & np.isfinite(lower) & np.isfinite(upper)
            if finite.any():
                ax.errorbar(
                    pos[finite],
                    med[finite],
                    yerr=np.vstack((lower[finite], upper[finite])),
                    fmt="none",
                    color="black",
                )
            ax.set_xticks(pos, [str(x) for x in rows["target_category"]])
            _bar_grid(ax)
            plots.append(
                PlotResult(
                    fig,
                    "target_classification_numeric",
                    title,
                    "target_analysis",
                    feature,
                    _meta(
                        feature=feature,
                        analysis_type="classification_numeric",
                        target_categories=_json(rows["target_category"].tolist()),
                        metric="median",
                        error_bars="q25_q75",
                    ),
                )
            )
        elif kind == "classification_category":
            cats = list(pd.unique(rows["feature_category"]))
            targets = list(pd.unique(rows["target_category"]))
            counts = {
                (row.feature_category, row.target_category): int(row.count)
                for row in rows.itertuples(index=False)
            }
            pos = np.arange(len(cats))
            width = 0.8 / max(1, len(targets))
            title = f"{feature} by {result.target}"
            fig, ax = _axes(title, feature, "Count")
            for index, target in enumerate(targets):
                values = [counts[(cat, target)] for cat in cats]
                ax.bar(
                    pos - 0.4 + width / 2 + index * width,
                    values,
                    width,
                    label=str(target),
                    color=_PALETTE[index % len(_PALETTE)],
                )
            ax.set_xticks(pos, [str(cat) for cat in cats])
            ax.legend(title=result.target)
            _bar_grid(ax)
            plots.append(
                PlotResult(
                    fig,
                    "target_classification_categorical",
                    title,
                    "target_analysis",
                    feature,
                    _meta(
                        feature=feature,
                        analysis_type="classification_categorical",
                        feature_categories=_json(cats),
                        target_categories=_json(targets),
                        metric="count",
                    ),
                )
            )
        else:
            pos = np.arange(len(rows))
            title = f"{feature} by {result.target}"
            fig, ax = _axes(title, feature, f"{result.target} median")
            ax.bar(pos, rows["target_median"].to_numpy(float), color=_BLUE)
            ax.set_xticks(pos, [str(cat) for cat in rows["feature_category"]])
            _bar_grid(ax)
            plots.append(
                PlotResult(
                    fig,
                    "target_regression_categorical",
                    title,
                    "target_analysis",
                    feature,
                    _meta(
                        feature=feature,
                        analysis_type="regression_categorical",
                        feature_categories=_json(rows["feature_category"].tolist()),
                        target_categories=json.dumps([], separators=(",", ":")),
                        metric="target_median",
                    ),
                )
            )
    return _collection(20, len(candidates), plots)


def plot_classification_evaluation(
    result: ClassificationEvaluation,
) -> PlotCollection:
    """Plot frozen classification holdout detail without evaluating a model.

    Parameters
    ----------
    result
        A complete, internally consistent ``ClassificationEvaluation``.

    Returns
    -------
    PlotCollection
        A confusion-matrix Figure and, for binary score-capable evaluations, a
        ROC Figure. Figures are caller-owned and are neither displayed nor saved.

    Raises
    ------
    ValueError
        If ``result`` has the wrong type or is malformed.

    Notes
    -----
    This function only reads frozen evaluation fields; it does not fit, predict,
    recompute metrics, mutate its input, or change global plotting state.

    Examples
    --------
    >>> # result = evaluate_classifier(training_result)
    >>> # plots = plot_classification_evaluation(result)
    """
    if not isinstance(result, ClassificationEvaluation):
        raise ValueError("result must be a ClassificationEvaluation")
    _validate_classification_evaluation(result)
    classes = list(result.classes)
    positions = np.arange(len(classes))
    plots: list[PlotResult] = []

    confusion_title = "Classification confusion matrix"
    figure, axes = _axes(confusion_title, "Predicted label", "True label")
    matrix = np.asarray(result.confusion_matrix, dtype=int)
    axes.imshow(matrix, cmap="Blues")
    axes.set_xticks(positions, [str(label) for label in classes])
    axes.set_yticks(positions, [str(label) for label in classes])
    for row, column in np.ndindex(matrix.shape):
        axes.text(column, row, str(matrix[row, column]), ha="center", va="center")
    plots.append(
        PlotResult(
            figure,
            "classification_confusion_matrix",
            confusion_title,
            "classification_evaluation",
            None,
            (
                ("target", result.target),
                ("classes", _json(classes)),
                ("n_test", str(len(result.holdout_positions))),
                ("metric", "count"),
            ),
        )
    )
    if result.score_kind is not None:
        roc_title = f"ROC curve ({result.target})"
        figure, axes = _axes(roc_title, "False positive rate", "True positive rate")
        curve = np.asarray(result.roc_curve, dtype=float)
        axes.plot(curve[:, 0], curve[:, 1], label="ROC", color=_BLUE)
        axes.legend()
        plots.append(
            PlotResult(
                figure,
                "classification_roc_curve",
                roc_title,
                "classification_evaluation",
                None,
                (
                    ("target", result.target),
                    ("classes", _json(classes)),
                    ("positive_label", str(result.positive_label)),
                    ("n_test", str(len(result.holdout_positions))),
                    ("score_kind", result.score_kind),
                    ("roc_auc", format(float(result.roc_auc), ".12g")),
                    ("metric", "roc_auc"),
                ),
            )
        )
    return _classification_collection(plots)


def plot_regression_evaluation(result: RegressionEvaluation) -> PlotCollection:
    """Plot frozen regression holdout diagnostics without evaluating a model.

    Parameters
    ----------
    result
        A complete, internally consistent ``RegressionEvaluation``.

    Returns
    -------
    PlotCollection
        Caller-owned predicted-versus-actual and residual Figures. The function
        neither displays nor saves them.

    Raises
    ------
    ValueError
        If ``result`` has the wrong type or malformed frozen detail.

    Notes
    -----
    This function validates then only reads the prediction table. It never fits,
    predicts, recomputes metrics, mutates its input, or changes global plotting
    state.

    Examples
    --------
    >>> # evaluation = evaluate_regressor(training_result)
    >>> # plots = plot_regression_evaluation(evaluation)
    """
    if not isinstance(result, RegressionEvaluation):
        raise ValueError("result must be a RegressionEvaluation")
    _validate_regression_evaluation(result)
    detail = result.predictions
    actual = detail["actual"].to_numpy(dtype=float)
    predicted = detail["predicted"].to_numpy(dtype=float)
    residuals = detail["residual"].to_numpy(dtype=float)
    n_test = str(len(result.holdout_positions))

    predicted_title = f"Predicted vs actual ({result.target})"
    figure, axes = _axes(predicted_title, "Actual value", "Predicted value")
    axes.scatter(actual, predicted, color=_BLUE)
    lower = float(min(np.min(actual), np.min(predicted)))
    upper = float(max(np.max(actual), np.max(predicted)))
    axes.plot((lower, upper), (lower, upper), label="Ideal", color="black")
    axes.legend()

    residual_title = f"Residuals ({result.target})"
    residual_figure, residual_axes = _axes(
        residual_title, "Predicted value", "Residual (actual - predicted)"
    )
    residual_axes.scatter(predicted, residuals, color=_BLUE)
    residual_axes.axhline(0.0, label="Zero residual", color="black")
    residual_axes.legend()

    plots = (
        PlotResult(
            figure,
            "regression_predicted_vs_actual",
            predicted_title,
            "regression_evaluation",
            None,
            (("target", result.target), ("n_test", n_test), ("metric", "prediction")),
        ),
        PlotResult(
            residual_figure,
            "regression_residuals",
            residual_title,
            "regression_evaluation",
            None,
            (("target", result.target), ("n_test", n_test), ("metric", "residual")),
        ),
    )
    return PlotCollection(
        requested_count=2,
        available_count=2,
        actual_count=2,
        truncated=False,
        truncation_reason=None,
        plots=plots,
    )
