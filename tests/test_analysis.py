"""Contract tests for Task 07 non-target feature analysis."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from sharper import (
    analyze_categorical_features,
    analyze_numeric_features,
    compute_correlations,
    detect_outliers,
)

ANALYSIS_FUNCTIONS = (
    analyze_numeric_features,
    analyze_categorical_features,
    compute_correlations,
    detect_outliers,
)

NUMERIC_DTYPES = {
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

CATEGORICAL_SUMMARY_DTYPES = {
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

TOP_CATEGORIES_DTYPES = {
    "column": "object",
    "category": "object",
    "count": "int64",
    "rate": "float64",
    "rank": "int64",
}

CORRELATION_DTYPES = {
    "column_a": "object",
    "column_b": "object",
    "method": "object",
    "correlation": "float64",
    "n_pairs": "int64",
}

OUTLIER_SUMMARY_DTYPES = {
    "column": "object",
    "method": "object",
    "threshold": "float64",
    "lower_bound": "float64",
    "upper_bound": "float64",
    "outlier_count": "int64",
    "outlier_rate": "float64",
}

OUTLIER_DETAILS_DTYPES = {
    "column": "object",
    "row_index": "object",
    "value": "float64",
    "lower_bound": "float64",
    "upper_bound": "float64",
}


def _assert_schema(frame: pd.DataFrame, expected: dict[str, str]) -> None:
    assert frame.columns.tolist() == list(expected)
    assert {name: str(dtype) for name, dtype in frame.dtypes.items()} == expected


@pytest.mark.parametrize("function", ANALYSIS_FUNCTIONS)
def test_shared_validation_rejects_non_dataframe(function: object) -> None:
    """Every entry point rejects values outside the pandas boundary."""
    with pytest.raises(ValueError, match="df must be a pandas DataFrame"):
        function({"x": [1]})


@pytest.mark.parametrize("function", ANALYSIS_FUNCTIONS)
def test_shared_validation_rejects_non_string_dataframe_columns(
    function: object,
) -> None:
    """Every entry point preserves the string-name boundary."""
    with pytest.raises(ValueError, match="DataFrame column names must all be strings"):
        function(pd.DataFrame([[1]], columns=[1]))


@pytest.mark.parametrize("function", ANALYSIS_FUNCTIONS)
def test_shared_validation_rejects_duplicate_dataframe_columns(
    function: object,
) -> None:
    """Ambiguous duplicate DataFrame names use the frozen error."""
    frame = pd.DataFrame([[1, 2]], columns=["x", "x"])
    with pytest.raises(
        ValueError, match="duplicate DataFrame column names are not supported"
    ):
        function(frame)


@pytest.mark.parametrize("function", ANALYSIS_FUNCTIONS)
@pytest.mark.parametrize(
    ("columns", "message"),
    [
        (["missing"], "column not found"),
        (["x", "x"], "duplicate column parameter"),
        (["x", 1], "columns must contain only strings"),
    ],
)
def test_shared_validation_rejects_invalid_requested_columns(
    function: object,
    columns: list[object],
    message: str,
) -> None:
    """Requested column names must be strings, unique, and present."""
    with pytest.raises(ValueError, match=message):
        function(pd.DataFrame({"x": [1, 2]}), columns=columns)


@pytest.mark.parametrize("function", ANALYSIS_FUNCTIONS)
def test_analysis_does_not_mutate_input(function: object) -> None:
    """All four analyses leave values, dtypes, index, and names unchanged."""
    frame = pd.DataFrame(
        {"number": [0.0, np.nan, 10.0], "label": ["b", "a", "b"]},
        index=pd.Index(["r2", "r1", "r3"], name="row"),
    )
    before = frame.copy(deep=True)

    function(frame)

    pdt.assert_frame_equal(frame, before)


def test_result_dataclass_instances_are_frozen() -> None:
    """The public result containers reject field assignment."""
    result = analyze_numeric_features(pd.DataFrame({"x": [1, 2]}))
    with pytest.raises(FrozenInstanceError):
        result.n_rows = 3


def test_numeric_auto_selection_statistics_and_inf_behavior() -> None:
    """Numeric analysis selects non-bool dtypes and uses pandas defaults."""
    frame = pd.DataFrame(
        {
            "integer": [0, 2, 4, 6],
            "nullable": pd.Series([1, None, 3, 5], dtype="Int64"),
            "flag": [True, False, True, False],
            "label": ["a", "b", "c", "d"],
            "infinite": [1.0, np.inf, 3.0, 0.0],
        }
    )

    result = analyze_numeric_features(frame)

    assert result.requested_columns is None
    assert result.analyzed_columns == ("integer", "nullable", "infinite")
    assert result.skipped_columns == ()
    integer = result.summary.iloc[0]
    assert integer["count"] == 4
    assert integer["missing_count"] == 0
    assert integer["mean"] == pytest.approx(frame["integer"].mean())
    assert integer["std"] == pytest.approx(frame["integer"].std())
    assert integer["q25"] == pytest.approx(frame["integer"].quantile(0.25))
    assert integer["skew"] == pytest.approx(frame["integer"].skew())
    assert integer["zero_count"] == 1
    assert integer["zero_rate"] == pytest.approx(0.25)
    infinite = result.summary.iloc[2]
    assert np.isinf(infinite["max"])


def test_numeric_explicit_order_skips_and_fixed_schema() -> None:
    """Explicit order is retained and skip precedence follows the contract."""
    frame = pd.DataFrame(
        {
            "all_missing_numeric": pd.Series([None, None], dtype="Float64"),
            "text": [None, None],
            "second": [1.0, 2.0],
            "first": [0.0, np.nan],
        }
    )

    result = analyze_numeric_features(
        frame,
        columns=["second", "text", "all_missing_numeric", "first"],
    )

    assert result.requested_columns == (
        "second",
        "text",
        "all_missing_numeric",
        "first",
    )
    assert result.analyzed_columns == ("second", "first")
    assert result.skipped_columns == ("text", "all_missing_numeric")
    assert result.skipped_reasons == {
        "text": "not_numeric",
        "all_missing_numeric": "all_missing",
    }
    assert result.summary["column"].tolist() == ["second", "first"]
    assert result.summary.loc[1, "zero_rate"] == pytest.approx(1.0)
    _assert_schema(result.summary, NUMERIC_DTYPES)


def test_numeric_empty_result_has_fixed_schema() -> None:
    """No analyzable numeric columns still yields typed empty output."""
    result = analyze_numeric_features(pd.DataFrame({"x": pd.Series(dtype="float64")}))

    assert result.analyzed_columns == ()
    assert result.skipped_reasons == {"x": "all_missing"}
    assert result.summary.empty
    _assert_schema(result.summary, NUMERIC_DTYPES)


def test_categorical_auto_selection_and_fixed_schemas() -> None:
    """Object, string, category, and bool dtypes are selected in frame order."""
    frame = pd.DataFrame(
        {
            "object": pd.Series(["x", "y", "x"], dtype="object"),
            "string": pd.Series(["a", "a", None], dtype="string"),
            "category": pd.Series(["u", "v", "u"], dtype="category"),
            "boolean": pd.Series([True, False, True], dtype="boolean"),
            "number": [1, 2, 3],
        }
    )

    result = analyze_categorical_features(frame)

    assert result.requested_columns is None
    assert result.analyzed_columns == ("object", "string", "category", "boolean")
    assert result.skipped_columns == ()
    _assert_schema(result.summary, CATEGORICAL_SUMMARY_DTYPES)
    _assert_schema(result.top_categories, TOP_CATEGORIES_DTYPES)


def test_categorical_explicit_order_budget_ties_and_missing() -> None:
    """Category ties follow first appearance and top_n is per-column."""
    frame = pd.DataFrame(
        {
            "numeric": [1, 2, 3, 4, 5],
            "all_missing": pd.Series([None] * 5, dtype="string"),
            "label": ["b", "a", "c", "a", "b"],
        }
    )

    result = analyze_categorical_features(
        frame,
        columns=["label", "numeric", "all_missing"],
        top_n=2,
    )

    assert result.requested_columns == ("label", "numeric", "all_missing")
    assert result.analyzed_columns == ("label",)
    assert result.skipped_columns == ("numeric", "all_missing")
    assert result.skipped_reasons == {
        "numeric": "not_categorical",
        "all_missing": "all_missing",
    }
    assert result.top_n == 2
    assert result.summary.loc[0, "top"] == "b"
    assert result.summary.loc[0, "top_count"] == 2
    assert result.top_categories["category"].tolist() == ["b", "a"]
    assert result.top_categories["rank"].tolist() == [1, 2]
    assert result.top_categories["rate"].tolist() == pytest.approx([0.4, 0.4])


def test_categorical_mixed_hashable_values_preserve_first_appearance_ties() -> None:
    """Tuple categories remain scalar values during frequency lookup."""
    frame = pd.DataFrame({"mixed": [1, "a", (2, 3), "a", (2, 3), 1, "single"]})

    result = analyze_categorical_features(frame, top_n=3)

    assert result.analyzed_columns == ("mixed",)
    assert result.summary.loc[0, "unique_count"] == 4
    assert result.summary.loc[0, "top"] == 1
    assert result.summary.loc[0, "top_count"] == 2
    assert result.top_categories["category"].tolist() == [1, "a", (2, 3)]
    assert result.top_categories["count"].tolist() == [2, 2, 2]
    assert result.top_categories["rate"].tolist() == pytest.approx([2 / 7] * 3)
    assert result.top_categories["rank"].tolist() == [1, 2, 3]


@pytest.mark.parametrize("top_n", [0, -1, 1.5, True, "2"])
def test_categorical_rejects_invalid_top_n(top_n: object) -> None:
    """The display budget must be a positive, non-boolean integer."""
    with pytest.raises(ValueError, match="top_n must be a positive integer"):
        analyze_categorical_features(pd.DataFrame({"x": ["a"]}), top_n=top_n)


def test_categorical_empty_result_has_two_fixed_schemas() -> None:
    """Empty categorical output retains both frozen table schemas."""
    result = analyze_categorical_features(pd.DataFrame({"number": [1, 2]}))

    assert result.analyzed_columns == ()
    assert result.skipped_columns == ()
    _assert_schema(result.summary, CATEGORICAL_SUMMARY_DTYPES)
    _assert_schema(result.top_categories, TOP_CATEGORIES_DTYPES)


@pytest.mark.parametrize("method", ["kendall", "PEARSON", ""])
def test_correlation_rejects_invalid_method(method: str) -> None:
    """Task 07 exposes only Pearson and Spearman."""
    with pytest.raises(ValueError, match="method must be pearson or spearman"):
        compute_correlations(pd.DataFrame({"x": [1, 2]}), method=method)


@pytest.mark.parametrize("max_columns", [1, 1.5, True, "2"])
def test_correlation_rejects_invalid_max_columns(max_columns: object) -> None:
    """The correlation column budget is an integer of at least two."""
    with pytest.raises(ValueError, match="max_columns must be an integer >= 2"):
        compute_correlations(pd.DataFrame({"x": [1, 2]}), max_columns=max_columns)


@pytest.mark.parametrize("min_periods", [1, 1.5, True, "2"])
def test_correlation_rejects_invalid_min_periods(min_periods: object) -> None:
    """The effective-sample requirement is an integer of at least two."""
    with pytest.raises(ValueError, match="min_periods must be an integer >= 2"):
        compute_correlations(pd.DataFrame({"x": [1, 2]}), min_periods=min_periods)


def test_correlation_skip_precedence_and_budget_after_eligibility() -> None:
    """Earlier skip reasons win and only eligible columns consume the budget."""
    frame = pd.DataFrame(
        {
            "text": [None, None, None],
            "all_missing": pd.Series([None, None, None], dtype="Float64"),
            "insufficient": [1.0, np.nan, np.nan],
            "constant": [2.0, 2.0, np.nan],
            "kept_a": [1.0, 2.0, 3.0],
            "kept_b": [3.0, 2.0, 1.0],
            "over_budget": [1.0, 4.0, 9.0],
        }
    )

    result = compute_correlations(
        frame,
        columns=list(frame.columns),
        max_columns=2,
        min_periods=2,
    )

    assert result.analyzed_columns == ("kept_a", "kept_b")
    assert result.skipped_columns == (
        "text",
        "all_missing",
        "insufficient",
        "constant",
        "over_budget",
    )
    assert result.skipped_reasons == {
        "text": "not_numeric",
        "all_missing": "all_missing",
        "insufficient": "insufficient_non_missing",
        "constant": "constant",
        "over_budget": "exceeds_max_columns",
    }
    assert result.truncated is True


def test_correlation_pair_order_counts_min_periods_and_nan_omission() -> None:
    """Long-form pairs follow i<j order and expose pairwise sample sizes."""
    frame = pd.DataFrame(
        {
            "c": [1.0, 2.0, np.nan, 4.0],
            "a": [2.0, 4.0, 6.0, np.nan],
            "b": [4.0, np.nan, 2.0, 1.0],
        }
    )

    result = compute_correlations(
        frame, columns=["c", "a", "b"], method="spearman", min_periods=2
    )

    assert result.requested_columns == ("c", "a", "b")
    assert result.analyzed_columns == ("c", "a", "b")
    assert list(zip(result.correlations.column_a, result.correlations.column_b)) == [
        ("c", "a"),
        ("c", "b"),
        ("a", "b"),
    ]
    assert result.correlations["n_pairs"].tolist() == [2, 2, 2]
    expected = [
        frame["c"].corr(frame["a"], method="spearman"),
        frame["c"].corr(frame["b"], method="spearman"),
        frame["a"].corr(frame["b"], method="spearman"),
    ]
    assert result.correlations["correlation"].tolist() == pytest.approx(expected)
    assert not any(result.correlations.column_a == result.correlations.column_b)
    _assert_schema(result.correlations, CORRELATION_DTYPES)

    too_sparse = compute_correlations(frame, min_periods=3)
    assert too_sparse.correlations.empty
    _assert_schema(too_sparse.correlations, CORRELATION_DTYPES)

    infinite = compute_correlations(pd.DataFrame({"x": [1.0, np.inf], "y": [2.0, 3.0]}))
    assert infinite.analyzed_columns == ("x", "y")
    assert infinite.correlations.empty


def test_correlation_auto_selection_excludes_bool_and_non_numeric() -> None:
    """Auto-selection considers only numeric non-boolean columns."""
    result = compute_correlations(
        pd.DataFrame(
            {
                "a": [1, 2, 3],
                "flag": [True, False, True],
                "label": ["x", "y", "z"],
                "b": [2, 4, 8],
            }
        )
    )

    assert result.analyzed_columns == ("a", "b")
    assert result.skipped_columns == ()
    assert result.correlations[["column_a", "column_b"]].values.tolist() == [["a", "b"]]


@pytest.mark.parametrize("method", ["zscore", "mad", "IQR"])
def test_outlier_rejects_non_iqr_method(method: str) -> None:
    """Task 07 outlier detection exposes only IQR."""
    with pytest.raises(ValueError, match="method must be iqr"):
        detect_outliers(pd.DataFrame({"x": [1, 2]}), method=method)


@pytest.mark.parametrize(
    "threshold", [0, -1, True, np.bool_(False), np.nan, 1 + 2j, "1.5"]
)
def test_outlier_rejects_invalid_threshold(threshold: object) -> None:
    """The IQR multiplier must be a positive, non-boolean real number."""
    with pytest.raises(ValueError, match="threshold must be a positive number"):
        detect_outliers(pd.DataFrame({"x": [1, 2]}), threshold=threshold)


def test_outlier_skip_precedence() -> None:
    """Outlier skip reasons use the frozen single-reason precedence."""
    frame = pd.DataFrame(
        {
            "text": [None, None],
            "all_missing": pd.Series([None, None], dtype="Float64"),
            "non_finite": [np.inf, np.nan],
            "insufficient": [1.0, np.nan],
            "constant": [2.0, 2.0],
            "valid": [0.0, 1.0],
        }
    )

    result = detect_outliers(frame, columns=list(frame.columns))

    assert result.analyzed_columns == ("valid",)
    assert result.skipped_columns == (
        "text",
        "all_missing",
        "non_finite",
        "insufficient",
        "constant",
    )
    assert result.skipped_reasons == {
        "text": "not_numeric",
        "all_missing": "all_missing",
        "non_finite": "non_finite_values",
        "insufficient": "insufficient_non_missing",
        "constant": "constant",
    }


def test_outlier_bounds_original_labels_and_deterministic_row_order() -> None:
    """IQR details preserve requested column and original DataFrame row order."""
    frame = pd.DataFrame(
        {
            "second": [0.0, 0.0, 0.0, 9.0, -9.0],
            "first": [100.0, 1.0, 1.0, 1.0, -100.0],
        },
        index=["r3", "r1", "r4", "r2", "r0"],
    )

    result = detect_outliers(frame, columns=["first", "second"], threshold=1)

    assert result.requested_columns == ("first", "second")
    assert result.analyzed_columns == ("first", "second")
    assert result.summary["column"].tolist() == ["first", "second"]
    assert result.summary["threshold"].tolist() == [1.0, 1.0]
    assert result.outliers[["column", "row_index"]].values.tolist() == [
        ["first", "r3"],
        ["first", "r0"],
        ["second", "r2"],
        ["second", "r0"],
    ]
    for position, column in enumerate(["first", "second"]):
        values = frame[column]
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        expected_iqr = q3 - q1
        assert result.summary.loc[position, "lower_bound"] == pytest.approx(
            q1 - expected_iqr
        )
        assert result.summary.loc[position, "upper_bound"] == pytest.approx(
            q3 + expected_iqr
        )
        assert result.summary.loc[position, "outlier_rate"] == pytest.approx(0.4)
    _assert_schema(result.summary, OUTLIER_SUMMARY_DTYPES)
    _assert_schema(result.outliers, OUTLIER_DETAILS_DTYPES)


def test_outlier_iqr_zero_nonconstant_and_empty_details_schema() -> None:
    """A zero-IQR nonconstant column still applies its equal bounds."""
    zero_iqr = detect_outliers(pd.DataFrame({"x": [0.0, 0.0, 0.0, 0.0, 1.0]}))

    assert zero_iqr.summary.loc[0, "lower_bound"] == 0.0
    assert zero_iqr.summary.loc[0, "upper_bound"] == 0.0
    assert zero_iqr.outliers["value"].tolist() == [1.0]

    no_outliers = detect_outliers(pd.DataFrame({"x": [0.0, 1.0, 2.0]}))
    assert no_outliers.outliers.empty
    _assert_schema(no_outliers.outliers, OUTLIER_DETAILS_DTYPES)
