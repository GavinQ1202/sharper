"""Contract tests for Task 07 and Task 08 analysis APIs."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import sharper.analysis as analysis_module
from sharper import (
    GroupComparison,
    TargetAnalysis,
    analyze_categorical_features,
    analyze_numeric_features,
    analyze_target_relationships,
    compare_groups,
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

GROUP_SUMMARY_DTYPES = {
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

TARGET_NUMERIC_DETAILS_DTYPES = {
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

TARGET_CATEGORY_DETAILS_DTYPES = {
    "feature": "object",
    "feature_category": "object",
    "target_category": "object",
    "count": "int64",
    "rate": "float64",
    "target_mean": "float64",
    "target_median": "float64",
}

TARGET_TEST_DTYPES = {
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


def test_compare_groups_budget_missing_groups_and_deterministic_summary() -> None:
    """Groups use frequency/first-appearance ranking and disclose truncation."""
    frame = pd.DataFrame(
        {
            "group": ["b", "a", "b", "a", "c", None],
            "value": [1.0, 2.0, 3.0, np.nan, 9.0, 100.0],
        }
    )
    original = frame.copy(deep=True)

    result = compare_groups(frame, "group", max_groups=2)

    assert isinstance(result, GroupComparison)
    assert result.requested_values is None
    assert result.analyzed_values == ("value",)
    assert result.available_group_count == 3
    assert result.displayed_group_count == 2
    assert result.missing_group_count == 1
    assert result.truncated is True
    assert result.truncation_reason == "exceeds_max_groups"
    assert result.summary[["value", "group"]].values.tolist() == [
        ["value", "b"],
        ["value", "a"],
    ]
    assert result.summary["group_count"].tolist() == [2, 2]
    assert result.summary["count"].tolist() == [2, 1]
    assert result.summary["missing_count"].tolist() == [0, 1]
    assert result.summary["mean"].tolist() == [2.0, 2.0]
    _assert_schema(result.summary, GROUP_SUMMARY_DTYPES)
    pdt.assert_frame_equal(frame, original)


def test_compare_groups_keeps_all_missing_group_row_for_analyzed_value() -> None:
    """A displayed group with no value observations has one NaN summary row."""
    frame = pd.DataFrame(
        {"group": ["a", "a", "b", "b"], "value": [1.0, 3.0, np.nan, np.nan]}
    )

    result = compare_groups(frame, "group")

    missing_row = result.summary.loc[result.summary["group"].eq("b")].iloc[0]
    assert missing_row["group_count"] == 2
    assert missing_row["count"] == 0
    assert missing_row["missing_count"] == 2
    assert pd.isna(missing_row["mean"])
    assert pd.isna(missing_row["q25"])
    assert pd.isna(missing_row["median"])
    assert pd.isna(missing_row["q75"])


@pytest.mark.parametrize("max_groups", [True, 0, -1, 1.5])
def test_compare_groups_rejects_invalid_budget(max_groups: object) -> None:
    """The group budget is a positive non-boolean integer."""
    with pytest.raises(ValueError, match="max_groups must be an integer >= 1"):
        compare_groups(
            pd.DataFrame({"group": ["a"], "x": [1.0]}),
            "group",
            max_groups=max_groups,  # type: ignore[arg-type]
        )


def test_compare_groups_validation_and_explicit_non_numeric_error() -> None:
    """Group/value parameter errors remain function-level ValueErrors."""
    frame = pd.DataFrame({"group": ["a", "b"], "text": ["x", "y"]})
    with pytest.raises(ValueError, match="values must contain only numeric columns"):
        compare_groups(frame, "group", values=["text"])
    with pytest.raises(ValueError, match="group_by must not appear in values"):
        compare_groups(frame, "group", values=["group"])
    with pytest.raises(ValueError, match="group_by must be categorical"):
        compare_groups(pd.DataFrame({"group": [1, 2], "x": [1, 2]}), "group")
    with pytest.raises(
        ValueError, match="group_by must contain at least one non-missing value"
    ):
        compare_groups(pd.DataFrame({"group": [None], "x": [1.0]}), "group")


def test_compare_groups_bool_group_is_categorical() -> None:
    """Boolean group keys are valid and retain frequency/appearance ordering."""
    frame = pd.DataFrame(
        {"group": [True, False, True, False], "value": [1.0, 2.0, 3.0, 4.0]}
    )

    result = compare_groups(frame, "group")

    assert result.summary["group"].tolist() == [True, False]
    assert result.summary["group_count"].tolist() == [2, 2]
    assert result.summary["count"].tolist() == [2, 2]
    assert result.summary["mean"].tolist() == [2.0, 3.0]


def test_compare_groups_complex_values_follow_task08_real_numeric_contract() -> None:
    """Complex values are excluded automatically and rejected explicitly."""
    frame = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "real": [1.0, 2.0, 3.0, 4.0],
            "complex": np.array([1 + 1j, 2 + 1j, 3 + 1j, 4 + 1j]),
        }
    )

    automatic = compare_groups(frame, "group")

    assert automatic.analyzed_values == ("real",)
    assert "complex" not in automatic.skipped_reasons
    with pytest.raises(ValueError, match="values must contain only numeric columns"):
        compare_groups(frame, "group", values=["complex"])


def test_compare_groups_value_skip_precedence_and_empty_schema() -> None:
    """Value skips use the frozen vocabulary, precedence, and ordering."""
    frame = pd.DataFrame(
        {
            "group": ["a", "b"],
            "all_missing": pd.Series([None, None], dtype="Float64"),
            "non_finite": [1.0, np.inf],
            "insufficient": [1.0, np.nan],
            "constant": [2.0, 2.0],
        }
    )

    result = compare_groups(frame, "group")

    assert result.analyzed_values == ()
    assert result.skipped_values == (
        "all_missing",
        "non_finite",
        "insufficient",
        "constant",
    )
    assert result.skipped_reasons == {
        "all_missing": "all_missing",
        "non_finite": "non_finite_values",
        "insufficient": "insufficient_non_missing",
        "constant": "constant",
    }
    _assert_schema(result.summary, GROUP_SUMMARY_DTYPES)


def test_task08_shared_dataframe_validation() -> None:
    """Both Task 08 entry points enforce the shared pandas/name boundary."""
    with pytest.raises(ValueError, match="df must be a pandas DataFrame"):
        compare_groups({"g": ["a"]}, "g")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="df must be a pandas DataFrame"):
        analyze_target_relationships(
            {"target": [1]},  # type: ignore[arg-type]
            "target",
            task="regression",
        )
    non_string = pd.DataFrame([["a", 1]], columns=["g", 1])
    with pytest.raises(ValueError, match="DataFrame column names must all be strings"):
        compare_groups(non_string, "g")
    duplicate = pd.DataFrame([["a", "b"]], columns=["target", "target"])
    with pytest.raises(
        ValueError, match="duplicate DataFrame column names are not supported"
    ):
        analyze_target_relationships(duplicate, "target", task="classification")


def test_classification_numeric_retains_only_groups_of_size_two() -> None:
    """Kruskal details and effective samples exclude undersized target groups."""
    frame = pd.DataFrame(
        {
            "target": ["small", "a", "a", "b", "b"],
            "numeric": [100.0, 1.0, 2.0, 4.0, 8.0],
        }
    )
    result = analyze_target_relationships(frame, "target", task="classification")
    expected = analysis_module.stats.kruskal([1.0, 2.0], [4.0, 8.0])
    test_row = result.statistical_tests.iloc[0]

    assert isinstance(result, TargetAnalysis)
    assert result.analyzed_features == ("numeric",)
    assert result.numeric_details["target_category"].tolist() == ["a", "b"]
    assert test_row["n_obs"] == 4
    assert test_row["group_count"] == 2
    assert test_row["statistic"] == pytest.approx(expected.statistic)
    assert test_row["p_value"] == pytest.approx(expected.pvalue)
    assert test_row["effect_size"] == pytest.approx(
        max(0.0, (expected.statistic - 2 + 1) / (4 - 2))
    )
    assert test_row["effect_size_name"] == "epsilon_squared"
    _assert_schema(result.numeric_details, TARGET_NUMERIC_DETAILS_DTYPES)
    _assert_schema(result.category_details, TARGET_CATEGORY_DETAILS_DTYPES)
    _assert_schema(result.statistical_tests, TARGET_TEST_DTYPES)


def test_classification_numeric_skips_when_retained_groups_are_insufficient() -> None:
    """A feature with fewer than two retained target groups is skipped."""
    frame = pd.DataFrame(
        {"target": ["a", "a", "b", "c"], "numeric": [1.0, 2.0, 3.0, 4.0]}
    )

    result = analyze_target_relationships(frame, "target", task="classification")

    assert result.skipped_reasons == {"numeric": "insufficient_groups"}
    assert result.limitations == ()
    _assert_schema(result.statistical_tests, TARGET_TEST_DTYPES)


def test_classification_categorical_cartesian_zero_cells_and_limitations() -> None:
    """Chi-square details retain zero cells and use closed limitation codes."""
    frame = pd.DataFrame(
        {
            "target": ["a", "a", "a", "b"],
            "category": ["u", "u", "v", "v"],
        }
    )
    result = analyze_target_relationships(frame, "target", task="classification")
    table = np.array([[2, 0], [1, 1]])
    expected = analysis_module.stats.chi2_contingency(table)

    assert result.category_details[
        ["feature_category", "target_category"]
    ].values.tolist() == [
        ["u", "a"],
        ["u", "b"],
        ["v", "a"],
        ["v", "b"],
    ]
    zero = result.category_details.loc[
        result.category_details["feature_category"].eq("u")
        & result.category_details["target_category"].eq("b")
    ].iloc[0]
    assert zero["count"] == 0
    assert zero["rate"] == 0.0
    assert pd.isna(zero["target_mean"])
    assert pd.isna(zero["target_median"])
    test_row = result.statistical_tests.iloc[0]
    assert test_row["statistic"] == pytest.approx(expected.statistic)
    assert test_row["p_value"] == pytest.approx(expected.pvalue)
    assert test_row["effect_size"] == pytest.approx(
        np.sqrt(expected.statistic / (4 * 1))
    )
    assert test_row["limitation"] == (
        "exploratory_unadjusted_p_value; chi_square_expected_counts_may_be_small"
    )
    assert result.limitations == (
        "exploratory_unadjusted_p_values",
        "chi_square_expected_counts_may_be_small",
    )
    _assert_schema(result.category_details, TARGET_CATEGORY_DETAILS_DTYPES)


def test_classification_categorical_mixed_hashable_category_order() -> None:
    """Mixed hashable categories retain first appearance without type sorting."""
    mixed = ["text", "text", 7, 7, ("tuple",), ("tuple",)]
    frame = pd.DataFrame(
        {"target": ["a", "b"] * 3, "category": pd.Series(mixed, dtype=object)}
    )

    result = analyze_target_relationships(frame, "target", task="classification")

    assert result.category_details["feature_category"].tolist() == [
        "text",
        "text",
        7,
        7,
        ("tuple",),
        ("tuple",),
    ]
    assert result.category_details["target_category"].tolist() == [
        "a",
        "b",
        "a",
        "b",
        "a",
        "b",
    ]
    assert result.category_details["count"].tolist() == [1, 1, 1, 1, 1, 1]


def test_classification_non_2x2_chi_square_uses_scipy_default() -> None:
    """A 2x3 table matches SciPy default without 2x2-only correction effects."""
    frame = pd.DataFrame(
        {
            "target": ["a", "a", "b", "b", "c", "c"],
            "category": ["u", "u", "u", "v", "v", "v"],
        }
    )
    table = np.array([[2, 1, 0], [0, 1, 2]])
    expected = analysis_module.stats.chi2_contingency(table)
    uncorrected = analysis_module.stats.chi2_contingency(table, correction=False)

    result = analyze_target_relationships(frame, "target", task="classification")
    test_row = result.statistical_tests.iloc[0]

    assert test_row["statistic"] == pytest.approx(expected.statistic)
    assert test_row["p_value"] == pytest.approx(expected.pvalue)
    assert expected.statistic == pytest.approx(uncorrected.statistic)
    assert expected.pvalue == pytest.approx(uncorrected.pvalue)


def test_regression_numeric_pearson_and_empty_numeric_details() -> None:
    """Regression numeric analysis stores Pearson r and absolute effect size."""
    frame = pd.DataFrame(
        {"target": [1.0, 2.0, 3.0, 4.0], "numeric": [4.0, 3.0, 2.0, 1.0]}
    )
    result = analyze_target_relationships(frame, "target", task="regression")
    test_row = result.statistical_tests.iloc[0]

    assert test_row["analysis"] == "pearson"
    assert test_row["statistic"] == pytest.approx(-1.0)
    assert test_row["effect_size"] == pytest.approx(1.0)
    assert test_row["effect_size_name"] == "absolute_pearson_r"
    assert result.numeric_details.empty
    assert result.limitations == ("exploratory_unadjusted_p_values",)
    _assert_schema(result.numeric_details, TARGET_NUMERIC_DETAILS_DTYPES)


def test_regression_categorical_retained_groups_and_rates() -> None:
    """Regression category details and Kruskal metadata use retained groups only."""
    frame = pd.DataFrame(
        {
            "target": [100.0, 1.0, 2.0, 4.0, 8.0],
            "category": ["small", "a", "a", "b", "b"],
        }
    )
    result = analyze_target_relationships(frame, "target", task="regression")

    assert result.category_details["feature_category"].tolist() == ["a", "b"]
    assert result.category_details["count"].tolist() == [2, 2]
    assert result.category_details["rate"].tolist() == [0.5, 0.5]
    assert result.category_details["target_mean"].tolist() == [1.5, 6.0]
    test_row = result.statistical_tests.iloc[0]
    assert test_row["n_obs"] == 4
    assert test_row["group_count"] == 2
    assert test_row["analysis"] == "kruskal_wallis"


def test_classification_target_validation_and_feature_parameter_errors() -> None:
    """Target dtype/value and feature parameter errors use stable fragments."""
    datetime_frame = pd.DataFrame(
        {"target": pd.date_range("2024-01-01", periods=4), "x": [1, 2, 3, 4]}
    )
    with pytest.raises(
        ValueError,
        match="classification target must be categorical or low-cardinality numeric",
    ):
        analyze_target_relationships(datetime_frame, "target", task="classification")
    timedelta_frame = pd.DataFrame(
        {"target": pd.to_timedelta([1, 2, 3, 4], unit="D"), "x": [1, 2, 3, 4]}
    )
    with pytest.raises(
        ValueError,
        match="classification target must be categorical or low-cardinality numeric",
    ):
        analyze_target_relationships(timedelta_frame, "target", task="classification")
    with pytest.raises(
        ValueError, match="classification target must contain only finite values"
    ):
        analyze_target_relationships(
            pd.DataFrame({"target": [0.0, np.inf], "x": [1, 2]}),
            "target",
            task="classification",
        )
    with pytest.raises(
        ValueError,
        match="classification target must be categorical or low-cardinality numeric",
    ):
        analyze_target_relationships(
            pd.DataFrame(
                {
                    "target": np.array([1 + 1j, 1 + 2j, 2 + 1j, 2 + 2j]),
                    "x": [1.0, 2.0, 3.0, 4.0],
                }
            ),
            "target",
            task="classification",
        )
    frame = pd.DataFrame({"target": ["a", "a", "b", "b"], "x": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="target must not appear in features"):
        analyze_target_relationships(
            frame, "target", task="classification", features=["target"]
        )
    with pytest.raises(ValueError, match="duplicate feature column parameter"):
        analyze_target_relationships(
            frame, "target", task="classification", features=["x", "x"]
        )


@pytest.mark.parametrize(
    "task",
    [[], {}, set(), (), None, 1, True, "invalid"],
    ids=["list", "dict", "set", "tuple", "none", "integer", "bool", "string"],
)
def test_invalid_task_always_raises_contract_value_error(task: object) -> None:
    """Invalid tasks, including unhashable values, never leak TypeError."""
    frame = pd.DataFrame({"target": ["a", "a", "b", "b"], "x": [1, 2, 3, 4]})

    with pytest.raises(ValueError, match="task must be classification or regression"):
        analyze_target_relationships(
            frame,
            "target",
            task=task,  # type: ignore[arg-type]
        )


def test_regression_target_validation() -> None:
    """Regression targets must be numeric, finite, sufficiently large, and variable."""
    with pytest.raises(ValueError, match="regression target must be numeric"):
        analyze_target_relationships(
            pd.DataFrame({"target": ["a", "b", "c"], "x": [1, 2, 3]}),
            "target",
            task="regression",
        )
    with pytest.raises(
        ValueError,
        match="regression target must contain only finite non-missing values",
    ):
        analyze_target_relationships(
            pd.DataFrame({"target": [1.0, 2.0, np.inf], "x": [1, 2, 3]}),
            "target",
            task="regression",
        )
    with pytest.raises(
        ValueError, match="regression target must contain at least three finite values"
    ):
        analyze_target_relationships(
            pd.DataFrame({"target": [1.0, 2.0], "x": [1, 2]}),
            "target",
            task="regression",
        )
    with pytest.raises(ValueError, match="regression target must not be constant"):
        analyze_target_relationships(
            pd.DataFrame({"target": [1.0, 1.0, 1.0], "x": [1, 2, 3]}),
            "target",
            task="regression",
        )
    with pytest.raises(ValueError, match="regression target must be numeric"):
        analyze_target_relationships(
            pd.DataFrame(
                {
                    "target": np.array([1 + 0j, 2 + 0j, 3 + 0j]),
                    "x": [3.0, 2.0, 1.0],
                }
            ),
            "target",
            task="regression",
        )


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_complex_target_features_are_skipped_without_scipy_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    task: str,
) -> None:
    """Complex features use unsupported_dtype and never enter statistical paths."""

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("SciPy statistical function was called")

    for name in ("kruskal", "chi2_contingency", "pearsonr"):
        monkeypatch.setattr(analysis_module.stats, name, fail)
    target: list[object]
    if task == "classification":
        target = ["a", "a", "b", "b"]
    else:
        target = [1.0, 2.0, 3.0, 4.0]
    frame = pd.DataFrame(
        {
            "target": target,
            "complex": np.array([1 + 1j, 2 + 1j, 3 + 1j, 4 + 1j]),
        }
    )

    result = analyze_target_relationships(
        frame,
        "target",
        task=task,  # type: ignore[arg-type]
        features=["complex"],
    )

    assert result.skipped_features == ("complex",)
    assert result.skipped_reasons == {"complex": "unsupported_dtype"}
    assert result.statistical_tests.empty


def test_task08_real_numeric_predicate_does_not_change_task07_predicate() -> None:
    """Task 07 keeps complex as numeric while Task 08 narrows to real numeric."""
    dtype = np.dtype("complex128")
    assert analysis_module._is_numeric_non_bool(dtype) is True
    assert analysis_module._is_task08_real_numeric_non_bool(dtype) is False


def test_target_feature_skip_vocabulary_and_order() -> None:
    """Regression features cover dtype, missing, finite, sample, and constant skips."""
    frame = pd.DataFrame(
        {
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
            "unsupported": pd.date_range("2024-01-01", periods=5),
            "all_missing": pd.Series([None] * 5, dtype="Float64"),
            "non_finite": [1.0, 2.0, 3.0, 4.0, np.inf],
            "insufficient": [1.0, np.nan, np.nan, np.nan, np.nan],
            "constant": [2.0] * 5,
        }
    )

    result = analyze_target_relationships(
        frame,
        "target",
        task="regression",
        features=[
            "unsupported",
            "all_missing",
            "non_finite",
            "insufficient",
            "constant",
        ],
    )

    assert result.skipped_features == (
        "unsupported",
        "all_missing",
        "non_finite",
        "insufficient",
        "constant",
    )
    assert result.skipped_reasons == {
        "unsupported": "unsupported_dtype",
        "all_missing": "all_missing",
        "non_finite": "non_finite_values",
        "insufficient": "insufficient_non_missing",
        "constant": "constant",
    }
    assert result.available_feature_count == 0
    assert result.limitations == ()


def test_category_budget_uses_complete_cases_without_result_truncation() -> None:
    """Twenty categories pass while twenty-one skip without result truncation."""
    boundary_categories = [f"c{index}" for index in range(20) for _ in range(2)]
    boundary = analyze_target_relationships(
        pd.DataFrame({"target": ["a", "b"] * 20, "category": boundary_categories}),
        "target",
        task="classification",
    )
    assert boundary.analyzed_features == ("category",)
    assert boundary.skipped_reasons == {}

    categories = [f"c{index}" for index in range(21) for _ in range(2)]
    frame = pd.DataFrame(
        {
            "target": ["a", "b"] * 21 + [None, None],
            "category": categories + ["ignored1", "ignored2"],
        }
    )

    result = analyze_target_relationships(frame, "target", task="classification")

    assert result.available_feature_count == 1
    assert result.skipped_reasons == {"category": "exceeds_max_categories"}
    assert result.truncated is False
    assert result.truncation_reason is None


def test_feature_budget_and_deterministic_ordering() -> None:
    """Earlier skips consume no slots before the first 50 eligible features."""
    data: dict[str, list[object]] = {
        "target": [1.0, 2.0, 3.0, 4.0],
        "unsupported": list(pd.date_range("2024-01-01", periods=4)),
        "all_missing": [np.nan] * 4,
        "constant": [1.0] * 4,
    }
    for index in range(51):
        data[f"x{index}"] = [1.0 + index, 2.0 + index, 3.0 + index, 4.0 + index]
    features = ["unsupported", "all_missing", "constant"] + [
        f"x{index}" for index in range(51)
    ]
    result = analyze_target_relationships(
        pd.DataFrame(data), "target", task="regression", features=features
    )

    assert result.available_feature_count == 51
    assert result.analyzed_features == tuple(f"x{index}" for index in range(50))
    assert result.skipped_features == (
        "unsupported",
        "all_missing",
        "constant",
        "x50",
    )
    assert result.skipped_reasons == {
        "unsupported": "unsupported_dtype",
        "all_missing": "all_missing",
        "constant": "constant",
        "x50": "exceeds_max_features",
    }
    assert result.truncated is True
    assert result.truncation_reason == "exceeds_max_features"
    assert result.statistical_tests["feature"].tolist() == [
        f"x{index}" for index in range(50)
    ]


@pytest.mark.parametrize(
    ("statistic_name", "frame", "task", "feature", "patch_result"),
    [
        (
            "kruskal",
            pd.DataFrame({"target": ["a", "a", "b", "b"], "x": [1, 2, 3, 4]}),
            "classification",
            "x",
            (np.nan, 0.5),
        ),
        (
            "chi2_contingency",
            pd.DataFrame({"target": ["a", "a", "b", "b"], "x": ["u", "v", "u", "v"]}),
            "classification",
            "x",
            (1.0, np.nan, 1, np.ones((2, 2))),
        ),
        (
            "pearsonr",
            pd.DataFrame({"target": [1.0, 2.0, 3.0], "x": [3.0, 2.0, 1.0]}),
            "regression",
            "x",
            (np.nan, 0.5),
        ),
        (
            "kruskal",
            pd.DataFrame(
                {
                    "target": [1.0, 2.0, 3.0, 4.0],
                    "x": ["a", "a", "b", "b"],
                }
            ),
            "regression",
            "x",
            (1.0, np.inf),
        ),
    ],
)
def test_non_finite_scipy_results_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
    statistic_name: str,
    frame: pd.DataFrame,
    task: str,
    feature: str,
    patch_result: tuple[object, ...],
) -> None:
    """Every statistical path maps non-finite SciPy output to one reason."""
    monkeypatch.setattr(
        analysis_module.stats, statistic_name, lambda *args, **kwargs: patch_result
    )

    result = analyze_target_relationships(
        frame,
        "target",
        task=task,  # type: ignore[arg-type]
    )

    assert result.skipped_reasons == {feature: "statistical_test_not_applicable"}
    assert result.statistical_tests.empty
    assert result.numeric_details.empty
    assert result.category_details.empty
    assert result.limitations == ()


def test_non_finite_effect_size_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-finite derived effect size emits no detail or test row."""
    monkeypatch.setattr(analysis_module, "_epsilon_squared", lambda *args: np.inf)
    frame = pd.DataFrame({"target": ["a", "a", "b", "b"], "x": [1.0, 2.0, 3.0, 4.0]})

    result = analyze_target_relationships(frame, "target", task="classification")

    assert result.skipped_reasons == {"x": "statistical_test_not_applicable"}
    assert result.statistical_tests.empty
    assert result.numeric_details.empty


def test_task08_does_not_call_task07_public_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task 08 performs no hidden calls to Task 07 public analysis APIs."""

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("Task 07 public function was called")

    for name in (
        "analyze_numeric_features",
        "analyze_categorical_features",
        "compute_correlations",
        "detect_outliers",
    ):
        monkeypatch.setattr(analysis_module, name, fail)

    compare_groups(pd.DataFrame({"g": ["a", "b"], "x": [1.0, 2.0]}), "g")
    analyze_target_relationships(
        pd.DataFrame({"target": [1.0, 2.0, 3.0], "x": [3.0, 2.0, 1.0]}),
        "target",
        task="regression",
    )


def test_task08_result_instances_are_frozen() -> None:
    """Returned Task 08 dataclass instances reject field assignment."""
    group_result = compare_groups(pd.DataFrame({"g": ["a", "b"], "x": [1.0, 2.0]}), "g")
    target_result = analyze_target_relationships(
        pd.DataFrame({"target": [1.0, 2.0, 3.0], "x": [3.0, 2.0, 1.0]}),
        "target",
        task="regression",
    )
    with pytest.raises(FrozenInstanceError):
        group_result.n_rows = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        target_result.task = "classification"  # type: ignore[misc]


def test_target_analysis_does_not_mutate_nullable_input() -> None:
    """Target analysis leaves nullable dtypes, values, index, and columns untouched."""
    frame = pd.DataFrame(
        {
            "target": [1.0, 2.0, 3.0, 4.0],
            "numeric": [4.0, 3.0, 2.0, 1.0],
            "category": ["a", "a", "b", "b"],
        },
        index=[3, 1, 4, 2],
    ).astype({"target": "Float64", "numeric": "Float64", "category": "string"})
    original = frame.copy(deep=True)

    analyze_target_relationships(frame, "target", task="regression")

    pdt.assert_frame_equal(frame, original)


def test_target_analysis_duplicate_index_labels_do_not_change_statistics() -> None:
    """Complete-case masks preserve rows when index labels are duplicated."""
    frame = pd.DataFrame(
        {
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "numeric": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        },
        index=[0, 0, 1, 1, 2, 2],
    )
    original = frame.copy(deep=True)

    result = analyze_target_relationships(frame, "target", task="regression")

    test_row = result.statistical_tests.iloc[0]
    assert test_row["n_obs"] == 6
    assert test_row["statistic"] == pytest.approx(-1.0)
    assert result.analyzed_features == ("numeric",)
    pdt.assert_frame_equal(frame, original)
