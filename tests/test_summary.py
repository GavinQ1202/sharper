"""Contract tests for DataFrame summaries."""

import numpy as np
import pandas as pd
import pytest

from sharper import DataFrameSummary, SchemaReport, infer_schema, summarize_dataframe

_SUMMARY_DTYPES = {
    "column": "object",
    "pandas_dtype": "object",
    "logical_type": "object",
    "non_null_count": "int64",
    "missing_count": "int64",
    "missing_rate": "float64",
    "unique_count": "int64",
    "unique_rate": "float64",
    "is_constant": "bool",
    "is_id_like": "bool",
    "min": "object",
    "max": "object",
    "mean": "float64",
    "std": "float64",
    "q25": "float64",
    "median": "float64",
    "q75": "float64",
}


def _assert_summary_dtypes(summary: pd.DataFrame) -> None:
    assert {name: str(dtype) for name, dtype in summary.dtypes.items()} == (
        _SUMMARY_DTYPES
    )


def test_summarize_dataframe_matches_explicit_pandas_baselines() -> None:
    """Shape, memory, missingness, uniqueness, and statistics match pandas."""
    frame = pd.DataFrame(
        {
            "value": [1.0, 2.0, np.nan, 4.0],
            "group": ["a", "a", None, "b"],
        }
    )

    result = summarize_dataframe(frame)
    details = result.column_summary.set_index("column")

    assert isinstance(result, DataFrameSummary)
    assert isinstance(result.schema, SchemaReport)
    assert result.n_rows == frame.shape[0]
    assert result.n_columns == frame.shape[1]
    assert result.memory_usage_bytes == int(
        frame.memory_usage(index=True, deep=True).sum()
    )
    assert result.total_missing_cells == int(frame.isna().sum().sum())
    assert result.total_missing_rate == pytest.approx(
        frame.isna().sum().sum() / frame.size
    )
    assert details.loc["value", "non_null_count"] == frame["value"].count()
    assert details.loc["value", "unique_count"] == frame["value"].nunique(dropna=True)
    assert details.loc["value", "min"] == frame["value"].min()
    assert details.loc["value", "max"] == frame["value"].max()
    assert details.loc["value", "mean"] == pytest.approx(frame["value"].mean())
    assert details.loc["value", "std"] == pytest.approx(frame["value"].std(ddof=1))
    assert details.loc["value", "q25"] == pytest.approx(frame["value"].quantile(0.25))
    assert details.loc["value", "median"] == pytest.approx(
        frame["value"].quantile(0.50)
    )
    assert details.loc["value", "q75"] == pytest.approx(frame["value"].quantile(0.75))
    assert details.loc["group", "min"] is None
    assert pd.isna(details.loc["group", "mean"])


def test_column_summary_order_and_dtypes_are_frozen() -> None:
    """The detail table exactly follows the 17-column contract."""
    result = summarize_dataframe(pd.DataFrame({"value": [1.0, 2.0, 1.0]}))

    assert list(result.column_summary.columns) == list(_SUMMARY_DTYPES)
    _assert_summary_dtypes(result.column_summary)


def test_zero_by_zero_dataframe_has_typed_empty_summary() -> None:
    """A 0x0 frame returns a 0-row detail table with every frozen dtype."""
    result = summarize_dataframe(pd.DataFrame())

    assert result.n_rows == 0
    assert result.n_columns == 0
    assert result.total_missing_cells == 0
    assert result.total_missing_rate == 0.0
    assert result.column_summary.empty
    assert list(result.column_summary.columns) == list(_SUMMARY_DTYPES)
    _assert_summary_dtypes(result.column_summary)


def test_zero_rows_with_columns_has_one_typed_summary_row_per_column() -> None:
    """Typed empty input columns remain represented while their schema is unknown."""
    frame = pd.DataFrame(
        {
            "value": pd.Series(dtype="float64"),
            "category": pd.Series(dtype="string"),
        }
    )

    result = summarize_dataframe(frame)

    assert result.column_summary["column"].tolist() == ["value", "category"]
    assert result.column_summary["logical_type"].tolist() == ["unknown", "unknown"]
    assert result.column_summary["non_null_count"].tolist() == [0, 0]
    assert result.column_summary["missing_rate"].tolist() == [0.0, 0.0]
    assert result.column_summary["min"].tolist() == [None, None]
    assert (
        result.column_summary[["mean", "std", "q25", "median", "q75"]]
        .isna()
        .all(axis=None)
    )
    _assert_summary_dtypes(result.column_summary)


def test_all_missing_and_single_value_numeric_statistics() -> None:
    """Undefined statistics use the frozen None and NaN representations."""
    frame = pd.DataFrame(
        {
            "all_missing": pd.Series([pd.NA, pd.NA], dtype="Float64"),
            "one_value": [3.5, np.nan],
        }
    )

    details = summarize_dataframe(frame).column_summary.set_index("column")

    assert details.loc["all_missing", "min"] is None
    assert details.loc["all_missing", "max"] is None
    assert pd.isna(details.loc["all_missing", "mean"])
    assert details.loc["one_value", "min"] == 3.5
    assert details.loc["one_value", "max"] == 3.5
    assert pd.isna(details.loc["one_value", "std"])


def test_supplied_schema_is_reused_and_validated() -> None:
    """A matching schema is preserved; shape, names, and order mismatches fail."""
    frame = pd.DataFrame({"value": [1.0, 2.0], "group": ["a", "a"]})
    schema = infer_schema(frame)

    result = summarize_dataframe(frame, schema=schema)

    assert result.schema is schema
    with pytest.raises(ValueError, match="schema must match"):
        summarize_dataframe(frame[["group", "value"]], schema=schema)
    with pytest.raises(ValueError, match="schema must match"):
        summarize_dataframe(frame.iloc[:1], schema=schema)


def test_summary_rejects_non_string_and_duplicate_column_names() -> None:
    """Summary applies the same input-column boundary as schema inference."""
    with pytest.raises(
        ValueError,
        match="DataFrame column names must all be strings",
    ):
        summarize_dataframe(pd.DataFrame([[1]], columns=[1]))

    with pytest.raises(ValueError, match="must be unique"):
        summarize_dataframe(pd.DataFrame([[1, 2]], columns=["x", "x"]))


def test_summarize_dataframe_does_not_mutate_input() -> None:
    """Summary calculations leave the caller's DataFrame unchanged."""
    frame = pd.DataFrame(
        {
            "value": pd.Series([1.0, pd.NA, 3.0], dtype="Float64"),
            "category": pd.Series(["a", None, "b"], dtype="string"),
        }
    )
    before = frame.copy(deep=True)

    summarize_dataframe(frame)

    pd.testing.assert_frame_equal(frame, before)
