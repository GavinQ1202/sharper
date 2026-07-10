"""Dataset-level and column-level descriptive summaries."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from sharper.schema import (
    SchemaReport,
    _validate_dataframe_columns,
    infer_schema,
)

_COLUMN_SUMMARY_DTYPES = {
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


@dataclass(frozen=True)
class DataFrameSummary:
    """Contain dataset-level and ordered column-level descriptive statistics.

    Attributes
    ----------
    n_rows
        Number of input rows.
    n_columns
        Number of input columns.
    memory_usage_bytes
        Deep memory usage including the index.
    total_missing_cells
        Number of missing cells in the complete frame.
    total_missing_rate
        Missing cells divided by all cells, or zero when there are no cells.
    schema
        Validated schema used to select applicable statistics.
    column_summary
        Seventeen-column DataFrame with the frozen Task 03 order and dtypes.
    """

    n_rows: int
    n_columns: int
    memory_usage_bytes: int
    total_missing_cells: int
    total_missing_rate: float
    schema: SchemaReport
    column_summary: pd.DataFrame


def summarize_dataframe(
    df: pd.DataFrame,
    *,
    schema: SchemaReport | None = None,
) -> DataFrameSummary:
    """Summarize a DataFrame without cleaning or modifying it.

    Parameters
    ----------
    df
        DataFrame with unique string column names.
    schema
        Optional schema previously inferred for the same shape, names, and order.
        When omitted, :func:`sharper.schema.infer_schema` is called.

    Returns
    -------
    DataFrameSummary
        Shape, deep memory use, total missingness, schema, and ordered column
        statistics.

    Raises
    ------
    ValueError
        If column names are duplicated or non-string, or a supplied schema does
        not match the DataFrame shape, column names, and order.

    Notes
    -----
    Missing values are excluded from numeric statistics. Only logical numeric
    columns receive min, max, mean, sample standard deviation, and quantiles.
    The function does not mutate ``df`` or a supplied ``schema``.

    Examples
    --------
    >>> import pandas as pd
    >>> from sharper import summarize_dataframe
    >>> result = summarize_dataframe(pd.DataFrame({"score": [1.0, 2.0]}))
    >>> result.n_rows
    2
    """
    _validate_dataframe_columns(df)
    resolved_schema = infer_schema(df) if schema is None else schema
    _validate_schema_matches(df, resolved_schema)

    n_rows, n_columns = df.shape
    total_missing_cells = int(df.isna().sum().sum())
    total_cells = n_rows * n_columns
    return DataFrameSummary(
        n_rows=n_rows,
        n_columns=n_columns,
        memory_usage_bytes=int(df.memory_usage(index=True, deep=True).sum()),
        total_missing_cells=total_missing_cells,
        total_missing_rate=(
            float(total_missing_cells / total_cells) if total_cells else 0.0
        ),
        schema=resolved_schema,
        column_summary=_build_column_summary(df, resolved_schema),
    )


def _validate_schema_matches(df: pd.DataFrame, schema: SchemaReport) -> None:
    schema_names = [column.name for column in schema.columns]
    if (
        schema.n_rows != len(df)
        or schema.n_columns != len(df.columns)
        or schema_names != list(df.columns)
    ):
        raise ValueError(
            "schema must match the DataFrame shape, column names, and column order"
        )


def _build_column_summary(
    df: pd.DataFrame,
    schema: SchemaReport,
) -> pd.DataFrame:
    values: dict[str, list[object]] = {
        column_name: [] for column_name in _COLUMN_SUMMARY_DTYPES
    }
    for column_schema in schema.columns:
        series = df[column_schema.name]
        non_null_count = len(series) - column_schema.missing_count
        statistics = _numeric_statistics(series, column_schema.logical_type)
        row = {
            "column": column_schema.name,
            "pandas_dtype": column_schema.pandas_dtype,
            "logical_type": column_schema.logical_type,
            "non_null_count": non_null_count,
            "missing_count": column_schema.missing_count,
            "missing_rate": column_schema.missing_rate,
            "unique_count": column_schema.unique_count,
            "unique_rate": column_schema.unique_rate,
            "is_constant": column_schema.is_constant,
            "is_id_like": column_schema.is_id_like,
            **statistics,
        }
        for column_name in _COLUMN_SUMMARY_DTYPES:
            values[column_name].append(row[column_name])

    return pd.DataFrame(
        {
            column_name: pd.Series(values[column_name], dtype=dtype)
            for column_name, dtype in _COLUMN_SUMMARY_DTYPES.items()
        }
    )


def _numeric_statistics(
    series: pd.Series,
    logical_type: str,
) -> dict[str, object]:
    if logical_type != "numeric" or series.dropna().empty:
        return {
            "min": None,
            "max": None,
            "mean": float("nan"),
            "std": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
        }

    return {
        "min": series.min(skipna=True),
        "max": series.max(skipna=True),
        "mean": float(series.mean(skipna=True)),
        "std": float(series.std(skipna=True, ddof=1)),
        "q25": float(series.quantile(0.25)),
        "median": float(series.quantile(0.50)),
        "q75": float(series.quantile(0.75)),
    }
