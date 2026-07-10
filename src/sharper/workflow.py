"""Minimal public analysis workflow for Task 05."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from sharper.quality import QualityReport, check_data_quality
from sharper.schema import SchemaReport, infer_schema
from sharper.summary import DataFrameSummary, summarize_dataframe

_SKIPPED = ("modeling", "visualization", "feature_engineering")


@dataclass(frozen=True)
class AnalysisRun:
    """Contain the deterministic results and recorded options of an analysis run.

    Attributes
    ----------
    schema
        Schema inferred from the complete input DataFrame.
    summary
        Descriptive summary of the complete input DataFrame.
    quality
        Minimal data-quality report for the complete input DataFrame.
    target
        Validated target name, or ``None``.
    task
        ``"classification"``, ``"regression"``, or ``None``.
    include_model
        Always ``False`` in Task 05.
    id_columns
        Validated identifier columns, recorded but not yet applied.
    exclude_columns
        Validated exclusion columns, recorded but not yet applied.
    random_state
        Validated integer seed, recorded but not used in Task 05.
    skipped
        Capabilities deliberately not executed in Task 05.
    warnings
        Ordered notices about recorded options not yet applied.
    """

    schema: SchemaReport
    summary: DataFrameSummary
    quality: QualityReport
    target: str | None
    task: str | None
    include_model: bool
    id_columns: tuple[str, ...]
    exclude_columns: tuple[str, ...]
    random_state: int
    skipped: tuple[str, ...]
    warnings: tuple[str, ...]


def run_analysis(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    task: str | None = None,
    include_model: bool = False,
    id_columns: Sequence[str] = (),
    exclude_columns: Sequence[str] = (),
    random_state: int = 42,
) -> AnalysisRun:
    """Run the minimal schema, summary, and quality workflow.

    Parameters
    ----------
    df
        DataFrame with unique string column names. It is inspected but not modified.
    target
        Optional existing target column. Task 05 records it but performs no
        target-aware analysis.
    task
        Optional ``"classification"`` or ``"regression"`` task. A target is
        required when this is supplied.
    include_model
        Must remain ``False`` because modeling is unavailable in Task 05.
    id_columns
        Existing columns to record as identifiers; they are not applied yet.
    exclude_columns
        Existing columns to record as exclusions; they are not applied yet.
    random_state
        Integer seed to record. Task 05 performs no random operation.

    Returns
    -------
    AnalysisRun
        Immutable workflow results, recorded options, skipped capabilities, and
        warnings.

    Raises
    ------
    ValueError
        If the DataFrame or any workflow option violates the Task 05 contract.

    Notes
    -----
    Missing values are handled by the composed schema, summary, and quality APIs.
    The function has no file-system side effects and does not mutate ``df``.

    Examples
    --------
    >>> import pandas as pd
    >>> from sharper import run_analysis
    >>> run_analysis(pd.DataFrame({"value": [1, 2]})).summary.n_rows
    2
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if include_model:
        raise ValueError("modeling is not available in Task 05")
    if task is not None and task not in {"classification", "regression"}:
        raise ValueError("task must be classification or regression")
    if task is not None and target is None:
        raise ValueError("task requires target")
    if not isinstance(random_state, int):
        raise ValueError("random_state must be an integer")

    resolved_id_columns = tuple(id_columns)
    resolved_exclude_columns = tuple(exclude_columns)
    _validate_column_parameters(
        df,
        id_columns=resolved_id_columns,
        exclude_columns=resolved_exclude_columns,
    )

    schema = infer_schema(df, target=target)
    summary = summarize_dataframe(df)
    quality = check_data_quality(df, schema=schema)

    warnings: list[str] = []
    if target is not None:
        warnings.append(
            "target recorded but target analysis is not available in Task 05"
        )
    if task is not None:
        warnings.append("task recorded but modeling is not available in Task 05")
    if resolved_id_columns:
        warnings.append("id_columns recorded but not applied in Task 05")
    if resolved_exclude_columns:
        warnings.append("exclude_columns recorded but not applied in Task 05")

    return AnalysisRun(
        schema=schema,
        summary=summary,
        quality=quality,
        target=target,
        task=task,
        include_model=False,
        id_columns=resolved_id_columns,
        exclude_columns=resolved_exclude_columns,
        random_state=random_state,
        skipped=_SKIPPED,
        warnings=tuple(warnings),
    )


def _validate_column_parameters(
    df: pd.DataFrame,
    *,
    id_columns: tuple[str, ...],
    exclude_columns: tuple[str, ...],
) -> None:
    if len(set(id_columns)) != len(id_columns) or len(set(exclude_columns)) != len(
        exclude_columns
    ):
        raise ValueError("duplicate column parameter")
    if set(id_columns) & set(exclude_columns):
        raise ValueError("id_columns and exclude_columns must not overlap")
    for column in (*id_columns, *exclude_columns):
        if column not in df.columns:
            raise ValueError(f"column not found: {column!r}")
