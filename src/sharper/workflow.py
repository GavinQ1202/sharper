"""The complete, public-API-only Task 13 analysis workflow."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Integral, Real
from typing import Literal

import pandas as pd

from sharper.analysis import (
    CategoricalAnalysis,
    CorrelationAnalysis,
    GroupComparison,
    NumericAnalysis,
    OutlierAnalysis,
    TargetAnalysis,
    analyze_categorical_features,
    analyze_numeric_features,
    analyze_target_relationships,
    compare_groups,
    compute_correlations,
    detect_outliers,
)
from sharper.evaluation import (
    ClassificationEvaluation,
    RegressionEvaluation,
    evaluate_model,
)
from sharper.features import FeatureSuggestionReport, suggest_feature_derivations
from sharper.modeling import (
    RegressionTrainingResult,
    TrainingResult,
    train_classifier,
    train_regressor,
)
from sharper.quality import QualityReport, check_data_quality
from sharper.schema import SchemaReport, infer_schema
from sharper.summary import DataFrameSummary, summarize_dataframe
from sharper.visualization import (
    PlotCollection,
    PlotResult,
    plot_classification_evaluation,
    plot_correlations,
    plot_distributions,
    plot_group_comparison,
    plot_missingness,
    plot_outliers,
    plot_regression_evaluation,
    plot_target_relationships,
)


@dataclass(frozen=True)
class AnalysisRun:
    """All Task 13 results and recorded workflow configuration.

    The result contains only upstream result objects, never the input DataFrame.
    Nested DataFrames, estimators, and Figures remain caller-owned until reporting
    acquires Figure cleanup ownership.
    """

    schema: SchemaReport
    summary: DataFrameSummary
    quality: QualityReport
    target: str | None
    task: Literal["classification", "regression"] | None
    include_model: bool
    id_columns: tuple[str, ...]
    exclude_columns: tuple[str, ...]
    features: tuple[str, ...] | None
    time_column: str | None
    group_by: str | None
    reference_date: str | None
    max_suggestions: int
    test_size: float
    random_state: int | None
    numeric_analysis: NumericAnalysis
    categorical_analysis: CategoricalAnalysis
    correlation_analysis: CorrelationAnalysis
    outlier_analysis: OutlierAnalysis
    group_comparison: GroupComparison | None
    target_analysis: TargetAnalysis | None
    feature_suggestions: FeatureSuggestionReport
    training: TrainingResult | RegressionTrainingResult | None
    evaluation: ClassificationEvaluation | RegressionEvaluation | None
    distribution_plots: PlotCollection
    missingness_plot: PlotResult
    correlation_plot: PlotResult
    outlier_plots: PlotCollection
    group_plots: PlotCollection | None
    target_plots: PlotCollection | None
    evaluation_plots: PlotCollection | None
    skipped: tuple[str, ...]
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]


def run_analysis(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    task: Literal["classification", "regression"] | None = None,
    include_model: bool = False,
    id_columns: Sequence[str] = (),
    exclude_columns: Sequence[str] = (),
    features: Sequence[str] | None = None,
    time_column: str | None = None,
    group_by: str | None = None,
    reference_date: str | date | datetime | pd.Timestamp | None = None,
    max_suggestions: int = 50,
    test_size: float = 0.20,
    random_state: int | None = 42,
) -> AnalysisRun:
    """Run the deterministic Task 13 analysis workflow without mutating ``df``.

    All domain work is delegated once to the frozen Tasks 03--12 public APIs.
    Target-aware analysis needs both ``target`` and ``task``; model training also
    needs ``include_model=True``. Missing values are handled by those APIs.
    """
    _validate_df(df)
    if not isinstance(include_model, bool):
        raise ValueError("include_model must be a boolean")
    if task is not None and task not in {"classification", "regression"}:
        raise ValueError("task must be classification or regression")
    if task is not None and target is None:
        raise ValueError("task requires target")
    if include_model and (target is None or task is None):
        raise ValueError("modeling requires target and task")
    if target is not None and (not isinstance(target, str) or target not in df.columns):
        raise ValueError(f"target column not found: {target!r}")
    if (
        isinstance(max_suggestions, bool)
        or not isinstance(max_suggestions, Integral)
        or max_suggestions < 1
    ):
        raise ValueError("max_suggestions must be a positive integer")
    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, Real)
        or not 0 < float(test_size) < 1
    ):
        raise ValueError("test_size must be strictly between 0 and 1")
    if random_state is not None and (
        isinstance(random_state, bool)
        or not isinstance(random_state, Integral)
        or random_state < 0
    ):
        raise ValueError("random_state must be a non-negative integer or None")
    ids = _columns(
        df,
        id_columns,
        "id_columns and exclude_columns must be sequences of unique column names",
    )
    excludes = _columns(
        df,
        exclude_columns,
        "id_columns and exclude_columns must be sequences of unique column names",
    )
    if set(ids) & set(excludes):
        raise ValueError("id_columns and exclude_columns must not overlap")
    if target in {*ids, *excludes}:
        raise ValueError("target must not appear in id_columns or exclude_columns")
    resolved_features = (
        None
        if features is None
        else _columns(
            df,
            features,
            "features must be a non-empty sequence of unique column names",
            empty=False,
        )
    )
    if resolved_features is not None:
        if target in resolved_features:
            raise ValueError("target must not appear in features")
        if set(resolved_features) & set(excludes):
            raise ValueError("features and exclude_columns must not overlap")
        if not include_model:
            raise ValueError("features require include_model=True")
    if time_column is not None:
        if not isinstance(time_column, str):
            raise ValueError("time_column must be a column name string or None")
        if time_column not in df.columns:
            raise ValueError(f"time column not found: {time_column!r}")
        if not include_model:
            raise ValueError("time_column requires include_model=True")
    if group_by is not None:
        if not isinstance(group_by, str):
            raise ValueError("group_by must be a column name string or None")
        if group_by not in df.columns:
            raise ValueError(f"group column not found: {group_by!r}")
        if group_by == target or group_by in {*ids, *excludes}:
            raise ValueError("group_by must not be target, id, or excluded")

    analysis_columns = tuple(
        c for c in df.columns if c != target and c not in {*ids, *excludes}
    )
    schema = _step("infer_schema", infer_schema, df, target=target)
    summary = _step("summarize_dataframe", summarize_dataframe, df, schema=schema)
    quality = _step("check_data_quality", check_data_quality, df, schema=schema)
    numeric = _step(
        "analyze_numeric_features",
        analyze_numeric_features,
        df,
        columns=analysis_columns,
    )
    categorical = _step(
        "analyze_categorical_features",
        analyze_categorical_features,
        df,
        columns=analysis_columns,
    )
    correlations = _step(
        "compute_correlations", compute_correlations, df, columns=analysis_columns
    )
    outliers = _step("detect_outliers", detect_outliers, df, columns=analysis_columns)
    group = (
        _step(
            "compare_groups",
            compare_groups,
            df.loc[:, analysis_columns],
            group_by,
            values=None,
        )
        if group_by
        else None
    )
    target_analysis = (
        _step(
            "analyze_target_relationships",
            analyze_target_relationships,
            df,
            target,
            task=task,
            features=analysis_columns,
        )
        if target and task
        else None
    )
    effective_exclusions = tuple(dict.fromkeys((*ids, *excludes)))
    suggestions = _step(
        "suggest_feature_derivations",
        suggest_feature_derivations,
        df,
        schema=schema,
        target=target,
        exclude_columns=effective_exclusions,
        reference_date=reference_date,
        max_suggestions=int(max_suggestions),
    )
    training: TrainingResult | RegressionTrainingResult | None = None
    evaluation: ClassificationEvaluation | RegressionEvaluation | None = None
    if include_model:
        trainer = train_classifier if task == "classification" else train_regressor
        training = _step(
            trainer.__name__,
            trainer,
            df,
            target,
            features=resolved_features,
            exclude_columns=effective_exclusions,
            time_column=time_column,
            test_size=float(test_size),
            random_state=None if random_state is None else int(random_state),
        )
        evaluation = _step("evaluate_model", evaluate_model, training)
    distributions = _step("plot_distributions", plot_distributions, df)
    missingness = _step("plot_missingness", plot_missingness, df)
    correlation_plot = _step("plot_correlations", plot_correlations, correlations)
    outlier_plots = _step("plot_outliers", plot_outliers, outliers)
    group_plots = (
        _step("plot_group_comparison", plot_group_comparison, group) if group else None
    )
    target_plots = (
        _step("plot_target_relationships", plot_target_relationships, target_analysis)
        if target_analysis
        else None
    )
    evaluation_plots = None
    if evaluation:
        plotter = (
            plot_classification_evaluation
            if task == "classification"
            else plot_regression_evaluation
        )
        evaluation_plots = _step(plotter.__name__, plotter, evaluation)
    skipped = tuple(
        x
        for x, enabled in (
            ("group_comparison_not_requested", bool(group_by)),
            ("target_analysis_not_requested", bool(target and task)),
            ("modeling_not_requested", include_model),
            ("evaluation_not_requested", bool(evaluation)),
        )
        if not enabled
    )
    warnings = training.warnings if training else ()
    limitations = tuple(
        dict.fromkeys(
            (
                *((target_analysis.limitations) if target_analysis else ()),
                *((training.limitations) if training else ()),
                *((evaluation.limitations) if evaluation else ()),
            )
        )
    )
    return AnalysisRun(
        schema,
        summary,
        quality,
        target,
        task,
        include_model,
        ids,
        excludes,
        resolved_features,
        time_column,
        group_by,
        suggestions.reference_date,
        int(max_suggestions),
        float(test_size),
        None if random_state is None else int(random_state),
        numeric,
        categorical,
        correlations,
        outliers,
        group,
        target_analysis,
        suggestions,
        training,
        evaluation,
        distributions,
        missingness,
        correlation_plot,
        outlier_plots,
        group_plots,
        target_plots,
        evaluation_plots,
        skipped,
        warnings,
        limitations,
    )


def _validate_df(df: object) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if not df.columns.is_unique or not all(isinstance(c, str) for c in df.columns):
        raise ValueError("DataFrame column names must be unique strings")


def _columns(
    df: pd.DataFrame, value: Sequence[str], message: str, *, empty: bool = True
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(message)
    result = tuple(value)
    if (
        (not empty and not result)
        or not all(isinstance(x, str) for x in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError(message)
    for column in result:
        if column not in df.columns:
            raise ValueError(f"column not found: {column!r}")
    return result


def _step(name: str, function: object, *args: object, **kwargs: object) -> object:
    try:
        return function(*args, **kwargs)  # type: ignore[operator]
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception as error:
        raise ValueError(f"workflow step failed: {name}: {error}") from error
