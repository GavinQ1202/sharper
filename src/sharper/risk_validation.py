"""Leakage-aware binary risk validation with frozen audit tables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)
from sklearn.base import ClassifierMixin, clone, is_classifier
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

from sharper.evaluation import (
    _RISK_CALIBRATION_COLUMNS,
    _RISK_GAINS_COLUMNS,
    _RISK_METRIC_COLUMNS,
    _RISK_THRESHOLD_COLUMNS,
    _binary_risk_business_primitive,
    _binary_risk_evaluation,
    _risk_float_array,
)
from sharper.modeling import _fit_classifier_fold, _label_key

RiskLabel = str | int | bool
RiskLabelInput = RiskLabel | np.generic


@dataclass(frozen=True)
class BinaryRiskValidationConfig:
    """Configure one leakage-aware binary-risk validation plan.

    Attributes
    ----------
    validation_mode
        The closed stratified, group, or time validation strategy.
    n_splits, test_size, random_state
        Mode-specific split controls. Unused controls must remain ``None``.
    group_column
        Required only by group modes and never used as a model feature.
    observation_time_column, event_time_column, outcome_end_time_column,
    label_available_time_column, maturity_source, prediction_horizon,
    prediction_horizon_column, reporting_delay, fold_cutoffs, validation_end,
    analysis_as_of
        Explicit time-outcome and label-maturity provenance.
    thresholds, threshold_kind, operating_metric, calibration_bins,
    gain_fractions
        Bounded, caller-declared diagnostic controls.
    exposure_column, observed_loss_column,
    observed_loss_available_time_column, observed_loss_is_mature_snapshot,
    loss_fraction, exposure_unit
        Optional model-independent risk-summary inputs and provenance.

    Notes
    -----
    The dataclass is shallow frozen. Construction has no side effects; runtime
    validation occurs in :func:`validate_binary_risk`.

    Examples
    --------
    >>> config = BinaryRiskValidationConfig(
    ...     validation_mode="stratified_kfold", n_splits=5
    ... )
    """

    validation_mode: Literal[
        "stratified_holdout",
        "stratified_kfold",
        "group_holdout",
        "group_kfold",
        "time_holdout",
        "time_forward",
    ]
    n_splits: int | None = None
    test_size: float | None = None
    random_state: int = 42
    group_column: str | None = None
    observation_time_column: str | None = None
    event_time_column: str | None = None
    outcome_end_time_column: str | None = None
    label_available_time_column: str | None = None
    maturity_source: (
        Literal["label_available_time", "observation_horizon", "outcome_end"] | None
    ) = None
    prediction_horizon: timedelta | None = None
    prediction_horizon_column: str | None = None
    reporting_delay: timedelta = timedelta(0)
    fold_cutoffs: tuple[datetime, ...] = ()
    validation_end: datetime | None = None
    analysis_as_of: datetime | None = None
    thresholds: tuple[float, ...] = ()
    threshold_kind: Literal["ranking_score", "event_probability"] | None = None
    operating_metric: (
        Literal[
            "sensitivity", "specificity", "precision", "negative_predictive_value", "f1"
        ]
        | None
    ) = None
    calibration_bins: int = 10
    gain_fractions: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.50, 1.0)
    exposure_column: str | None = None
    observed_loss_column: str | None = None
    observed_loss_available_time_column: str | None = None
    observed_loss_is_mature_snapshot: bool = False
    loss_fraction: float | str | None = None
    exposure_unit: str | None = None


@dataclass(frozen=True)
class ExternalRiskPredictions:
    """Provide caller-frozen OOF/validation scores with fit provenance.

    Attributes
    ----------
    row_positions, fold_ids, fold_fit_row_positions
        Exact zero-based row-position membership and per-fold fit rows.
    ranking_scores, ranking_direction
        Optional finite ranking values and their explicit risk direction.
    event_probabilities, probability_positive_label, probability_provenance
        Optional event probabilities, exact positive-label mapping, and source
        declaration.

    Notes
    -----
    The object is shallow frozen and is never modified. Missing predictions are
    not imputed; membership must exactly match the reconstructed plan.

    Examples
    --------
    >>> external = ExternalRiskPredictions(
    ...     (2, 3), (0, 0), ((0, (0, 1)),),
    ...     (0.2, 0.8), "higher_risk", None, None, None
    ... )
    """

    row_positions: tuple[int, ...]
    fold_ids: tuple[int, ...]
    fold_fit_row_positions: tuple[tuple[int, tuple[int, ...]], ...]
    ranking_scores: tuple[float, ...] | None
    ranking_direction: Literal["higher_risk", "lower_risk"] | None
    event_probabilities: tuple[float, ...] | None
    probability_positive_label: RiskLabelInput | None
    probability_provenance: (
        Literal["predict_proba", "fold_safe_calibrated", "external_declared"] | None
    )


@dataclass(frozen=True)
class BinaryRiskValidationResult:
    """Return frozen binary-risk validation and provenance evidence.

    Attributes
    ----------
    target, positive_label, validation_mode, config
        Resolved target semantics and a normalized configuration snapshot.
    prediction_scope, score_source, score_direction, probability_provenance
        Prediction source and direction provenance.
    input_n_rows, eligible_n_rows, predicted_n_rows, evaluable_n_rows
        Population counts after target, plan, prediction, and maturity rules.
    requested_threshold_count, actual_threshold_count
        Requested and deduplicated threshold counts.
    observed_loss_maturity_mode, observed_loss_analysis_as_of,
    observed_loss_mature_n, observed_loss_excluded_n
        Independent observed-loss maturity provenance and counts.
    folds, predictions, excluded_rows, metrics, gains, calibration,
    threshold_analysis, operating_point, business_metrics
        Deep-copied tables with frozen schemas and deterministic row ordering.
    warnings, limitations
        Ordered closed-vocabulary audit messages.

    Notes
    -----
    The dataclass is shallow frozen and contains no estimator, Pipeline,
    calibrator, Figure, raw DataFrame, or file handle.

    Examples
    --------
    >>> # result = validate_binary_risk(frame, "target", config=config, estimator=model)
    >>> # result.metrics.loc[result.metrics["scope"] == "overall"]
    """

    target: str
    positive_label: str | int | bool
    validation_mode: Literal[
        "stratified_holdout",
        "stratified_kfold",
        "group_holdout",
        "group_kfold",
        "time_holdout",
        "time_forward",
    ]
    config: BinaryRiskValidationConfig
    prediction_scope: Literal["validation", "oof"]
    score_source: Literal[
        "estimator_predict_proba",
        "estimator_decision_function",
        "external_ranking_score",
        "external_event_probability",
        "external_ranking_and_probability",
    ]
    score_direction: Literal["higher_positive_event_risk"]
    probability_provenance: (
        Literal["predict_proba", "fold_safe_calibrated", "external_declared"] | None
    )
    input_n_rows: int
    eligible_n_rows: int
    predicted_n_rows: int
    evaluable_n_rows: int
    requested_threshold_count: int
    actual_threshold_count: int
    observed_loss_maturity_mode: Literal[
        "not_provided", "availability_column", "mature_snapshot"
    ]
    observed_loss_analysis_as_of: datetime | None
    observed_loss_mature_n: int
    observed_loss_excluded_n: int
    folds: pd.DataFrame
    predictions: pd.DataFrame
    excluded_rows: pd.DataFrame
    metrics: pd.DataFrame
    gains: pd.DataFrame
    calibration: pd.DataFrame
    threshold_analysis: pd.DataFrame
    operating_point: pd.DataFrame
    business_metrics: pd.DataFrame
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]


_FOLD_COLUMNS = (
    "fold_id",
    "cutoff",
    "validation_start",
    "validation_end",
    "analysis_as_of",
    "outcome_end_source",
    "prediction_horizon_source",
    "prediction_horizon_value",
    "prediction_horizon_column",
    "reporting_delay_source",
    "reporting_delay",
    "train_row_positions",
    "validation_row_positions",
    "evaluable_validation_row_positions",
    "feature_columns",
    "train_candidate_n",
    "train_n",
    "train_positive_n",
    "train_positive_rate",
    "validation_n",
    "validation_mature_n",
    "validation_excluded_n",
    "evaluable_validation_n",
    "evaluable_positive_n",
    "evaluable_positive_rate",
    "immature_train_n",
    "purged_train_n",
    "immature_validation_n",
    "maturity_source",
)
_PREDICTION_COLUMNS = (
    "row_position",
    "fold_id",
    "target_value",
    "is_evaluable",
    "unevaluable_reason",
    "ranking_score",
    "event_probability",
)
_EXCLUDED_COLUMNS = ("row_position", "reason")
_OPERATING_COLUMNS = (
    "threshold_kind",
    "operating_metric",
    "threshold",
    "metric_value",
    "candidate_count",
    "status",
    "reason",
)
_BUSINESS_COLUMNS = (
    "segment_kind",
    "segment_value",
    "metric",
    "value",
    "status",
    "reason",
    "n_rows",
    "n_evaluable_rows",
    "n_observed_loss_mature_rows",
    "unit",
)
_MODES = (
    "stratified_holdout",
    "stratified_kfold",
    "group_holdout",
    "group_kfold",
    "time_holdout",
    "time_forward",
)
_OPERATING_METRICS = (
    "sensitivity",
    "specificity",
    "precision",
    "negative_predictive_value",
    "f1",
)


@dataclass(frozen=True)
class _FoldPlan:
    fold_id: int
    train_positions: tuple[int, ...]
    validation_positions: tuple[int, ...]
    evaluable_positions: tuple[int, ...]
    train_candidate_n: int
    immature_train_n: int
    cutoff: datetime | None = None
    validation_start: datetime | None = None
    validation_end: datetime | None = None


def _result_frame(
    rows: list[dict[str, object]],
    columns: tuple[str, ...],
    *,
    integers: tuple[str, ...] = (),
    floats: tuple[str, ...] = (),
    booleans: tuple[str, ...] = (),
    objects: tuple[str, ...] = (),
) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=columns)
    for column in integers:
        frame[column] = pd.array(frame[column], dtype="Int64")
    for column in floats:
        frame[column] = pd.array(frame[column], dtype="Float64")
    for column in booleans:
        frame[column] = pd.array(frame[column], dtype="boolean")
    for column in objects:
        frame[column] = pd.Series(frame[column].tolist(), dtype="object")
    return frame


def _config_error() -> ValueError:
    return ValueError("binary risk validation config has invalid schema")


def _normalise_label(value: object) -> RiskLabel:
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) is bool:
        return value
    if type(value) is int:
        return value
    if type(value) is str:
        return value
    raise ValueError(
        "binary target labels must be homogeneous string, integer, or boolean values"
    )


def _normalise_datetime(value: object) -> datetime:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if type(value) is not datetime or value.tzinfo is not None:
        raise _config_error()
    return value


def _normalise_duration(value: object) -> timedelta:
    if isinstance(value, pd.Timedelta):
        value = value.to_pytimedelta()
    if type(value) is not timedelta:
        raise _config_error()
    return value


def _is_real(value: object) -> bool:
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, Real)


def _validate_name(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value:
        raise _config_error()
    return value


def _normalise_config(
    config: BinaryRiskValidationConfig,
) -> tuple[
    BinaryRiskValidationConfig, tuple[float, ...], tuple[float, ...], bool, bool
]:
    if type(config) is not BinaryRiskValidationConfig:
        raise _config_error()
    if type(config.validation_mode) is not str or config.validation_mode not in _MODES:
        raise _config_error()
    if config.maturity_source is not None and (
        type(config.maturity_source) is not str
        or config.maturity_source
        not in ("label_available_time", "observation_horizon", "outcome_end")
    ):
        raise _config_error()
    if config.threshold_kind is not None and type(config.threshold_kind) is not str:
        raise _config_error()
    if config.operating_metric is not None and type(config.operating_metric) is not str:
        raise _config_error()
    if type(config.random_state) is not int or config.random_state < 0:
        raise _config_error()
    names = {
        name: _validate_name(getattr(config, name))
        for name in (
            "group_column",
            "observation_time_column",
            "event_time_column",
            "outcome_end_time_column",
            "label_available_time_column",
            "prediction_horizon_column",
            "exposure_column",
            "observed_loss_column",
            "observed_loss_available_time_column",
            "exposure_unit",
        )
    }
    if type(config.observed_loss_is_mature_snapshot) is not bool:
        raise _config_error()
    delay = _normalise_duration(config.reporting_delay)
    if delay < timedelta(0):
        raise _config_error()
    horizon = (
        None
        if config.prediction_horizon is None
        else _normalise_duration(config.prediction_horizon)
    )
    if horizon is not None and horizon <= timedelta(0):
        raise _config_error()
    if (
        type(config.calibration_bins) is not int
        or not 2 <= config.calibration_bins <= 50
    ):
        raise _config_error()
    if type(config.thresholds) is not tuple or len(config.thresholds) > 100:
        raise _config_error()
    requested_thresholds: list[float] = []
    for value in config.thresholds:
        if not _is_real(value) or not np.isfinite(float(value)):
            raise ValueError("threshold candidates are invalid")
        requested_thresholds.append(float(value))
    thresholds = tuple(sorted(set(requested_thresholds)))
    duplicate_thresholds = len(thresholds) != len(requested_thresholds)
    if thresholds:
        if config.threshold_kind not in ("ranking_score", "event_probability"):
            raise _config_error()
        if config.threshold_kind == "event_probability" and any(
            value < 0.0 or value > 1.0 for value in thresholds
        ):
            raise ValueError("threshold candidates are invalid")
    elif config.threshold_kind is not None:
        raise _config_error()
    if config.operating_metric is not None and (
        not thresholds or config.operating_metric not in _OPERATING_METRICS
    ):
        raise _config_error()
    if (
        type(config.gain_fractions) is not tuple
        or not 1 <= len(config.gain_fractions) <= 100
    ):
        raise _config_error()
    requested_fractions: list[float] = []
    for value in config.gain_fractions:
        if (
            not _is_real(value)
            or not np.isfinite(float(value))
            or not 0.0 < float(value) <= 1.0
        ):
            raise _config_error()
        requested_fractions.append(float(value))
    fractions = tuple(sorted(set(requested_fractions)))
    if 1.0 not in fractions:
        raise _config_error()
    duplicate_fractions = len(fractions) != len(requested_fractions)
    if config.loss_fraction is not None and type(config.loss_fraction) is not str:
        if not _is_real(config.loss_fraction) or not np.isfinite(
            float(config.loss_fraction)
        ):
            raise ValueError("binary risk business inputs are invalid")
        if not 0.0 <= float(config.loss_fraction) <= 1.0:
            raise ValueError("binary risk business inputs are invalid")
    if type(config.loss_fraction) is str and not config.loss_fraction:
        raise ValueError("binary risk business inputs are invalid")
    if config.n_splits is not None and (
        type(config.n_splits) is not int or not 2 <= config.n_splits <= 20
    ):
        raise _config_error()
    if config.test_size is not None and (
        not _is_real(config.test_size)
        or not np.isfinite(float(config.test_size))
        or not 0.0 < float(config.test_size) < 1.0
    ):
        raise _config_error()
    if type(config.fold_cutoffs) is not tuple:
        raise _config_error()
    cutoffs = tuple(_normalise_datetime(value) for value in config.fold_cutoffs)
    if any(left >= right for left, right in zip(cutoffs, cutoffs[1:])):
        raise _config_error()
    validation_end = (
        None
        if config.validation_end is None
        else _normalise_datetime(config.validation_end)
    )
    analysis_as_of = (
        None
        if config.analysis_as_of is None
        else _normalise_datetime(config.analysis_as_of)
    )
    snapshot = BinaryRiskValidationConfig(
        validation_mode=config.validation_mode,
        n_splits=config.n_splits,
        test_size=None if config.test_size is None else float(config.test_size),
        random_state=int(config.random_state),
        group_column=names["group_column"],
        observation_time_column=names["observation_time_column"],
        event_time_column=names["event_time_column"],
        outcome_end_time_column=names["outcome_end_time_column"],
        label_available_time_column=names["label_available_time_column"],
        maturity_source=config.maturity_source,
        prediction_horizon=horizon,
        prediction_horizon_column=names["prediction_horizon_column"],
        reporting_delay=delay,
        fold_cutoffs=cutoffs,
        validation_end=validation_end,
        analysis_as_of=analysis_as_of,
        thresholds=tuple(requested_thresholds),
        threshold_kind=config.threshold_kind,
        operating_metric=config.operating_metric,
        calibration_bins=int(config.calibration_bins),
        gain_fractions=tuple(requested_fractions),
        exposure_column=names["exposure_column"],
        observed_loss_column=names["observed_loss_column"],
        observed_loss_available_time_column=names[
            "observed_loss_available_time_column"
        ],
        observed_loss_is_mature_snapshot=config.observed_loss_is_mature_snapshot,
        loss_fraction=(
            float(config.loss_fraction)
            if _is_real(config.loss_fraction)
            else config.loss_fraction
        ),
        exposure_unit=names["exposure_unit"],
    )
    return snapshot, thresholds, fractions, duplicate_thresholds, duplicate_fractions


def _target_labels(
    series: pd.Series, positive_label: RiskLabelInput | None
) -> tuple[
    tuple[RiskLabel | None, ...],
    tuple[RiskLabel, RiskLabel],
    RiskLabel,
    tuple[int, ...],
]:
    labels: list[RiskLabel | None] = []
    missing_positions: list[int] = []
    kinds: set[type[object]] = set()
    classes: list[RiskLabel] = []
    seen: set[tuple[type[object], object]] = set()
    for position, value in enumerate(series.tolist()):
        try:
            missing = pd.isna(value)
        except Exception as error:
            raise ValueError(
                "binary target labels must be homogeneous string, integer, "
                "or boolean values"
            ) from error
        if not isinstance(missing, (bool, np.bool_)):
            raise ValueError(
                "binary target labels must be homogeneous string, integer, "
                "or boolean values"
            )
        if bool(missing):
            labels.append(None)
            missing_positions.append(position)
            continue
        label = _normalise_label(value)
        labels.append(label)
        kinds.add(type(label))
        key = _label_key(label)
        if key not in seen:
            seen.add(key)
            classes.append(label)
    if len(missing_positions) == len(series):
        raise ValueError("binary target has no non-missing labels")
    if len(kinds) != 1:
        raise ValueError(
            "binary target labels must be homogeneous string, integer, "
            "or boolean values"
        )
    if len(classes) != 2:
        raise ValueError("binary target must contain exactly two non-missing classes")
    if positive_label is None:
        keys = {_label_key(value) for value in classes}
        if keys == {_label_key(False), _label_key(True)}:
            positive = True
        elif keys == {_label_key(0), _label_key(1)}:
            positive = 1
        else:
            raise ValueError(
                "positive_label must be provided for non-canonical binary labels"
            )
    else:
        try:
            positive = _normalise_label(positive_label)
        except ValueError as error:
            raise ValueError(
                "positive_label is not present in binary target"
            ) from error
        if _label_key(positive) not in {_label_key(value) for value in classes}:
            raise ValueError("positive_label is not present in binary target")
    return tuple(labels), (classes[0], classes[1]), positive, tuple(missing_positions)


def _datetime_series(
    df: pd.DataFrame, name: str, positions: tuple[int, ...]
) -> pd.Series:
    series = df[name]
    if not is_datetime64_any_dtype(series.dtype) or isinstance(
        series.dtype, pd.DatetimeTZDtype
    ):
        raise ValueError("time validation metadata is invalid")
    selected = series.iloc[list(positions)].reset_index(drop=True)
    if selected.isna().any():
        raise ValueError("time validation metadata is invalid")
    return selected


def _time_metadata(
    df: pd.DataFrame,
    config: BinaryRiskValidationConfig,
    eligible: tuple[int, ...],
    labels: tuple[RiskLabel | None, ...],
    positive: RiskLabel,
) -> tuple[pd.Series, pd.Series, str, str, object, object]:
    if (
        config.observation_time_column is None
        or config.maturity_source
        not in (
            "label_available_time",
            "observation_horizon",
            "outcome_end",
        )
        or not config.fold_cutoffs
        or config.validation_end is None
        or config.analysis_as_of is None
    ):
        raise ValueError("label maturity metadata is missing or inconsistent")
    observation = _datetime_series(df, config.observation_time_column, eligible)
    if (
        config.validation_end <= config.fold_cutoffs[-1]
        or config.analysis_as_of < config.validation_end
    ):
        raise ValueError("time validation metadata is invalid")
    has_horizon = (
        config.prediction_horizon is not None
        or config.prediction_horizon_column is not None
    )
    if (
        config.prediction_horizon is not None
        and config.prediction_horizon_column is not None
    ):
        raise ValueError("label maturity metadata is missing or inconsistent")
    if config.outcome_end_time_column is None and not has_horizon:
        raise ValueError("label maturity metadata is missing or inconsistent")
    if config.prediction_horizon_column is not None:
        horizon_series = df[config.prediction_horizon_column]
        if not pd.api.types.is_timedelta64_dtype(horizon_series.dtype):
            raise ValueError("time validation metadata is invalid")
        horizons = horizon_series.iloc[list(eligible)].reset_index(drop=True)
        if horizons.isna().any() or (horizons <= pd.Timedelta(0)).any():
            raise ValueError("time validation metadata is invalid")
        horizon_value: object = pd.NA
        horizon_column: object = config.prediction_horizon_column
    elif config.prediction_horizon is not None:
        horizons = pd.Series(
            [pd.Timedelta(config.prediction_horizon)] * len(eligible),
            index=observation.index,
        )
        horizon_value = config.prediction_horizon
        horizon_column = pd.NA
    else:
        horizons = None
        horizon_value = pd.NA
        horizon_column = pd.NA
    try:
        derived_outcome = None if horizons is None else observation + horizons
    except (ArithmeticError, OverflowError, ValueError) as error:
        raise ValueError("time validation metadata is invalid") from error
    if config.outcome_end_time_column is not None:
        outcome = _datetime_series(df, config.outcome_end_time_column, eligible)
        outcome_source = "column"
        if derived_outcome is not None and not bool((outcome == derived_outcome).all()):
            raise ValueError("label maturity metadata is missing or inconsistent")
    else:
        if derived_outcome is None:
            raise ValueError("label maturity metadata is missing or inconsistent")
        outcome = derived_outcome
        outcome_source = "derived_horizon"
    if not (outcome > observation).all():
        raise ValueError("time validation metadata is invalid")
    try:
        derived_available = outcome + pd.Timedelta(config.reporting_delay)
    except (ArithmeticError, OverflowError, ValueError) as error:
        raise ValueError("time validation metadata is invalid") from error
    if config.maturity_source == "label_available_time":
        if config.label_available_time_column is None:
            raise ValueError("label maturity metadata is missing or inconsistent")
        available = _datetime_series(df, config.label_available_time_column, eligible)
        if not (available >= derived_available).all():
            raise ValueError("label maturity metadata is missing or inconsistent")
        delay_source = "config_minimum_check"
    else:
        if config.maturity_source == "observation_horizon" and horizons is None:
            raise ValueError("label maturity metadata is missing or inconsistent")
        if (
            config.maturity_source == "outcome_end"
            and config.outcome_end_time_column is None
        ):
            raise ValueError("label maturity metadata is missing or inconsistent")
        available = derived_available
        delay_source = "config_derivation"
        if config.label_available_time_column is not None:
            explicit = _datetime_series(
                df, config.label_available_time_column, eligible
            )
            if not bool((explicit == available).all()):
                raise ValueError("label maturity metadata is missing or inconsistent")
    if config.event_time_column is not None:
        events = df[config.event_time_column]
        if not is_datetime64_any_dtype(events.dtype) or isinstance(
            events.dtype, pd.DatetimeTZDtype
        ):
            raise ValueError("time validation metadata is invalid")
        for local, position in enumerate(eligible):
            value = events.iloc[position]
            is_positive = _label_key(labels[position]) == _label_key(positive)
            if pd.isna(value):
                continue
            if (
                not is_positive
                or not observation.iloc[local] < value <= outcome.iloc[local]
            ):
                raise ValueError("time validation metadata is invalid")
            if available.iloc[local] < value:
                raise ValueError("time validation metadata is invalid")
    return (
        observation,
        available,
        outcome_source,
        delay_source,
        horizon_value,
        horizon_column,
    )


def _plans(
    df: pd.DataFrame,
    config: BinaryRiskValidationConfig,
    eligible: tuple[int, ...],
    labels: tuple[RiskLabel | None, ...],
    positive: RiskLabel,
) -> tuple[list[_FoldPlan], dict[str, object]]:
    y = np.asarray(
        [
            int(_label_key(labels[position]) == _label_key(positive))
            for position in eligible
        ]
    )
    mode = config.validation_mode
    metadata: dict[str, object] = {}
    plans: list[_FoldPlan] = []
    if mode == "stratified_holdout":
        if config.test_size is None or config.n_splits is not None:
            raise _config_error()
        try:
            train, validation = train_test_split(
                np.asarray(eligible),
                stratify=y,
                test_size=config.test_size,
                random_state=config.random_state,
            )
        except ValueError as error:
            raise ValueError("binary risk validation split is infeasible") from error
        train_tuple = tuple(sorted(int(value) for value in train))
        val_tuple = tuple(sorted(int(value) for value in validation))
        plans.append(
            _FoldPlan(0, train_tuple, val_tuple, val_tuple, len(train_tuple), 0)
        )
    elif mode == "stratified_kfold":
        if config.n_splits is None or config.test_size is not None:
            raise _config_error()
        try:
            splits = StratifiedKFold(
                n_splits=config.n_splits, shuffle=True, random_state=config.random_state
            ).split(np.asarray(eligible), y)
            for fold_id, (train_index, validation_index) in enumerate(splits):
                train_tuple = tuple(
                    sorted(eligible[int(index)] for index in train_index)
                )
                val_tuple = tuple(
                    sorted(eligible[int(index)] for index in validation_index)
                )
                plans.append(
                    _FoldPlan(
                        fold_id, train_tuple, val_tuple, val_tuple, len(train_tuple), 0
                    )
                )
        except ValueError as error:
            raise ValueError("binary risk validation split is infeasible") from error
    elif mode in ("group_holdout", "group_kfold"):
        if config.group_column is None:
            raise _config_error()
        groups = df[config.group_column].iloc[list(eligible)].tolist()
        keys: list[tuple[type[object], object]] = []
        codes: list[int] = []
        mapping: dict[tuple[type[object], object], int] = {}
        for value in groups:
            if isinstance(value, np.generic):
                value = value.item()
            if type(value) not in (str, int, bool) or pd.isna(value):
                raise ValueError("binary risk validation split is infeasible")
            key = (type(value), value)
            if key not in mapping:
                mapping[key] = len(keys)
                keys.append(key)
            codes.append(mapping[key])
        if len(keys) < 2:
            raise ValueError("binary risk validation split is infeasible")
        if mode == "group_holdout":
            if config.test_size is None or config.n_splits is not None:
                raise _config_error()
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=config.test_size, random_state=config.random_state
            )
            split_iter = splitter.split(np.asarray(eligible), y, np.asarray(codes))
        else:
            if (
                config.n_splits is None
                or config.test_size is not None
                or config.n_splits > len(keys)
            ):
                raise _config_error()
            splitter = StratifiedGroupKFold(
                n_splits=config.n_splits, shuffle=True, random_state=config.random_state
            )
            split_iter = splitter.split(np.asarray(eligible), y, np.asarray(codes))
        try:
            for fold_id, (train_index, validation_index) in enumerate(split_iter):
                train_tuple = tuple(
                    sorted(eligible[int(index)] for index in train_index)
                )
                val_tuple = tuple(
                    sorted(eligible[int(index)] for index in validation_index)
                )
                train_groups = {codes[int(index)] for index in train_index}
                validation_groups = {codes[int(index)] for index in validation_index}
                if train_groups & validation_groups:
                    raise ValueError(f"validation fold {fold_id} has group overlap")
                plans.append(
                    _FoldPlan(
                        fold_id, train_tuple, val_tuple, val_tuple, len(train_tuple), 0
                    )
                )
        except ValueError as error:
            if str(error).endswith("has group overlap"):
                raise
            raise ValueError("binary risk validation split is infeasible") from error
    else:
        if (
            config.group_column is not None
            or config.n_splits is not None
            or config.test_size is not None
        ):
            raise _config_error()
        (
            observation,
            available,
            outcome_source,
            delay_source,
            horizon_value,
            horizon_column,
        ) = _time_metadata(df, config, eligible, labels, positive)
        if mode == "time_holdout" and len(config.fold_cutoffs) != 1:
            raise _config_error()
        if mode == "time_forward" and not 2 <= len(config.fold_cutoffs) <= 20:
            raise _config_error()
        position_to_local = {position: index for index, position in enumerate(eligible)}
        ends = (*config.fold_cutoffs[1:], config.validation_end)
        for fold_id, (cutoff, end) in enumerate(
            zip(config.fold_cutoffs, ends, strict=True)
        ):
            before = tuple(
                position
                for position in eligible
                if observation.iloc[position_to_local[position]] < cutoff
            )
            train = tuple(
                position
                for position in before
                if available.iloc[position_to_local[position]] <= cutoff
            )
            validation = tuple(
                position
                for position in eligible
                if cutoff <= observation.iloc[position_to_local[position]] < end
            )
            if not validation:
                raise ValueError(f"validation fold {fold_id} has no validation rows")
            evaluable = tuple(
                position
                for position in validation
                if available.iloc[position_to_local[position]] <= config.analysis_as_of
            )
            plans.append(
                _FoldPlan(
                    fold_id,
                    tuple(sorted(train)),
                    tuple(sorted(validation)),
                    tuple(sorted(evaluable)),
                    len(before),
                    len(before) - len(train),
                    cutoff,
                    cutoff,
                    end,
                )
            )
        metadata = {
            "outcome_end_source": outcome_source,
            "prediction_horizon_source": (
                "column"
                if config.prediction_horizon_column is not None
                else "scalar"
                if config.prediction_horizon is not None
                else "not_provided"
            ),
            "prediction_horizon_value": horizon_value,
            "prediction_horizon_column": horizon_column,
            "reporting_delay_source": delay_source,
        }
    return plans, metadata


def _validate_columns(df: pd.DataFrame, config: BinaryRiskValidationConfig) -> None:
    for role, name in (
        ("group", config.group_column),
        ("observation_time", config.observation_time_column),
        ("event_time", config.event_time_column),
        ("outcome_end_time", config.outcome_end_time_column),
        ("label_available_time", config.label_available_time_column),
        ("prediction_horizon", config.prediction_horizon_column),
        ("exposure", config.exposure_column),
        ("observed_loss", config.observed_loss_column),
        ("observed_loss_available_time", config.observed_loss_available_time_column),
    ):
        if name is not None and name not in df.columns:
            raise ValueError(f"{role} column not found: {name!r}")
    if isinstance(config.loss_fraction, str) and config.loss_fraction not in df.columns:
        raise ValueError(f"loss_fraction column not found: {config.loss_fraction!r}")


def _validate_mode_fields(config: BinaryRiskValidationConfig) -> None:
    time_mode = config.validation_mode in ("time_holdout", "time_forward")
    time_values = (
        config.observation_time_column,
        config.event_time_column,
        config.outcome_end_time_column,
        config.label_available_time_column,
        config.maturity_source,
        config.prediction_horizon,
        config.prediction_horizon_column,
        config.fold_cutoffs,
        config.validation_end,
    )
    if time_mode:
        if config.group_column is not None:
            raise _config_error()
    elif any(
        value not in (None, ()) for value in time_values
    ) or config.reporting_delay != timedelta(0):
        raise _config_error()
    if config.validation_mode.startswith("group"):
        if config.group_column is None:
            raise _config_error()
    elif not time_mode and config.group_column is not None:
        raise _config_error()
    if (
        not time_mode
        and config.analysis_as_of is not None
        and config.observed_loss_column is None
    ):
        raise _config_error()


def _name_sequence(value: object, *, name: str, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise _config_error()
    names = tuple(value)
    if (
        (not allow_empty and not names)
        or not all(type(column) is str for column in names)
        or len(set(names)) != len(names)
    ):
        raise _config_error()
    return names


def _external_arrays(
    external: ExternalRiskPredictions,
    plans: list[_FoldPlan],
    positive: RiskLabel,
) -> tuple[np.ndarray, np.ndarray | None, str, str | None]:
    if type(external) is not ExternalRiskPredictions:
        raise ValueError("external risk predictions have invalid schema")
    expected_positions = tuple(
        sorted(position for plan in plans for position in plan.validation_positions)
    )
    expected_fold_by_position = {
        position: plan.fold_id
        for plan in plans
        for position in plan.validation_positions
    }
    if (
        type(external.row_positions) is not tuple
        or any(type(value) is not int or value < 0 for value in external.row_positions)
        or tuple(sorted(set(external.row_positions))) != external.row_positions
        or external.row_positions != expected_positions
        or type(external.fold_ids) is not tuple
        or external.fold_ids
        != tuple(expected_fold_by_position[position] for position in expected_positions)
        or type(external.fold_fit_row_positions) is not tuple
        or external.fold_fit_row_positions
        != tuple((plan.fold_id, plan.train_positions) for plan in plans)
    ):
        raise ValueError(
            "external prediction provenance does not match validation plan"
        )
    n = len(expected_positions)
    ranking: np.ndarray | None = None
    probability: np.ndarray | None = None
    if external.ranking_scores is not None:
        if (
            type(external.ranking_scores) is not tuple
            or len(external.ranking_scores) != n
            or type(external.ranking_direction) is not str
        ):
            raise ValueError("external risk predictions have invalid schema")
        if external.ranking_direction not in (
            "higher_risk",
            "lower_risk",
        ):
            raise ValueError(
                "ranking scores must be finite real values with explicit direction"
            )
        ranking = _risk_float_array(external.ranking_scores, probability=False)
        if external.ranking_direction == "lower_risk":
            ranking = -ranking
    elif external.ranking_direction is not None:
        raise ValueError("external risk predictions have invalid schema")
    if external.event_probabilities is not None:
        if (
            type(external.event_probabilities) is not tuple
            or len(external.event_probabilities) != n
        ):
            raise ValueError("external risk predictions have invalid schema")
        probability = _risk_float_array(external.event_probabilities, probability=True)
        try:
            mapped = _normalise_label(external.probability_positive_label)
        except ValueError as error:
            raise ValueError(
                "event probability positive-label mapping is invalid"
            ) from error
        if _label_key(mapped) != _label_key(positive):
            raise ValueError("event probability positive-label mapping is invalid")
        if type(external.probability_provenance) is not str or (
            external.probability_provenance
            not in (
                "predict_proba",
                "fold_safe_calibrated",
                "external_declared",
            )
        ):
            raise ValueError("external risk predictions have invalid schema")
    elif (
        external.probability_positive_label is not None
        or external.probability_provenance is not None
    ):
        raise ValueError("external risk predictions have invalid schema")
    if ranking is None and probability is None:
        raise ValueError("external risk predictions have invalid schema")
    if ranking is None:
        ranking = probability.copy()
        source = "external_event_probability"
    elif probability is None:
        source = "external_ranking_score"
    else:
        source = "external_ranking_and_probability"
    return ranking, probability, source, external.probability_provenance


def _numeric_values(
    df: pd.DataFrame, column: str, positions: tuple[int, ...]
) -> np.ndarray:
    series = df[column]
    if (
        not is_numeric_dtype(series.dtype)
        or is_bool_dtype(series.dtype)
        or is_complex_dtype(series.dtype)
    ):
        raise ValueError("binary risk business inputs are invalid")
    try:
        values = series.iloc[list(positions)].astype("float64").to_numpy(copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("binary risk business inputs are invalid") from error
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("binary risk business inputs are invalid")
    return values


def _validate_business_declaration(
    df: pd.DataFrame, config: BinaryRiskValidationConfig
) -> None:
    any_business = any(
        value is not None
        for value in (
            config.exposure_column,
            config.observed_loss_column,
            config.loss_fraction,
        )
    )
    if any_business != (config.exposure_unit is not None):
        raise ValueError("binary risk business inputs are invalid")
    if config.exposure_column is not None:
        series = df[config.exposure_column]
        if (
            not is_numeric_dtype(series.dtype)
            or is_bool_dtype(series.dtype)
            or is_complex_dtype(series.dtype)
        ):
            raise ValueError("binary risk business inputs are invalid")
    if config.observed_loss_column is None:
        if (
            config.observed_loss_available_time_column is not None
            or config.observed_loss_is_mature_snapshot
        ):
            raise ValueError("binary risk business inputs are invalid")
    else:
        availability_mode = config.observed_loss_available_time_column is not None
        if config.analysis_as_of is None or (
            availability_mode == config.observed_loss_is_mature_snapshot
        ):
            raise ValueError("binary risk business inputs are invalid")
        series = df[config.observed_loss_column]
        if (
            not is_numeric_dtype(series.dtype)
            or is_bool_dtype(series.dtype)
            or is_complex_dtype(series.dtype)
        ):
            raise ValueError("binary risk business inputs are invalid")
        if availability_mode:
            availability = df[str(config.observed_loss_available_time_column)]
            if not is_datetime64_any_dtype(availability.dtype) or isinstance(
                availability.dtype, pd.DatetimeTZDtype
            ):
                raise ValueError("binary risk business inputs are invalid")
    if isinstance(config.loss_fraction, str):
        series = df[config.loss_fraction]
        if (
            not is_numeric_dtype(series.dtype)
            or is_bool_dtype(series.dtype)
            or is_complex_dtype(series.dtype)
        ):
            raise ValueError("binary risk business inputs are invalid")


def _business_inputs(
    df: pd.DataFrame,
    config: BinaryRiskValidationConfig,
    positions: tuple[int, ...],
    has_probability: bool,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
    np.ndarray | None,
    str,
    datetime | None,
    int,
    int,
]:
    any_business = any(
        value is not None
        for value in (
            config.exposure_column,
            config.observed_loss_column,
            config.loss_fraction,
        )
    )
    if any_business and config.exposure_unit is None:
        raise ValueError("binary risk business inputs are invalid")
    if not any_business and config.exposure_unit is not None:
        raise ValueError("binary risk business inputs are invalid")
    exposure = (
        None
        if config.exposure_column is None
        else _numeric_values(df, config.exposure_column, positions)
    )
    if config.observed_loss_column is None:
        if (
            config.observed_loss_available_time_column is not None
            or config.observed_loss_is_mature_snapshot
        ):
            raise ValueError("binary risk business inputs are invalid")
        observed = None
        mature = np.zeros(len(positions), dtype=bool)
        mode = "not_provided"
        as_of = None
        mature_n = excluded_n = 0
    else:
        if config.analysis_as_of is None:
            raise ValueError("binary risk business inputs are invalid")
        availability_mode = config.observed_loss_available_time_column is not None
        snapshot_mode = config.observed_loss_is_mature_snapshot
        if availability_mode == snapshot_mode:
            raise ValueError("binary risk business inputs are invalid")
        if availability_mode:
            try:
                available = _datetime_series(
                    df, str(config.observed_loss_available_time_column), positions
                )
            except ValueError as error:
                raise ValueError("binary risk business inputs are invalid") from error
            mature = (available <= config.analysis_as_of).to_numpy(dtype=bool)
            mode = "availability_column"
        else:
            mature = np.ones(len(positions), dtype=bool)
            mode = "mature_snapshot"
        observed = np.full(len(positions), np.nan, dtype="float64")
        mature_indices = np.flatnonzero(mature)
        if len(mature_indices):
            mature_positions = tuple(positions[int(index)] for index in mature_indices)
            observed_values = _numeric_values(
                df, config.observed_loss_column, mature_positions
            )
            observed[mature_indices] = observed_values
        as_of = config.analysis_as_of
        mature_n = int(mature.sum())
        excluded_n = len(positions) - mature_n
    if config.loss_fraction is None:
        loss_fraction = None
    elif isinstance(config.loss_fraction, str):
        loss_fraction = _numeric_values(df, config.loss_fraction, positions)
        if (loss_fraction > 1.0).any():
            raise ValueError("binary risk business inputs are invalid")
    else:
        loss_fraction = np.full(
            len(positions), float(config.loss_fraction), dtype="float64"
        )
    if loss_fraction is not None and not has_probability:
        # The assumption is retained for audit, but expected loss stays unavailable.
        pass
    return exposure, observed, mature, loss_fraction, mode, as_of, mature_n, excluded_n


def _operating_point(
    threshold_analysis: pd.DataFrame,
    config: BinaryRiskValidationConfig,
    thresholds: tuple[float, ...],
) -> pd.DataFrame:
    if not thresholds or config.operating_metric is None:
        return _result_frame(
            [],
            _OPERATING_COLUMNS,
            integers=("candidate_count",),
            floats=("threshold", "metric_value"),
        )
    overall = threshold_analysis.loc[threshold_analysis["scope"] == "overall"]
    metric = config.operating_metric
    available = overall.loc[overall[f"{metric}_status"] == "available"]
    if available.empty:
        row = {
            "threshold_kind": config.threshold_kind,
            "operating_metric": metric,
            "threshold": pd.NA,
            "metric_value": pd.NA,
            "candidate_count": len(thresholds),
            "status": "undefined",
            "reason": "objective_undefined",
        }
    else:
        best_value = float(available[metric].max())
        best = available.loc[available[metric] == best_value].iloc[-1]
        row = {
            "threshold_kind": config.threshold_kind,
            "operating_metric": metric,
            "threshold": float(best["threshold"]),
            "metric_value": best_value,
            "candidate_count": len(thresholds),
            "status": "available",
            "reason": pd.NA,
        }
    return _result_frame(
        [row],
        _OPERATING_COLUMNS,
        integers=("candidate_count",),
        floats=("threshold", "metric_value"),
    )


def _business_table(
    *,
    y: np.ndarray,
    evaluable: np.ndarray,
    scores: np.ndarray,
    probabilities: np.ndarray | None,
    exposure: np.ndarray | None,
    observed_loss: np.ndarray | None,
    observed_loss_mature: np.ndarray,
    loss_fraction: np.ndarray | None,
    fractions: tuple[float, ...],
    thresholds: tuple[float, ...],
    threshold_kind: str | None,
    unit: str | None,
) -> pd.DataFrame:
    n = len(scores)
    segments: list[tuple[str, object, np.ndarray]] = [
        ("all", pd.NA, np.ones(n, dtype=bool))
    ]
    for fraction in fractions:
        target_count = max(1, int(np.ceil(fraction * n)))
        boundary = float(np.sort(scores)[::-1][target_count - 1])
        segments.append(("top_fraction", fraction, scores >= boundary))
    threshold_values = (
        probabilities if threshold_kind == "event_probability" else scores
    )
    for threshold in thresholds:
        if threshold_values is None:
            raise ValueError("event-probability thresholds require event probabilities")
        segments.append(
            ("threshold_selected", threshold, threshold_values >= threshold)
        )
    rows: list[dict[str, object]] = []
    metric_order = (
        "event_rate",
        "predicted_positive_rate",
        "exposure_sum",
        "observed_loss_sum",
        "expected_loss_sum",
    )
    for segment_kind, segment_value, selected in segments:
        primitive = _binary_risk_business_primitive(
            y=y,
            evaluable=evaluable,
            selected=selected,
            probabilities=probabilities,
            exposure=exposure,
            observed_loss=observed_loss,
            observed_loss_mature=observed_loss_mature,
            loss_fraction=loss_fraction,
        )
        if segment_kind == "threshold_selected":
            predicted_positive = (float(selected.sum() / n), "available", pd.NA)
        else:
            predicted_positive = (pd.NA, "unavailable", "not_threshold_segment")
        primitive["predicted_positive_rate"] = predicted_positive
        for metric in metric_order:
            value, status, reason = primitive[metric]
            rows.append(
                {
                    "segment_kind": segment_kind,
                    "segment_value": segment_value,
                    "metric": metric,
                    "value": value,
                    "status": status,
                    "reason": reason,
                    "n_rows": int(selected.sum()),
                    "n_evaluable_rows": int((selected & evaluable).sum()),
                    "n_observed_loss_mature_rows": int(
                        (selected & observed_loss_mature).sum()
                    ),
                    "unit": unit
                    if metric
                    in ("exposure_sum", "observed_loss_sum", "expected_loss_sum")
                    else pd.NA,
                }
            )
    return _result_frame(
        rows,
        _BUSINESS_COLUMNS,
        integers=(
            "n_rows",
            "n_evaluable_rows",
            "n_observed_loss_mature_rows",
        ),
        floats=("segment_value", "value"),
    )


def _validate_binary_risk_validation_result(result: BinaryRiskValidationResult) -> None:
    if type(result) is not BinaryRiskValidationResult:
        raise ValueError("result must be a BinaryRiskValidationResult")
    schemas = (
        (result.folds, _FOLD_COLUMNS),
        (result.predictions, _PREDICTION_COLUMNS),
        (result.excluded_rows, _EXCLUDED_COLUMNS),
        (result.metrics, _RISK_METRIC_COLUMNS),
        (result.gains, _RISK_GAINS_COLUMNS),
        (result.calibration, _RISK_CALIBRATION_COLUMNS),
        (result.threshold_analysis, _RISK_THRESHOLD_COLUMNS),
        (result.operating_point, _OPERATING_COLUMNS),
        (result.business_metrics, _BUSINESS_COLUMNS),
    )
    if any(
        not isinstance(table, pd.DataFrame) or tuple(table.columns) != columns
        for table, columns in schemas
    ):
        raise ValueError("binary risk validation result has invalid schema")
    if (
        result.predicted_n_rows != len(result.predictions)
        or result.evaluable_n_rows != int(result.predictions["is_evaluable"].sum())
        or tuple(result.predictions["row_position"])
        != tuple(sorted(result.predictions["row_position"]))
        or not isinstance(result.warnings, tuple)
        or not isinstance(result.limitations, tuple)
    ):
        raise ValueError("binary risk validation result has invalid schema")


def validate_binary_risk(
    df: pd.DataFrame,
    target: str,
    *,
    positive_label: RiskLabelInput | None = None,
    config: BinaryRiskValidationConfig,
    estimator: ClassifierMixin | None = None,
    external_predictions: ExternalRiskPredictions | None = None,
    features: Sequence[str] | None = None,
    exclude_columns: Sequence[str] = (),
) -> BinaryRiskValidationResult:
    """Validate binary risk predictions on leakage-aware folds.

    Parameters
    ----------
    df, target
        Source frame and binary target. The frame is never modified.
    positive_label, config
        Explicit event label when non-canonical and the frozen validation plan.
    estimator, external_predictions
        Exactly one prediction source must be supplied.
    features, exclude_columns
        Estimator-only feature controls; external predictions require defaults.

    Returns
    -------
    BinaryRiskValidationResult
        Independent fold, prediction, metric, calibration, threshold, and loss tables.

    Raises
    ------
    ValueError
        If inputs, provenance, folds, predictions, or business values are invalid.

    Notes
    -----
    Missing targets are excluded with provenance. Time validation excludes immature
    outcomes from metrics. No caller object is modified and no files are written.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sharper import BinaryRiskValidationConfig
    >>> from sharper import validate_binary_risk
    >>> frame = pd.DataFrame(
    ...     {"feature": [0, 1, 0, 1, 2, 0, 2, 1], "target": [0, 1] * 4}
    ... )
    >>> config = BinaryRiskValidationConfig(
    ...     validation_mode="stratified_kfold", n_splits=2
    ... )
    >>> result = validate_binary_risk(
    ...     frame,
    ...     "target",
    ...     config=config,
    ...     estimator=LogisticRegression(),
    ...     features=("feature",),
    ... )
    >>> result.prediction_scope
    'oof'
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if type(config) is not BinaryRiskValidationConfig:
        raise _config_error()
    if (
        external_predictions is not None
        and type(external_predictions) is not ExternalRiskPredictions
    ):
        raise ValueError("external risk predictions have invalid schema")
    if (estimator is None) == (external_predictions is None):
        raise ValueError(
            "exactly one of estimator and external_predictions must be provided"
        )
    snapshot, thresholds, fractions, duplicate_thresholds, duplicate_fractions = (
        _normalise_config(config)
    )
    _validate_mode_fields(snapshot)
    if external_predictions is not None and (
        features is not None or exclude_columns != ()
    ):
        raise _config_error()
    if (
        not all(type(column) is str for column in df.columns)
        or df.columns.has_duplicates
    ):
        raise ValueError("DataFrame column names must be unique strings")
    if type(target) is not str or target not in df.columns:
        raise ValueError(f"target column not found: {target!r}")
    labels, classes, positive, missing_positions = _target_labels(
        df[target], positive_label
    )
    _validate_columns(df, snapshot)
    _validate_business_declaration(df, snapshot)
    eligible = tuple(
        position for position, label in enumerate(labels) if label is not None
    )
    plans, time_metadata = _plans(df, snapshot, eligible, labels, positive)
    predicted_positions = tuple(
        sorted(position for plan in plans for position in plan.validation_positions)
    )
    source_has_probability = (
        external_predictions is not None
        and external_predictions.event_probabilities is not None
    ) or (estimator is not None and callable(getattr(estimator, "predict_proba", None)))
    business_inputs = _business_inputs(
        df, snapshot, predicted_positions, has_probability=source_has_probability
    )

    if external_predictions is not None:
        ranking, probabilities, score_source, probability_provenance = _external_arrays(
            external_predictions, plans, positive
        )
        feature_columns_by_fold = {plan.fold_id: () for plan in plans}
    else:
        requested_features = (
            None
            if features is None
            else _name_sequence(features, name="features", allow_empty=False)
        )
        exclusions = _name_sequence(
            exclude_columns, name="exclude_columns", allow_empty=True
        )
        role_columns = {
            target,
            *(
                name
                for name in (
                    snapshot.group_column,
                    snapshot.observation_time_column,
                    snapshot.event_time_column,
                    snapshot.outcome_end_time_column,
                    snapshot.label_available_time_column,
                    snapshot.prediction_horizon_column,
                    snapshot.exposure_column,
                    snapshot.observed_loss_column,
                    snapshot.observed_loss_available_time_column,
                    snapshot.loss_fraction
                    if isinstance(snapshot.loss_fraction, str)
                    else None,
                )
                if name is not None
            ),
        }
        if any(name not in df.columns for name in exclusions):
            missing = next(name for name in exclusions if name not in df.columns)
            raise ValueError(f"excluded column not found: {missing!r}")
        if requested_features is not None:
            if any(name not in df.columns for name in requested_features):
                missing = next(
                    name for name in requested_features if name not in df.columns
                )
                raise ValueError(f"feature column not found: {missing!r}")
            if set(requested_features) & (role_columns | set(exclusions)):
                raise ValueError(
                    "model features must not contain role or excluded columns"
                )
        model_columns = tuple(
            column
            for column in df.columns
            if column not in role_columns and column not in exclusions
        )
        candidates = model_columns if requested_features is None else requested_features
        fold_model_columns = (
            model_columns if requested_features is None else requested_features
        )
        if (
            not is_classifier(estimator)
            or not callable(getattr(estimator, "fit", None))
            or not (
                callable(getattr(estimator, "predict_proba", None))
                or callable(getattr(estimator, "decision_function", None))
            )
        ):
            raise ValueError(
                "estimator must be a cloneable binary classifier with a score interface"
            )
        try:
            clone(estimator)
        except Exception as error:
            raise ValueError("binary risk estimator could not be cloned") from error
        ranking_by_position: dict[int, float] = {}
        probability_by_position: dict[int, float] = {}
        kinds: set[str] = set()
        feature_columns_by_fold: dict[int, tuple[str, ...]] = {}
        label_values = tuple(
            value if value is not None else classes[0] for value in labels
        )
        for plan in plans:
            if not plan.train_positions:
                raise ValueError(
                    f"validation fold {plan.fold_id} has no eligible training rows"
                )
            train_keys = {
                _label_key(label_values[position]) for position in plan.train_positions
            }
            if len(train_keys) != 2:
                message = (
                    f"validation fold {plan.fold_id} training target must contain "
                    "both classes"
                )
                raise ValueError(message)
            prediction_error = (
                f"binary risk estimator prediction failed in fold {plan.fold_id}"
            )
            output_error = (
                f"binary risk estimator has invalid output in fold {plan.fold_id}"
            )
            fold = _fit_classifier_fold(
                df,
                labels=label_values,
                classes=classes,
                positive_label=positive,
                model_columns=fold_model_columns,
                candidates=candidates,
                requested_features=requested_features,
                train_positions=plan.train_positions,
                validation_positions=plan.validation_positions,
                estimator_source=estimator,
                clone_source=True,
                require_score=True,
                clone_error_message="binary risk estimator could not be cloned",
                fit_error_message=(
                    f"binary risk estimator fit failed in fold {plan.fold_id}"
                ),
                prediction_error_message=prediction_error,
                output_error_message=output_error,
            )
            feature_columns_by_fold[plan.fold_id] = fold.feature_columns
            kinds.add(str(fold.score_kind))
            for position, value in zip(
                plan.validation_positions, fold.ranking_scores or (), strict=True
            ):
                ranking_by_position[position] = value
            if fold.event_probabilities is not None:
                for position, value in zip(
                    plan.validation_positions, fold.event_probabilities, strict=True
                ):
                    probability_by_position[position] = value
        if len(kinds) != 1:
            raise ValueError("binary risk estimator has invalid output in fold 0")
        predicted_positions = tuple(sorted(ranking_by_position))
        ranking = np.asarray(
            [ranking_by_position[position] for position in predicted_positions],
            dtype="float64",
        )
        if probability_by_position:
            probabilities = np.asarray(
                [probability_by_position[position] for position in predicted_positions],
                dtype="float64",
            )
            probability_provenance = "predict_proba"
            score_source = "estimator_predict_proba"
        else:
            probabilities = None
            probability_provenance = None
            score_source = "estimator_decision_function"

    fold_by_position = {
        position: plan.fold_id
        for plan in plans
        for position in plan.validation_positions
    }
    evaluable_positions = {
        position for plan in plans for position in plan.evaluable_positions
    }
    y = np.asarray(
        [
            int(_label_key(labels[position]) == _label_key(positive))
            if position in evaluable_positions
            else 0
            for position in predicted_positions
        ],
        dtype="int8",
    )
    evaluable = np.asarray(
        [position in evaluable_positions for position in predicted_positions],
        dtype=bool,
    )
    fold_ids = np.asarray(
        [fold_by_position[position] for position in predicted_positions], dtype=int
    )
    if snapshot.threshold_kind == "event_probability" and probabilities is None:
        raise ValueError("event-probability thresholds require event probabilities")
    tables = _binary_risk_evaluation(
        target_values=y,
        evaluable=evaluable,
        ranking_scores=ranking,
        event_probabilities=probabilities,
        fold_ids=fold_ids,
        fold_order=tuple(plan.fold_id for plan in plans),
        calibration_bins=snapshot.calibration_bins,
        gain_fractions=fractions,
        thresholds=thresholds,
        threshold_kind=snapshot.threshold_kind,
    )

    prediction_rows = []
    for index, position in enumerate(predicted_positions):
        prediction_rows.append(
            {
                "row_position": position,
                "fold_id": int(fold_ids[index]),
                "target_value": labels[position] if evaluable[index] else pd.NA,
                "is_evaluable": bool(evaluable[index]),
                "unevaluable_reason": pd.NA
                if evaluable[index]
                else "immature_validation_outcome",
                "ranking_score": float(ranking[index]),
                "event_probability": (
                    pd.NA if probabilities is None else float(probabilities[index])
                ),
            }
        )
    predictions = _result_frame(
        prediction_rows,
        _PREDICTION_COLUMNS,
        integers=("row_position", "fold_id"),
        floats=("ranking_score", "event_probability"),
        booleans=("is_evaluable",),
        objects=("target_value",),
    )

    fold_rows = []
    for plan in plans:
        train_positive = sum(
            _label_key(labels[position]) == _label_key(positive)
            for position in plan.train_positions
        )
        evaluable_positive = sum(
            _label_key(labels[position]) == _label_key(positive)
            for position in plan.evaluable_positions
        )
        time_mode = snapshot.validation_mode in ("time_holdout", "time_forward")
        fold_rows.append(
            {
                "fold_id": plan.fold_id,
                "cutoff": plan.cutoff if time_mode else pd.NA,
                "validation_start": plan.validation_start if time_mode else pd.NA,
                "validation_end": plan.validation_end if time_mode else pd.NA,
                "analysis_as_of": snapshot.analysis_as_of if time_mode else pd.NA,
                "outcome_end_source": time_metadata.get(
                    "outcome_end_source", "not_applicable"
                ),
                "prediction_horizon_source": time_metadata.get(
                    "prediction_horizon_source", "not_applicable"
                ),
                "prediction_horizon_value": time_metadata.get(
                    "prediction_horizon_value", pd.NA
                ),
                "prediction_horizon_column": time_metadata.get(
                    "prediction_horizon_column", pd.NA
                ),
                "reporting_delay_source": time_metadata.get(
                    "reporting_delay_source", "not_applicable"
                ),
                "reporting_delay": snapshot.reporting_delay if time_mode else pd.NA,
                "train_row_positions": plan.train_positions,
                "validation_row_positions": plan.validation_positions,
                "evaluable_validation_row_positions": plan.evaluable_positions,
                "feature_columns": feature_columns_by_fold[plan.fold_id],
                "train_candidate_n": plan.train_candidate_n,
                "train_n": len(plan.train_positions),
                "train_positive_n": train_positive,
                "train_positive_rate": (
                    train_positive / len(plan.train_positions)
                    if plan.train_positions
                    else pd.NA
                ),
                "validation_n": len(plan.validation_positions),
                "validation_mature_n": len(plan.evaluable_positions)
                if time_mode
                else 0,
                "validation_excluded_n": (
                    len(plan.validation_positions) - len(plan.evaluable_positions)
                    if time_mode
                    else 0
                ),
                "evaluable_validation_n": len(plan.evaluable_positions),
                "evaluable_positive_n": evaluable_positive,
                "evaluable_positive_rate": (
                    evaluable_positive / len(plan.evaluable_positions)
                    if plan.evaluable_positions
                    else pd.NA
                ),
                "immature_train_n": plan.immature_train_n if time_mode else 0,
                "purged_train_n": plan.immature_train_n if time_mode else 0,
                "immature_validation_n": (
                    len(plan.validation_positions) - len(plan.evaluable_positions)
                    if time_mode
                    else 0
                ),
                "maturity_source": snapshot.maturity_source
                if time_mode
                else "not_applicable",
            }
        )
    folds = _result_frame(
        fold_rows,
        _FOLD_COLUMNS,
        integers=(
            "fold_id",
            "train_candidate_n",
            "train_n",
            "train_positive_n",
            "validation_n",
            "validation_mature_n",
            "validation_excluded_n",
            "evaluable_validation_n",
            "evaluable_positive_n",
            "immature_train_n",
            "purged_train_n",
            "immature_validation_n",
        ),
        floats=("train_positive_rate", "evaluable_positive_rate"),
    )

    excluded: dict[int, str] = {
        position: "missing_target" for position in missing_positions
    }
    predicted_set = set(predicted_positions)
    for position in eligible:
        if position in predicted_set:
            continue
        if snapshot.validation_mode.endswith(
            "holdout"
        ) and not snapshot.validation_mode.startswith("time"):
            excluded[position] = "training_only"
        elif snapshot.validation_mode.startswith("time"):
            observation = df[snapshot.observation_time_column].iloc[position]
            excluded[position] = (
                "before_first_validation_window"
                if observation < snapshot.fold_cutoffs[0]
                else "outside_validation_window"
            )
    excluded_rows = _result_frame(
        [
            {"row_position": position, "reason": reason}
            for position, reason in sorted(excluded.items())
        ],
        _EXCLUDED_COLUMNS,
        integers=("row_position",),
    )

    (
        exposure,
        observed,
        loss_mature,
        loss_fraction,
        loss_mode,
        loss_as_of,
        loss_mature_n,
        loss_excluded_n,
    ) = business_inputs
    business_metrics = _business_table(
        y=y,
        evaluable=evaluable,
        scores=ranking,
        probabilities=probabilities,
        exposure=exposure,
        observed_loss=observed,
        observed_loss_mature=loss_mature,
        loss_fraction=loss_fraction,
        fractions=fractions,
        thresholds=thresholds,
        threshold_kind=snapshot.threshold_kind,
        unit=snapshot.exposure_unit,
    )
    operating_point = _operating_point(tables.threshold_analysis, snapshot, thresholds)

    warnings: list[str] = []
    if df.index.has_duplicates:
        warnings.append("duplicate_index")
    try:
        if bool(df.duplicated().any()):
            warnings.append("duplicate_rows")
    except Exception:
        pass
    if missing_positions:
        warnings.append("missing_target_rows_excluded")
    if external_predictions is not None:
        warnings.append("external_fit_not_verifiable")
    if duplicate_thresholds:
        warnings.append("duplicate_thresholds_removed")
    if duplicate_fractions:
        warnings.append("duplicate_gain_fractions_removed")
    if len(df) > 1_000_000:
        warnings.append("large_input")
    if estimator is not None:
        warnings.append("custom_estimator_random_state_not_managed")

    limitations: list[str] = []
    if snapshot.validation_mode in (
        "stratified_holdout",
        "stratified_kfold",
        "group_holdout",
        "group_kfold",
    ):
        limitations.append("random_or_group_validation_not_time_safe")
    if snapshot.validation_mode.startswith("stratified"):
        limitations.append("entity_isolation_not_checked")
    if snapshot.validation_mode.startswith("time"):
        limitations.append("time_validation_not_general_feature_audit")
    if external_predictions is not None:
        limitations.append("external_fit_not_verifiable")
    if (
        external_predictions is not None
        and external_predictions.ranking_scores is not None
        and probabilities is not None
    ):
        limitations.append("ranking_probability_order_may_differ")
    if probabilities is None:
        limitations.append("probability_metrics_unavailable")
    limitations.append("calibration_diagnostic_only")
    if snapshot.validation_mode.startswith("time") and not evaluable.all():
        limitations.append("partial_validation_maturity")
    if any(
        len({_label_key(labels[position]) for position in plan.evaluable_positions}) < 2
        for plan in plans
        if plan.evaluable_positions
    ):
        limitations.append("single_class_validation_fold")
    limitations.append("observed_association_not_causal")
    if estimator is not None:
        limitations.append("custom_estimator_determinism_not_guaranteed")

    result = BinaryRiskValidationResult(
        target=target,
        positive_label=positive,
        validation_mode=snapshot.validation_mode,
        config=snapshot,
        prediction_scope=(
            "validation"
            if snapshot.validation_mode
            in ("stratified_holdout", "group_holdout", "time_holdout")
            else "oof"
        ),
        score_source=score_source,
        score_direction="higher_positive_event_risk",
        probability_provenance=probability_provenance,
        input_n_rows=len(df),
        eligible_n_rows=len(eligible),
        predicted_n_rows=len(predicted_positions),
        evaluable_n_rows=int(evaluable.sum()),
        requested_threshold_count=len(snapshot.thresholds),
        actual_threshold_count=len(thresholds),
        observed_loss_maturity_mode=loss_mode,
        observed_loss_analysis_as_of=loss_as_of,
        observed_loss_mature_n=loss_mature_n,
        observed_loss_excluded_n=loss_excluded_n,
        folds=folds.copy(deep=True),
        predictions=predictions.copy(deep=True),
        excluded_rows=excluded_rows.copy(deep=True),
        metrics=tables.metrics.copy(deep=True),
        gains=tables.gains.copy(deep=True),
        calibration=tables.calibration.copy(deep=True),
        threshold_analysis=tables.threshold_analysis.copy(deep=True),
        operating_point=operating_point.copy(deep=True),
        business_metrics=business_metrics.copy(deep=True),
        warnings=tuple(warnings),
        limitations=tuple(limitations),
    )
    _validate_binary_risk_validation_result(result)
    return result
