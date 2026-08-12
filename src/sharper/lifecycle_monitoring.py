"""Typed foundation for offline post-loan lifecycle monitoring.

Task 18 owns point-in-time orchestration.  Closed comparison semantics remain
owned by Task 16's private condition kernel.
"""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from math import isfinite
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from sharper._condition_kernel import (
    _ConditionNode,
    _ConditionOperand,
    _evaluate_condition,
)
from sharper.data_audit import DataAuditResult
from sharper.risk_validation import _validate_binary_risk_validation_result

if TYPE_CHECKING:
    from sharper.risk_validation import BinaryRiskValidationResult


@dataclass(frozen=True)
class MonitoringCondition:
    """Declare one closed condition tree for warning or lifecycle evidence.

    Attributes
    ----------
    kind
        Required. One of ``atomic``, ``and``, ``or``, or ``not``.
    operator, left_kind, left, right_kind, right, window, children
        Required or empty according to the frozen condition shape. ``children``
        defaults to ``()`` and only logical nodes may contain children.

    Validation and Errors
    ---------------------
    :func:`monitor_lifecycle` validates the complete tree before evaluating a
    row and raises ``ValueError`` with a stable lifecycle-condition key.

    Missing and Unavailable Behavior
    --------------------------------
    Atomic truth, including missing operands, is delegated to Task 16.

    Side Effects and Immutability
    -----------------------------
    This shallow-frozen data object is never mutated.

    Examples
    --------
    >>> MonitoringCondition("atomic", "gt", "column", "days_past_due", "literal", 0)
    MonitoringCondition("atomic", "gt", "column", "days_past_due", "literal", 0)
    """

    kind: Literal["atomic", "and", "or", "not"]
    operator: (
        Literal[
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
            "in",
            "not_in",
            "between",
            "is_missing",
            "is_not_missing",
        ]
        | None
    ) = None
    left_kind: (
        Literal[
            "column",
            "ranking_score",
            "event_probability",
            "prior_value",
            "change",
            "trend",
            "history_mean",
            "peer_deviation",
            "prior_state",
            "state_transition",
        ]
        | None
    ) = None
    left: str | None = None
    right_kind: Literal["literal", "column"] | None = None
    right: object | None = None
    window: Literal["recent", "history"] | None = None
    children: tuple[MonitoringCondition, ...] = ()


@dataclass(frozen=True)
class EarlyWarningRule:
    """Declare an offline warning rule.

    Attributes
    ----------
    rule_key, priority, alert_level, condition
        Required. ``rule_key`` is unique within its scenario.
    persistence_observations
        Default: ``1``.
    resolution_observations
        Default: ``1``.
    cooldown
        Default: ``timedelta(0)``.
    enabled
        Default: ``True``. Disabled rules are still structurally validated.
    effective_from, expires_at, description_key
        Default: ``None``. Effective boundaries are validated at execution.

    Validation and Errors
    ---------------------
    Invalid keys, windows, and condition trees raise stable ``ValueError`` keys.

    Missing and Unavailable Behavior
    --------------------------------
    Missing operands are kernel unknowns rather than false alerts.

    Side Effects and Immutability
    -----------------------------
    This shallow-frozen declaration is never modified and sends no notification.

    Examples
    --------
    >>> EarlyWarningRule("arrears", 0, "high", MonitoringCondition("atomic"))
    """

    rule_key: str
    priority: int
    alert_level: str
    condition: MonitoringCondition
    persistence_observations: int = 1
    resolution_observations: int = 1
    cooldown: timedelta = timedelta(0)
    enabled: bool = True
    effective_from: datetime | None = None
    expires_at: datetime | None = None
    description_key: str | None = None


@dataclass(frozen=True)
class WarningScenario:
    """Declare a bounded offline warning scenario.

    Attributes
    ----------
    scenario_key, scenario_kind, rules
        Required. ``rules`` is a tuple and is validated as a closed scenario.

    Validation and Errors
    ---------------------
    The scenario shape and every rule are validated before any row evaluation.

    Missing and Unavailable Behavior
    --------------------------------
    ``no_alert`` scenarios retain unavailable event metrics in later phases.

    Side Effects and Immutability
    -----------------------------
    This shallow-frozen declaration has no external side effects.

    Examples
    --------
    >>> WarningScenario("reference", "no_alert", ())
    WarningScenario(scenario_key='reference', scenario_kind='no_alert', rules=())
    """

    scenario_key: str
    scenario_kind: Literal[
        "no_alert", "single_threshold", "rule_set", "model_score", "model_plus_rules"
    ]
    rules: tuple[EarlyWarningRule, ...]


@dataclass(frozen=True)
class LifecycleState:
    """Declare one caller-ranked lifecycle state.

    Attributes
    ----------
    state_key, state_rank, priority, condition
        Required. Rank describes direction; priority resolves matching states.
    terminal
        Default: ``False``.
    enabled
        Default: ``True``. Disabled state conditions remain fully validated.
    description_key
        Default: ``None``.

    Validation and Errors
    ---------------------
    State declarations and their conditions are validated by the monitor.

    Missing and Unavailable Behavior
    --------------------------------
    Unknown condition truth is distinct from a caller default state.

    Side Effects and Immutability
    -----------------------------
    This shallow-frozen declaration is not changed by a monitoring run.

    Examples
    --------
    >>> LifecycleState("performing", 0, 0, MonitoringCondition("atomic"))
    """

    state_key: str
    state_rank: int
    priority: int
    condition: MonitoringCondition
    terminal: bool = False
    enabled: bool = True
    description_key: str | None = None


@dataclass(frozen=True)
class LifecycleMonitoringConfig:
    """Configure one deterministic, point-in-time offline monitoring run.

    Attributes
    ----------
    monitoring_key, monitoring_version, analysis_as_of, entity_column,
    observation_time_column, available_time_column, condition_feature_columns,
    event_time_column, positive_event_key, prediction_horizon,
    horizon_end_inclusive, recent_window, history_window,
    history_start_inclusive, expected_observation_interval, period_unit,
    time_zone, scenarios, reference_scenario_key, alert_level_ranks, states,
    default_state_key, unknown_state_key
        Required. Availability and time semantics are explicit; no clock is read.
    allowed_transitions, adverse_state_keys, cure_state_keys
        Default: ``()``.
    cohort_time_column, cohort_column, peer_reference_start, peer_reference_end,
    ranking_score_column, ranking_score_direction, exposure_column,
    loss_fraction, observed_loss_column, observed_loss_available_time_column
        Default: ``None``.
    peer_group_columns, segment_columns
        Default: ``()``.
    observed_loss_is_mature_snapshot
        Default: ``False``.
    time_frequency
        Default: ``"month"``.

    Validation and Errors
    ---------------------
    :func:`monitor_lifecycle` validates this complete configuration before data
    processing and raises a stable lifecycle-config or resource-limit error.

    Missing and Unavailable Behavior
    --------------------------------
    Optional sources remain unavailable unless explicitly declared and aligned.

    Side Effects and Immutability
    -----------------------------
    The shallow-frozen configuration and its tuples remain unchanged.

    Examples
    --------
    >>> # LifecycleMonitoringConfig(...) declares all time and source roles.
    """

    monitoring_key: str
    monitoring_version: str
    analysis_as_of: datetime
    entity_column: str
    observation_time_column: str
    available_time_column: str
    condition_feature_columns: tuple[str, ...]
    event_time_column: str | None
    positive_event_key: str | None
    prediction_horizon: timedelta | None
    horizon_end_inclusive: bool
    recent_window: timedelta
    history_window: timedelta
    history_start_inclusive: bool
    expected_observation_interval: timedelta | None
    period_unit: Literal["day", "week", "month", "quarter"]
    time_zone: str | None
    scenarios: tuple[WarningScenario, ...]
    reference_scenario_key: str
    alert_level_ranks: tuple[tuple[str, int], ...]
    states: tuple[LifecycleState, ...]
    default_state_key: str
    unknown_state_key: str
    allowed_transitions: tuple[tuple[str, str], ...] = ()
    adverse_state_keys: tuple[str, ...] = ()
    cure_state_keys: tuple[str, ...] = ()
    cohort_time_column: str | None = None
    cohort_column: str | None = None
    peer_group_columns: tuple[str, ...] = ()
    peer_reference_start: datetime | None = None
    peer_reference_end: datetime | None = None
    ranking_score_column: str | None = None
    ranking_score_direction: Literal["higher_risk", "lower_risk"] | None = None
    exposure_column: str | None = None
    loss_fraction: float | str | None = None
    observed_loss_column: str | None = None
    observed_loss_available_time_column: str | None = None
    observed_loss_is_mature_snapshot: bool = False
    segment_columns: tuple[str, ...] = ()
    time_frequency: Literal["day", "week", "month", "quarter"] = "month"


@dataclass(frozen=True)
class LifecycleMonitoringResult:
    """Return eleven typed, privacy-preserving offline monitoring tables.

    Attributes
    ----------
    monitoring_key, monitoring_version, monitoring_fingerprint, input_n_rows,
    entity_count, evaluable_observation_count, requested_scenario_count,
    requested_rule_count, active_rule_count, requested_state_count
        Immutable run identifiers and requested/input counts.
    observation_history, rule_evaluations, notifications, alert_episodes,
    event_matches, state_history, state_transitions, monitoring_summary,
    scenario_comparison, lifecycle_summary, provenance
        The eleven frozen-schema DataFrame results.
    warnings, limitations
        Ordered, sanitized closed-vocabulary messages.

    Validation and Errors
    ---------------------
    A result is returned only after whole-config, role, and time validation.

    Missing and Unavailable Behavior
    --------------------------------
    Empty tables retain their exact nullable schemas; unavailable evidence is
    never represented as a fabricated zero.

    Side Effects and Immutability
    -----------------------------
    The dataclass is shallow frozen and never retains the input DataFrame.

    Examples
    --------
    >>> # result = monitor_lifecycle(frame, config)
    """

    monitoring_key: str
    monitoring_version: str
    monitoring_fingerprint: str
    input_n_rows: int
    entity_count: int
    evaluable_observation_count: int
    requested_scenario_count: int
    requested_rule_count: int
    active_rule_count: int
    requested_state_count: int
    observation_history: pd.DataFrame
    rule_evaluations: pd.DataFrame
    notifications: pd.DataFrame
    alert_episodes: pd.DataFrame
    event_matches: pd.DataFrame
    state_history: pd.DataFrame
    state_transitions: pd.DataFrame
    monitoring_summary: pd.DataFrame
    scenario_comparison: pd.DataFrame
    lifecycle_summary: pd.DataFrame
    provenance: pd.DataFrame
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class _ScopeGroup:
    """Retain only an anonymous ordinal and physical row membership."""

    scope_position: int
    row_positions: frozenset[int]


@dataclass(frozen=True)
class _ScopeFacts:
    """Private approved scope memberships; raw category values never escape."""

    segment_groups: tuple[_ScopeGroup, ...]
    time_groups: tuple[_ScopeGroup, ...]
    cohort_groups: tuple[_ScopeGroup, ...]
    vintage_groups: tuple[_ScopeGroup, ...]
    segment_time_groups: tuple[_ScopeGroup, ...]
    cohort_time_groups: tuple[_ScopeGroup, ...]
    cohort_positions: tuple[object, ...]
    period_indices: tuple[object, ...]


_TABLE_SCHEMAS: dict[str, tuple[tuple[str, str], ...]] = {
    "observation_history": (
        ("row_position", "int64"),
        ("entity_position", "int64"),
        ("observation_time", "datetime64[ns]"),
        ("observation_status", "string"),
        ("observation_reason", "string"),
        ("is_consecutive", "boolean"),
        ("period_index", "Int64"),
        ("cohort_position", "Int64"),
        ("primary_scenario_key", "string"),
        ("primary_rule_key", "string"),
        ("primary_alert_level", "string"),
        ("primary_alert_rank", "Int64"),
        ("active_rule_count", "int64"),
        ("emitted_notification_count", "int64"),
        ("maturity_status", "string"),
        ("event_within_horizon", "boolean"),
        ("effective_state_key", "string"),
        ("effective_state_rank", "Int64"),
        ("state_status", "string"),
        ("state_reason", "string"),
    ),
    "rule_evaluations": (
        ("row_position", "int64"),
        ("entity_position", "int64"),
        ("observation_time", "datetime64[ns]"),
        ("scenario_key", "string"),
        ("scenario_order", "int64"),
        ("rule_key", "string"),
        ("rule_order", "int64"),
        ("alert_level", "string"),
        ("alert_rank", "int64"),
        ("path_status", "string"),
        ("truth", "string"),
        ("true_streak", "int64"),
        ("false_streak", "int64"),
        ("episode_status", "string"),
        ("notification_status", "string"),
        ("status", "string"),
        ("reason", "string"),
        ("finding_key", "string"),
    ),
    "notifications": (
        ("entity_position", "int64"),
        ("scenario_key", "string"),
        ("rule_key", "string"),
        ("episode_ordinal", "int64"),
        ("notification_ordinal", "int64"),
        ("row_position", "int64"),
        ("notification_time", "datetime64[ns]"),
        ("alert_level", "string"),
        ("alert_rank", "int64"),
        ("notification_kind", "string"),
        ("is_repeated", "boolean"),
        ("first_matched_event_ordinal", "Int64"),
        ("matched_event_count", "int64"),
        ("maturity_status", "string"),
        ("status", "string"),
        ("reason", "string"),
        ("finding_key", "string"),
    ),
    "alert_episodes": (
        ("entity_position", "int64"),
        ("scenario_key", "string"),
        ("rule_key", "string"),
        ("episode_ordinal", "int64"),
        ("alert_level", "string"),
        ("alert_rank", "int64"),
        ("episode_start_time", "datetime64[ns]"),
        ("episode_end_time", "datetime64[ns]"),
        ("duration_seconds", "Float64"),
        ("raw_hit_count", "int64"),
        ("notification_count", "int64"),
        ("suppressed_notification_count", "int64"),
        ("is_reopen", "boolean"),
        ("is_unresolved", "boolean"),
        ("status", "string"),
        ("reason", "string"),
        ("finding_key", "string"),
    ),
    "event_matches": (
        ("scenario_key", "string"),
        ("entity_position", "int64"),
        ("event_ordinal", "int64"),
        ("event_row_position", "Int64"),
        ("event_time", "datetime64[ns]"),
        ("duplicate_source_row_count", "int64"),
        ("event_status", "string"),
        ("match_status", "string"),
        ("captured", "boolean"),
        ("capturing_rule_key", "string"),
        ("capturing_episode_ordinal", "Int64"),
        ("capturing_notification_ordinal", "Int64"),
        ("capturing_notification_row_position", "Int64"),
        ("notification_time", "datetime64[ns]"),
        ("lead_time_seconds", "Float64"),
        ("candidate_notification_count", "int64"),
        ("status", "string"),
        ("reason", "string"),
        ("finding_key", "string"),
    ),
    "state_history": (
        ("row_position", "int64"),
        ("entity_position", "int64"),
        ("observation_time", "datetime64[ns]"),
        ("candidate_state_key", "string"),
        ("candidate_state_rank", "Int64"),
        ("candidate_state_priority", "Int64"),
        ("effective_state_key", "string"),
        ("effective_state_rank", "Int64"),
        ("matching_state_count", "int64"),
        ("status", "string"),
        ("reason", "string"),
        ("finding_key", "string"),
    ),
    "state_transitions": (
        ("entity_position", "int64"),
        ("from_row_position", "Int64"),
        ("to_row_position", "int64"),
        ("transition_time", "datetime64[ns]"),
        ("from_state_key", "string"),
        ("candidate_to_state_key", "string"),
        ("effective_to_state_key", "string"),
        ("from_rank", "Int64"),
        ("candidate_to_rank", "Int64"),
        ("effective_to_rank", "Int64"),
        ("transition_kind", "string"),
        ("transition_direction", "string"),
        ("is_allowed", "boolean"),
        ("is_consecutive", "boolean"),
        ("is_cure", "boolean"),
        ("exposure", "Float64"),
        ("observed_loss", "Float64"),
        ("status", "string"),
        ("reason", "string"),
        ("finding_key", "string"),
    ),
    "monitoring_summary": (
        ("scope_key", "string"),
        ("scope_position", "Int64"),
        ("scenario_key", "string"),
        ("rule_key", "string"),
        ("metric", "string"),
        ("metric_value", "Float64"),
        ("numerator", "Float64"),
        ("denominator", "Float64"),
        ("support_n", "int64"),
        ("support_unit", "string"),
        ("mature_n", "int64"),
        ("censored_n", "int64"),
        ("unit", "string"),
        ("status", "string"),
        ("reason", "string"),
        ("finding_key", "string"),
    ),
    "scenario_comparison": (
        ("reference_scenario_key", "string"),
        ("comparator_scenario_key", "string"),
        ("metric", "string"),
        ("scope_key", "string"),
        ("scope_position", "Int64"),
        ("rule_key", "string"),
        ("reference_value", "Float64"),
        ("comparator_value", "Float64"),
        ("delta", "Float64"),
        ("numerator", "Float64"),
        ("denominator", "Float64"),
        ("support_n", "int64"),
        ("support_unit", "string"),
        ("status", "string"),
        ("reason", "string"),
        ("finding_key", "string"),
    ),
    "lifecycle_summary": (
        ("scope_key", "string"),
        ("scope_position", "Int64"),
        ("from_state_key", "string"),
        ("to_state_key", "string"),
        ("metric", "string"),
        ("metric_value", "Float64"),
        ("numerator", "Float64"),
        ("denominator", "Float64"),
        ("support_n", "int64"),
        ("support_unit", "string"),
        ("unit", "string"),
        ("status", "string"),
        ("reason", "string"),
        ("finding_key", "string"),
    ),
    "provenance": (
        ("provenance_key", "string"),
        ("provenance_value", "string"),
        ("status", "string"),
        ("reason", "string"),
        ("finding_key", "string"),
    ),
}

_OPERATORS = frozenset(
    {
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "in",
        "not_in",
        "between",
        "is_missing",
        "is_not_missing",
    }
)
_DERIVED = frozenset(
    {"prior_value", "change", "trend", "history_mean", "peer_deviation"}
)


def _empty_table(name: str, aware: bool) -> pd.DataFrame:
    """Create a fresh typed empty contract table."""
    columns: dict[str, pd.Series] = {}
    for column, dtype in _TABLE_SCHEMAS[name]:
        resolved = (
            "datetime64[ns, UTC]" if aware and dtype == "datetime64[ns]" else dtype
        )
        columns[column] = pd.Series([], dtype=resolved)
    return pd.DataFrame(columns)


def _typed_table(
    name: str, rows: list[dict[str, object]], aware: bool
) -> pd.DataFrame:
    """Materialize one contract table with its exact nullable dtypes."""
    if not rows:
        return _empty_table(name, aware)
    populated = pd.DataFrame(
        rows, columns=[column for column, _ in _TABLE_SCHEMAS[name]]
    )
    for column, dtype in _TABLE_SCHEMAS[name]:
        resolved = (
            "datetime64[ns, UTC]" if aware and dtype == "datetime64[ns]" else dtype
        )
        if aware and dtype == "datetime64[ns]":
            populated[column] = pd.to_datetime(populated[column], utc=True)
        else:
            populated[column] = populated[column].astype(resolved)
    return populated


def _error(scope: str, key: str) -> ValueError:
    return ValueError(f"lifecycle {scope}: {key}")


def _key(value: object) -> str:
    if type(value) is not str or not value or len(value) > 64 or not value.isascii():
        raise _error("config is invalid", "invalid_key")
    if any(not (char.isalnum() or char in "._-") for char in value):
        raise _error("config is invalid", "invalid_key")
    return value


def _scalar(value: object) -> bool:
    """Check exact safe scalar families without invoking object protocols."""
    value_type = type(value)
    return (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or value_type
        in {
            bool,
            int,
            float,
            str,
            date,
            datetime,
            pd.Timestamp,
        }
    )


def _datetime(value: object, missing_key: str) -> datetime | pd.Timestamp:
    if value is None or value is pd.NaT:
        raise _error("input schema is invalid", missing_key)
    if type(value) not in {datetime, pd.Timestamp}:
        raise _error("input schema is invalid", "datetime_type_invalid")
    return value


def _aware(value: datetime | pd.Timestamp) -> bool:
    try:
        return value.tzinfo is not None and value.utcoffset() is not None
    except (TypeError, ValueError):
        raise _error("input schema is invalid", "datetime_timezone_invalid") from None


def _validate_declared_time_zone(
    value: datetime | pd.Timestamp,
    config: LifecycleMonitoringConfig,
    *,
    scope: str = "input schema is invalid",
) -> None:
    """Require an explicit source-zone declaration to match aware timestamps."""
    if config.time_zone is None or not _aware(value):
        return
    try:
        zone_name = value.tzname()
    except (TypeError, ValueError):
        raise _error(scope, "datetime_timezone_invalid") from None
    if type(zone_name) is not str or zone_name != config.time_zone:
        raise _error(scope, "datetime_timezone_invalid")


def _entity_equal(left: object, right: object) -> bool:
    """Compare only prevalidated exact builtin scalar values."""
    return type(left) is type(right) and left == right


def _condition_error(key: str) -> ValueError:
    return _error("condition", key)


def _validate_condition(
    condition: object,
    config: LifecycleMonitoringConfig,
    depth: int = 1,
    nodes: list[int] | None = None,
) -> None:
    if nodes is None:
        nodes = [0]
    if type(condition) is not MonitoringCondition:
        raise _condition_error("invalid_structure")
    nodes[0] += 1
    if depth > 8:
        raise _error("resource limit exceeded", "condition_depth")
    if nodes[0] > 128:
        raise _error("resource limit exceeded", "condition_nodes")
    if condition.kind == "atomic":
        if (
            condition.children
            or condition.operator not in _OPERATORS
            or condition.left_kind is None
        ):
            raise _condition_error("invalid_structure")
        unary = condition.operator in {"is_missing", "is_not_missing"}
        if unary and (condition.right_kind is not None or condition.right is not None):
            _validate_operand_role(condition, config)
            raise _condition_error("invalid_arity")
        if not unary and condition.right_kind not in {"literal", "column"}:
            raise _condition_error("invalid_arity")
        _validate_operand_role(condition, config)
        if condition.right_kind == "literal" and not _literal_safe(
            condition.right, condition.operator
        ):
            raise _condition_error("invalid_literal")
        return
    fields = (
        condition.operator,
        condition.left_kind,
        condition.left,
        condition.right_kind,
        condition.right,
        condition.window,
    )
    expected = (
        2 if condition.kind in {"and", "or"} else 1 if condition.kind == "not" else 0
    )
    if (
        condition.kind not in {"and", "or", "not"}
        or any(item is not None for item in fields)
        or len(condition.children) < expected
        or (condition.kind == "not" and len(condition.children) != 1)
    ):
        raise _condition_error("invalid_structure")
    for child in condition.children:
        _validate_condition(child, config, depth + 1, nodes)


def _literal_safe(value: object, operator: str | None) -> bool:
    if operator in {"in", "not_in", "between"}:
        if type(value) is not tuple or not value or len(value) > 100:
            return False
        if operator == "between" and len(value) != 2:
            return False
        return all(_literal_safe(item, "eq") for item in value)
    if not _scalar(value) or value is None or value is pd.NA or value is pd.NaT:
        return False
    return not (type(value) is float and not isfinite(value))


def _validate_operand_role(
    condition: MonitoringCondition, config: LifecycleMonitoringConfig
) -> None:
    reserved = _reserved_columns(config)
    if condition.left_kind == "column" or condition.left_kind in _DERIVED:
        if (
            type(condition.left) is not str
            or condition.left not in config.condition_feature_columns
            or condition.left in reserved
        ):
            raise _condition_error("forbidden_condition_role")
    elif condition.left_kind in {
        "ranking_score",
        "event_probability",
        "prior_state",
        "state_transition",
    }:
        if condition.left is not None:
            raise _condition_error("invalid_structure")
    else:
        raise _condition_error("invalid_structure")
    if condition.right_kind == "column":
        if (
            type(condition.right) is not str
            or condition.right not in config.condition_feature_columns
            or condition.right in reserved
        ):
            raise _condition_error("forbidden_condition_role")


def _reserved_columns(config: LifecycleMonitoringConfig) -> frozenset[str]:
    values: list[str | None] = [
        config.entity_column,
        config.observation_time_column,
        config.available_time_column,
        config.event_time_column,
        config.exposure_column,
        config.observed_loss_column,
        config.observed_loss_available_time_column,
        config.cohort_time_column,
        config.cohort_column,
        config.ranking_score_column,
    ]
    if type(config.loss_fraction) is str:
        values.append(config.loss_fraction)
    values.extend(config.segment_columns)
    values.extend(config.peer_group_columns)
    return frozenset(value for value in values if type(value) is str)


def _compile_condition(
    condition: MonitoringCondition, *, root: bool = True
) -> _ConditionNode:
    if condition.kind != "atomic":
        children = tuple(
            _compile_condition(child, root=False) for child in condition.children
        )
        return _ConditionNode(
            condition.kind,
            None,
            None,
            None,
            children,
            None,
            None,
            "task18" if root else None,
        )
    left = condition.left if condition.left_kind == "column" else "__task18_signal"
    right = None
    if condition.right_kind == "column":
        right = _ConditionOperand("column", condition.right)
    elif condition.right_kind == "literal":
        right = _ConditionOperand("literal", condition.right)
    return _ConditionNode(
        "atomic",
        condition.operator,
        _ConditionOperand("column", left),
        right,
        (),
        None,
        None,
        "task18" if root else None,
    )


def _canonical_encode(value: object) -> object:
    """Encode only approved fingerprint values without protocol fallbacks."""
    value_type = type(value)
    if value is None or value_type is bool or value_type is int or value_type is str:
        return value
    if value_type is float:
        if not isfinite(value):
            raise TypeError("non-finite canonical value")
        return value
    if value_type is date and value_type is not datetime:
        return {"__date__": value.isoformat()}
    if value_type in {datetime, pd.Timestamp}:
        timestamp = value
        if timestamp.tzinfo is not None and timestamp.utcoffset() is not None:
            timestamp = timestamp.astimezone(timezone.utc)
        return {"__datetime__": timestamp.isoformat()}
    if value_type is timedelta:
        return {"__timedelta_seconds__": value.total_seconds()}
    if value_type is tuple or value_type is list:
        return [_canonical_encode(item) for item in value]
    if value_type is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("canonical mapping keys must be strings")
        return {key: _canonical_encode(value[key]) for key in sorted(value)}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonical_encode(getattr(value, item.name))
            for item in fields(value)
        }
    raise TypeError("unsupported canonical value")


def _canonical_json(value: object) -> str:
    return json.dumps(
        _canonical_encode(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(config: LifecycleMonitoringConfig) -> str:
    payload = _canonical_json(config)
    return sha256(payload.encode("ascii")).hexdigest()


def _validate_config(config: object) -> LifecycleMonitoringConfig:
    if type(config) is not LifecycleMonitoringConfig:
        raise _error("config is invalid", "wrong_type")
    for value in (
        config.monitoring_key,
        config.monitoring_version,
        config.entity_column,
        config.observation_time_column,
        config.available_time_column,
    ):
        _key(value)
    if type(config.condition_feature_columns) is not tuple or any(
        type(column) is not str for column in config.condition_feature_columns
    ):
        raise _error("config is invalid", "invalid_key")
    if (
        type(config.analysis_as_of) not in {datetime, pd.Timestamp}
        or type(config.recent_window) is not timedelta
        or type(config.history_window) is not timedelta
        or config.recent_window <= timedelta(0)
        or config.history_window < config.recent_window
        or (
            config.expected_observation_interval is not None
            and (
                type(config.expected_observation_interval) is not timedelta
                or config.expected_observation_interval <= timedelta(0)
            )
        )
    ):
        raise _error("config is invalid", "invalid_time_window")
    try:
        analysis_aware = _aware(config.analysis_as_of)
    except ValueError:
        raise _error("config is invalid", "invalid_time_window") from None
    if config.time_zone is not None and (
        type(config.time_zone) is not str
        or not config.time_zone
        or len(config.time_zone) > 128
    ):
        raise _error("config is invalid", "invalid_time_window")
    if config.time_zone is not None and not analysis_aware:
        raise _error("config is invalid", "invalid_time_window")
    if config.time_zone is not None:
        try:
            zone_name = config.analysis_as_of.tzname()
        except (TypeError, ValueError):
            raise _error("config is invalid", "invalid_time_window") from None
        if type(zone_name) is not str or zone_name != config.time_zone:
            raise _error("config is invalid", "invalid_time_window")
    declaration = (
        config.event_time_column,
        config.positive_event_key,
        config.prediction_horizon,
    )
    if any(item is None for item in declaration) and any(
        item is not None for item in declaration
    ):
        raise _error("config is invalid", "invalid_horizon_declaration")
    if config.event_time_column is not None:
        _key(config.event_time_column)
        _key(config.positive_event_key)
        if type(
            config.prediction_horizon
        ) is not timedelta or config.prediction_horizon <= timedelta(0):
            raise _error("config is invalid", "invalid_horizon_declaration")
    if type(config.horizon_end_inclusive) is not bool:
        raise _error("config is invalid", "invalid_horizon_declaration")
    score_declaration = (config.ranking_score_column, config.ranking_score_direction)
    if any(item is None for item in score_declaration) and any(
        item is not None for item in score_declaration
    ):
        raise _error("config is invalid", "invalid_score_declaration")
    if config.ranking_score_direction is not None and (
        type(config.ranking_score_direction) is not str
        or config.ranking_score_direction not in {"higher_risk", "lower_risk"}
    ):
        raise _error("config is invalid", "invalid_score_declaration")
    if type(config.alert_level_ranks) is not tuple:
        raise _error("config is invalid", "invalid_alert_level_mapping")
    alert_keys: set[str] = set()
    for item in config.alert_level_ranks:
        if type(item) is not tuple or len(item) != 2:
            raise _error("config is invalid", "invalid_alert_level_mapping")
        level, rank = item
        if (
            type(level) is not str
            or not level
            or len(level) > 64
            or not level.isascii()
            or any(not (char.isalnum() or char in "._-") for char in level)
            or type(rank) is not int
            or rank < 0
            or rank > 9999
            or level in alert_keys
        ):
            raise _error("config is invalid", "invalid_alert_level_mapping")
        alert_keys.add(level)
    if not alert_keys:
        raise _error("config is invalid", "invalid_alert_level_mapping")
    if type(config.time_frequency) is not str or config.time_frequency not in {
        "day",
        "week",
        "month",
        "quarter",
    }:
        raise _error("config is invalid", "invalid_time_window")
    if type(config.segment_columns) is not tuple or len(config.segment_columns) > 4:
        raise _error("config is invalid", "invalid_key")
    declared_scope_columns = (
        *config.segment_columns,
        config.cohort_column,
        config.cohort_time_column,
    )
    for column in declared_scope_columns:
        if column is not None:
            _key(column)
    if type(config.period_unit) is not str or config.period_unit not in {
        "day",
        "week",
        "month",
        "quarter",
    }:
        raise _error("config is invalid", "invalid_time_window")
    if type(config.scenarios) is not tuple:
        raise _error("config is invalid", "invalid_scenario_shape")
    if len(config.scenarios) > 10:
        raise _error("resource limit exceeded", "scenarios")
    if sum(len(s.rules) for s in config.scenarios if type(s) is WarningScenario) > 100:
        raise _error("resource limit exceeded", "all_rules")
    if type(config.states) is not tuple:
        raise _error("config is invalid", "invalid_state")
    if len(config.states) > 50:
        raise _error("resource limit exceeded", "states")
    if type(config.allowed_transitions) is not tuple:
        raise _error("config is invalid", "invalid_transition")
    if len(config.allowed_transitions) > 2500:
        raise _error("resource limit exceeded", "allowed_transitions")
    return config


def _validate_data(
    data: object, config: LifecycleMonitoringConfig
) -> tuple[list[int], bool]:
    if type(data) is not pd.DataFrame:
        raise _error("input schema is invalid", "invalid_dataframe")
    if any(type(column) is not str or not column for column in data.columns):
        raise _error("input schema is invalid", "invalid_column_labels")
    if not data.columns.is_unique:
        raise _error("input schema is invalid", "duplicate_columns")
    if len(data) > 100000:
        raise _error("resource limit exceeded", "input_rows")
    required = {
        config.entity_column,
        config.observation_time_column,
        config.available_time_column,
        *config.condition_feature_columns,
        *config.peer_group_columns,
        *config.segment_columns,
    }
    if config.event_time_column is not None:
        required.add(config.event_time_column)
    if config.ranking_score_column is not None:
        required.add(config.ranking_score_column)
    if config.cohort_column is not None:
        required.add(config.cohort_column)
    if config.cohort_time_column is not None:
        required.add(config.cohort_time_column)
    if config.exposure_column is not None:
        required.add(config.exposure_column)
    if type(config.loss_fraction) is str:
        required.add(config.loss_fraction)
    if config.observed_loss_column is not None:
        required.add(config.observed_loss_column)
    if config.observed_loss_available_time_column is not None:
        required.add(config.observed_loss_available_time_column)
    if not required.issubset(data.columns):
        raise _error("input schema is invalid", "missing_column")
    entity_values: list[object] = []
    positions: list[int] = []
    seen: dict[int, set[datetime | pd.Timestamp]] = {}
    awareness: bool | None = None
    as_of = _datetime(config.analysis_as_of, "datetime_type_invalid")
    _validate_declared_time_zone(as_of, config)
    for row_position in range(len(data)):
        entity = data[config.entity_column].iat[row_position]
        if (
            not _scalar(entity)
            or entity is None
            or entity is pd.NA
            or entity is pd.NaT
            or (type(entity) is float and not isfinite(entity))
        ):
            raise _error("input schema is invalid", "missing_entity")
        for column in config.peer_group_columns:
            peer_value = data[column].iat[row_position]
            if not _scalar(peer_value) and not _scope_missing(peer_value):
                raise _error("input schema is invalid", "unsupported_dtype")
        entity_position = next(
            (
                i
                for i, prior in enumerate(entity_values)
                if _entity_equal(prior, entity)
            ),
            None,
        )
        if entity_position is None:
            entity_values.append(entity)
            entity_position = len(entity_values) - 1
        observation = _datetime(
            data[config.observation_time_column].iat[row_position],
            "observation_time_missing",
        )
        available = _datetime(
            data[config.available_time_column].iat[row_position],
            "available_time_missing",
        )
        for timestamp in (as_of, observation, available):
            is_aware = _aware(timestamp)
            _validate_declared_time_zone(timestamp, config)
            if awareness is None:
                awareness = is_aware
            elif awareness != is_aware:
                raise _error("input schema is invalid", "datetime_awareness_mismatch")
        if observation > as_of:
            raise _error("input schema is invalid", "observation_after_as_of")
        if available > as_of:
            raise _error("input schema is invalid", "available_after_as_of")
        if available > observation:
            raise _error("input schema is invalid", "available_after_observation")
        if config.event_time_column is not None:
            event_time = data[config.event_time_column].iat[row_position]
            if (
                event_time is not None
                and event_time is not pd.NA
                and event_time is not pd.NaT
            ):
                event_datetime = _datetime(event_time, "datetime_type_invalid")
                _validate_declared_time_zone(event_datetime, config)
                if _aware(event_datetime) != bool(awareness):
                    raise _error(
                        "input schema is invalid", "datetime_awareness_mismatch"
                    )
        entity_times = seen.setdefault(entity_position, set())
        if observation in entity_times:
            raise _error("input schema is invalid", "duplicate_entity_observation_time")
        entity_times.add(observation)
        positions.append(entity_position)
    if len(entity_values) > 50000:
        raise _error("resource limit exceeded", "entities")
    if any(positions.count(position) > 10000 for position in range(len(entity_values))):
        raise _error("resource limit exceeded", "observations_per_entity")
    return positions, bool(awareness)


def _scope_missing(value: object) -> bool:
    """Recognize only built-in missing sentinels without object dispatch."""
    return (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or (type(value) in {float, np.float64} and np.isnan(value))
    )


def _category_groups(
    values: pd.Series,
    *,
    maximum: int = 100,
    resource_key: str = "categories_per_scope_column",
) -> tuple[tuple[_ScopeGroup, ...], tuple[object, ...]]:
    """Assign safe category ordinals by first physical appearance, missing last."""
    seen: list[object] = []
    memberships: list[list[int]] = []
    missing_positions: list[int] = []
    positions: list[object] = [pd.NA] * len(values)
    for row_position, value in enumerate(values):
        if _scope_missing(value):
            missing_positions.append(row_position)
            continue
        if not _scalar(value):
            raise _error("input schema is invalid", "unsupported_dtype")
        ordinal = next(
            (
                index
                for index, prior in enumerate(seen)
                if _entity_equal(prior, value)
            ),
            None,
        )
        if ordinal is None:
            seen.append(value)
            memberships.append([])
            ordinal = len(memberships) - 1
        memberships[ordinal].append(row_position)
        positions[row_position] = ordinal
    if missing_positions:
        memberships.append(missing_positions)
        ordinal = len(memberships) - 1
        for row_position in missing_positions:
            positions[row_position] = ordinal
    if len(memberships) > maximum:
        raise _error("resource limit exceeded", resource_key)
    return (
        tuple(
            _ScopeGroup(ordinal, frozenset(row_positions))
            for ordinal, row_positions in enumerate(memberships)
        ),
        tuple(positions),
    )


def _period_index(
    observation: datetime | pd.Timestamp,
    origin: datetime | pd.Timestamp,
    unit: str,
) -> int:
    """Return the contract's non-negative complete elapsed calendar period."""
    elapsed_days = (observation - origin).days
    if unit == "day":
        return elapsed_days
    if unit == "week":
        return elapsed_days // 7
    months = 12 * (observation.year - origin.year) + observation.month - origin.month
    if (observation.day, observation.time()) < (origin.day, origin.time()):
        months -= 1
    return months if unit == "month" else months // 3


def _time_bucket(value: datetime | pd.Timestamp, frequency: str) -> tuple[int, ...]:
    """Build a private calendar bucket key; it is never materialized publicly."""
    if frequency == "day":
        return value.year, value.month, value.day
    if frequency == "week":
        iso = value.isocalendar()
        return int(iso.year), int(iso.week)
    if frequency == "month":
        return value.year, value.month
    return value.year, (value.month - 1) // 3 + 1


def _compound_groups(
    left: tuple[_ScopeGroup, ...],
    right_keys: tuple[object, ...],
) -> tuple[_ScopeGroup, ...]:
    """Materialize actual two-dimensional memberships in frozen left/time order."""
    groups: list[_ScopeGroup] = []
    for group in left:
        buckets: dict[object, list[int]] = {}
        for row_position in group.row_positions:
            key = right_keys[row_position]
            if key is pd.NA:
                continue
            buckets.setdefault(key, []).append(row_position)
        for key in sorted(buckets):
            groups.append(_ScopeGroup(len(groups), frozenset(buckets[key])))
    if len(groups) > 2000:
        raise _error("resource limit exceeded", "derived_scopes")
    return tuple(groups)


def _scope_facts(
    data: pd.DataFrame,
    config: LifecycleMonitoringConfig,
    entity_positions: list[int],
    aware: bool,
) -> _ScopeFacts:
    """Validate approved sources and construct private segment/cohort/vintage sets."""
    empty: tuple[_ScopeGroup, ...] = ()
    segment_groups: list[_ScopeGroup] = []
    for column in config.segment_columns:
        groups, _ = _category_groups(data[column])
        offset = len(segment_groups)
        segment_groups.extend(
            _ScopeGroup(offset + group.scope_position, group.row_positions)
            for group in groups
        )
    cohort_groups, cohort_positions = (
        _category_groups(
            data[config.cohort_column],
            maximum=240,
            resource_key="cohort_buckets",
        )
        if config.cohort_column is not None
        else (empty, tuple(pd.NA for _ in range(len(data))))
    )
    period_indices: list[object] = [pd.NA] * len(data)
    vintage_groups: tuple[_ScopeGroup, ...] = empty
    if config.cohort_time_column is not None:
        origins: dict[int, datetime | pd.Timestamp] = {}
        memberships: dict[int, list[int]] = {}
        for row_position, entity_position in enumerate(entity_positions):
            origin = _datetime(
                data[config.cohort_time_column].iat[row_position],
                "datetime_type_invalid",
            )
            _validate_declared_time_zone(origin, config)
            if _aware(origin) != aware:
                raise _error("input schema is invalid", "datetime_awareness_mismatch")
            prior = origins.get(entity_position)
            if prior is None:
                origins[entity_position] = origin
            elif prior != origin:
                raise _error("input schema is invalid", "invalid_cohort_time")
            observation = _datetime(
                data[config.observation_time_column].iat[row_position],
                "observation_time_missing",
            )
            if observation < origin:
                continue
            period = _period_index(observation, origin, config.period_unit)
            period_indices[row_position] = period
            memberships.setdefault(period, []).append(row_position)
        if len(memberships) > 240:
            raise _error("resource limit exceeded", "vintage_buckets")
        vintage_groups = tuple(
            _ScopeGroup(ordinal, frozenset(memberships[period]))
            for ordinal, period in enumerate(sorted(memberships))
        )
    buckets = tuple(
        _time_bucket(
            _datetime(
                data[config.observation_time_column].iat[row_position],
                "observation_time_missing",
            ),
            config.time_frequency,
        )
        for row_position in range(len(data))
    )
    bucket_memberships: dict[tuple[int, ...], list[int]] = {}
    for row_position, bucket in enumerate(buckets):
        bucket_memberships.setdefault(bucket, []).append(row_position)
    if len(bucket_memberships) > 240:
        raise _error("resource limit exceeded", "time_buckets")
    time_groups = tuple(
        _ScopeGroup(ordinal, frozenset(bucket_memberships[bucket]))
        for ordinal, bucket in enumerate(sorted(bucket_memberships))
    )
    return _ScopeFacts(
        tuple(segment_groups),
        time_groups,
        cohort_groups,
        vintage_groups,
        _compound_groups(tuple(segment_groups), buckets),
        _compound_groups(cohort_groups, buckets),
        tuple(cohort_positions),
        tuple(period_indices),
    )


def _validate_scenarios(config: LifecycleMonitoringConfig) -> None:
    keys: list[str] = []
    required_alert_levels: set[str] = set()
    for scenario in config.scenarios:
        if type(scenario) is not WarningScenario:
            raise _error("config is invalid", "invalid_scenario_shape")
        _key(scenario.scenario_key)
        if scenario.scenario_key in keys:
            raise _error("config is invalid", "duplicate_key")
        if (
            type(scenario.scenario_kind) is not str
            or scenario.scenario_kind
            not in {
                "no_alert",
                "single_threshold",
                "rule_set",
                "model_score",
                "model_plus_rules",
            }
            or type(scenario.rules) is not tuple
            or len(scenario.rules) > 50
        ):
            raise _error("config is invalid", "invalid_scenario_shape")
        keys.append(scenario.scenario_key)
        rule_keys: set[str] = set()
        for rule in scenario.rules:
            if type(rule) is not EarlyWarningRule:
                raise _error("config is invalid", "invalid_rule")
            if type(rule.rule_key) is not str:
                raise _error("config is invalid", "invalid_rule")
            _key(rule.rule_key)
            if rule.rule_key in rule_keys:
                raise _error("config is invalid", "duplicate_key")
            if (
                type(rule.priority) is not int
                or rule.priority < 0
                or rule.priority > 9999
                or type(rule.alert_level) is not str
            ):
                raise _error("config is invalid", "invalid_rule")
            if rule.alert_level not in dict(config.alert_level_ranks):
                raise _error("config is invalid", "invalid_alert_level_mapping")
            required_alert_levels.add(rule.alert_level)
            rule_keys.add(rule.rule_key)
            if (
                type(rule.persistence_observations) is not int
                or rule.persistence_observations < 1
                or type(rule.resolution_observations) is not int
                or rule.resolution_observations < 1
                or type(rule.cooldown) is not timedelta
                or rule.cooldown < timedelta(0)
                or type(rule.enabled) is not bool
            ):
                raise _error("config is invalid", "invalid_rule")
            _validate_rule_boundaries(
                rule,
                config,
                analysis_aware=_aware(config.analysis_as_of),
            )
            _validate_condition(rule.condition, config)
        rule_source_kinds = tuple(
            source
            for rule in scenario.rules
            for source in _condition_source_kinds(rule.condition)
        )
        has_score = any(
            source in {"ranking_score", "event_probability"}
            for source in rule_source_kinds
        )
        all_score = bool(rule_source_kinds) and all(
            source in {"ranking_score", "event_probability"}
            for source in rule_source_kinds
        )
        has_non_score = any(
            source not in {"ranking_score", "event_probability"}
            for source in rule_source_kinds
        )
        if scenario.scenario_kind == "no_alert" and scenario.rules:
            raise _error("config is invalid", "invalid_scenario_shape")
        if scenario.scenario_kind == "single_threshold" and (
            len(scenario.rules) != 1 or not all_score
        ):
            raise _error("config is invalid", "invalid_scenario_shape")
        if scenario.scenario_kind == "rule_set" and (
            not scenario.rules or any(
                source in {"ranking_score", "event_probability"}
                for source in rule_source_kinds
            )
        ):
            raise _error("config is invalid", "invalid_scenario_shape")
        if scenario.scenario_kind == "model_score" and (
            not scenario.rules or not all_score
        ):
            raise _error("config is invalid", "invalid_scenario_shape")
        if scenario.scenario_kind == "model_plus_rules" and (
            len(scenario.rules) < 2 or not has_score or not has_non_score
        ):
            raise _error("config is invalid", "invalid_scenario_shape")
    if not required_alert_levels.issubset(
        {level for level, _ in config.alert_level_ranks}
    ):
        raise _error("config is invalid", "invalid_alert_level_mapping")
    if config.reference_scenario_key not in keys:
        raise _error("config is invalid", "reference_scenario_missing")
    state_by_key: dict[str, LifecycleState] = {}
    for state in config.states:
        if type(state) is not LifecycleState:
            raise _error("config is invalid", "invalid_state")
        _key(state.state_key)
        if (
            state.state_key in state_by_key
            or type(state.state_rank) is not int
            or type(state.priority) is not int
        ):
            raise _error("config is invalid", "invalid_state")
        state_by_key[state.state_key] = state
        _validate_condition(state.condition, config)
        if _uses_state_transition_source(state.condition):
            raise _error("config is invalid", "invalid_state")
    default = state_by_key.get(config.default_state_key)
    unknown = state_by_key.get(config.unknown_state_key)
    if (
        default is None
        or unknown is None
        or default is unknown
        or not default.enabled
        or not unknown.enabled
        or default.terminal
        or unknown.terminal
    ):
        raise _error("config is invalid", "invalid_state")
    if type(config.allowed_transitions) is not tuple:
        raise _error("config is invalid", "invalid_transition")
    transition_pairs: set[tuple[str, str]] = set()
    for pair in config.allowed_transitions:
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or pair[0] not in state_by_key
            or pair[1] not in state_by_key
            or pair[0] == pair[1]
            or state_by_key[pair[0]].terminal
            or pair in transition_pairs
        ):
            raise _error("config is invalid", "invalid_transition")
        transition_pairs.add(pair)
    if (
        type(config.adverse_state_keys) is not tuple
        or type(config.cure_state_keys) is not tuple
    ):
        raise _error("config is invalid", "invalid_transition")
    adverse = set(config.adverse_state_keys)
    cure = set(config.cure_state_keys)
    if (
        len(adverse) != len(config.adverse_state_keys)
        or len(cure) != len(config.cure_state_keys)
        or adverse & cure
        or not adverse.issubset(state_by_key)
        or not cure.issubset(state_by_key)
    ):
        raise _error("config is invalid", "invalid_transition")


def _validate_rule_boundaries(
    rule: EarlyWarningRule,
    config: LifecycleMonitoringConfig,
    *,
    analysis_aware: bool,
) -> None:
    boundaries = (rule.effective_from, rule.expires_at)
    for boundary in boundaries:
        if boundary is None:
            continue
        if type(boundary) not in {datetime, pd.Timestamp}:
            raise _error("config is invalid", "invalid_rule")
        try:
            boundary_aware = _aware(boundary)
        except ValueError:
            raise _error("config is invalid", "invalid_rule") from None
        if boundary_aware != analysis_aware:
            raise _error("config is invalid", "invalid_rule")
        if config.time_zone is not None:
            try:
                zone_name = boundary.tzname()
            except (TypeError, ValueError):
                raise _error("config is invalid", "invalid_rule") from None
            if type(zone_name) is not str or zone_name != config.time_zone:
                raise _error("config is invalid", "invalid_rule")
    if (
        rule.effective_from is not None
        and rule.expires_at is not None
        and not rule.effective_from < rule.expires_at
    ):
        raise _error("config is invalid", "invalid_rule")


def _condition_source_kinds(condition: MonitoringCondition) -> tuple[str, ...]:
    if condition.kind == "atomic":
        return (condition.left_kind,) if condition.left_kind is not None else ()
    return tuple(
        source
        for child in condition.children
        for source in _condition_source_kinds(child)
    )


def _uses_state_transition_source(condition: MonitoringCondition) -> bool:
    """Keep state assignment independent of prior state transition evidence."""
    if condition.kind == "atomic":
        return condition.left_kind in {"prior_state", "state_transition"}
    return any(_uses_state_transition_source(child) for child in condition.children)


def _ordered_positions(
    data: pd.DataFrame, config: LifecycleMonitoringConfig, entities: list[int]
) -> list[int]:
    """Return the contract's entity/time/physical-row processing order."""
    return sorted(
        range(len(data)),
        key=lambda position: (
            entities[position],
            data[config.observation_time_column].iat[position],
            position,
        ),
    )


def _consecutive_positions(
    data: pd.DataFrame, config: LifecycleMonitoringConfig, entities: list[int]
) -> list[object]:
    """Return per-row cadence evidence without changing physical row identity."""
    result: list[object] = [pd.NA] * len(data)
    previous: dict[int, datetime | pd.Timestamp] = {}
    for position in _ordered_positions(data, config, entities):
        entity = entities[position]
        current = data[config.observation_time_column].iat[position]
        prior = previous.get(entity)
        if prior is not None:
            result[position] = (
                config.expected_observation_interval is None
                or current - prior <= config.expected_observation_interval
            )
        previous[entity] = current
    return result


def _warning_resource_gates(
    config: LifecycleMonitoringConfig, entities: list[int]
) -> None:
    """Reject bounded warning outputs before condition evaluation/materialization."""
    enabled_rules = sum(
        rule.enabled for scenario in config.scenarios for rule in scenario.rules
    )
    rule_evaluations = len(entities) * enabled_rules
    if rule_evaluations > 1_000_000:
        raise _error("resource limit exceeded", "rule_evaluations")
    notifications_upper = rule_evaluations
    if notifications_upper > 1_000_000:
        raise _error("resource limit exceeded", "notifications")
    counts: dict[int, int] = {}
    for entity in entities:
        counts[entity] = counts.get(entity, 0) + 1
    declared_rules = sum(len(scenario.rules) for scenario in config.scenarios)
    episodes_upper = sum((count + 1) // 2 for count in counts.values()) * declared_rules
    if episodes_upper > 500_000:
        raise _error("resource limit exceeded", "episodes")


def _state_resource_gates(
    config: LifecycleMonitoringConfig, entities: list[int]
) -> None:
    """Reject bounded state work before state condition evaluation."""
    state_evaluations = len(entities) * sum(state.enabled for state in config.states)
    if state_evaluations > 5_000_000:
        raise _error("resource limit exceeded", "state_evaluations")
    if len(entities) > 100_000:
        raise _error("resource limit exceeded", "state_history_rows")
    if len(entities) > 100_000:
        raise _error("resource limit exceeded", "state_transition_rows")


def _source_error(key: str) -> ValueError:
    return ValueError(f"lifecycle source alignment: {key}")


def _finite_source_value(
    value: object, key: str, *, probability: bool = False
) -> float:
    """Normalize only safe built-in Task 15 or caller score scalars."""
    if not _finite_real(value):
        raise _source_error(key)
    normalized = float(value)
    if probability and not 0.0 <= normalized <= 1.0:
        raise _source_error(key)
    return normalized


def _risk_source_values(
    data: pd.DataFrame,
    config: LifecycleMonitoringConfig,
    risk_validation: BinaryRiskValidationResult | None,
) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Validate and align frozen Task 15 prediction evidence by row position."""
    values = {
        "ranking_score": pd.Series([pd.NA] * len(data), dtype="Float64"),
        "event_probability": pd.Series([pd.NA] * len(data), dtype="Float64"),
    }
    metadata = {
        "score_source": "not_requested",
        "probability_source": "not_requested",
        "task15_evidence_status": "not_provided",
        "task15_evidence_fingerprint": "not_provided",
    }
    if config.ranking_score_column is not None:
        metadata["score_source"] = "dataframe"
        source = data[config.ranking_score_column]
        for position, value in enumerate(source):
            values["ranking_score"].iat[position] = _finite_source_value(
                value, "task15_schema_mismatch"
            )
    if risk_validation is None:
        return values, metadata
    try:
        _validate_binary_risk_validation_result(risk_validation)
    except ValueError as exc:
        raise _source_error("task15_schema_mismatch") from exc
    if risk_validation.input_n_rows != len(data):
        raise _source_error("row_scope_mismatch")
    predictions = risk_validation.predictions
    excluded = risk_validation.excluded_rows
    predicted_positions = predictions["row_position"].tolist()
    excluded_positions = excluded["row_position"].tolist()
    universe = set(range(len(data)))
    if (
        any(
            type(position) is not int
            for position in predicted_positions + excluded_positions
        )
        or len(set(predicted_positions)) != len(predicted_positions)
        or len(set(excluded_positions)) != len(excluded_positions)
        or set(predicted_positions) & set(excluded_positions)
        or set(predicted_positions) | set(excluded_positions) != universe
    ):
        raise _source_error("prediction_scope_mismatch")
    fold_members: dict[int, tuple[int, ...]] = {}
    evaluable_members: dict[int, tuple[int, ...]] = {}
    for _, fold in risk_validation.folds.iterrows():
        fold_id = fold["fold_id"]
        positions = fold["validation_row_positions"]
        evaluable = fold["evaluable_validation_row_positions"]
        if (
            type(fold_id) is not int
            or type(positions) is not tuple
            or type(evaluable) is not tuple
            or fold_id in fold_members
            or any(type(position) is not int for position in positions + evaluable)
            or tuple(sorted(positions)) != positions
            or tuple(sorted(evaluable)) != evaluable
            or not set(evaluable).issubset(positions)
        ):
            raise _source_error("fold_membership_mismatch")
        fold_members[fold_id] = positions
        evaluable_members[fold_id] = evaluable
        time_mode = risk_validation.validation_mode in {"time_holdout", "time_forward"}
        mature = fold["validation_mature_n"]
        if time_mode:
            if mature != fold["evaluable_validation_n"]:
                raise _source_error("time_mode_maturity_mismatch")
        elif not pd.isna(mature) and mature != 0:
            raise _source_error("non_time_maturity_count_nonzero")
    fold_union = set().union(*fold_members.values()) if fold_members else set()
    if fold_union != set(predicted_positions):
        raise _source_error("fold_membership_mismatch")
    score_column = predictions["ranking_score"]
    probability_column = predictions["event_probability"]
    if config.ranking_score_column is not None and score_column.notna().any():
        raise _error("config is invalid", "duplicate_ranking_source")
    if score_column.notna().any():
        metadata["score_source"] = "task15"
    if probability_column.notna().any():
        metadata["probability_source"] = "task15"
    evaluable_count = 0
    digest_rows: list[tuple[int, object, object, object]] = []
    for _, prediction in predictions.iterrows():
        position = prediction["row_position"]
        fold_id = prediction["fold_id"]
        evaluable = prediction["is_evaluable"]
        if (
            type(position) is not int
            or type(fold_id) is not int
            or type(evaluable) is not bool
            or fold_id not in fold_members
            or position not in fold_members[fold_id]
            or evaluable != (position in evaluable_members[fold_id])
        ):
            raise _source_error("evaluable_scope_mismatch")
        evaluable_count += evaluable
        ranking = prediction["ranking_score"]
        probability = prediction["event_probability"]
        if not pd.isna(ranking):
            values["ranking_score"].iat[position] = _finite_source_value(
                ranking, "task15_schema_mismatch"
            )
        if not pd.isna(probability):
            if not evaluable:
                raise _source_error("evaluable_scope_mismatch")
            values["event_probability"].iat[position] = _finite_source_value(
                probability, "task15_schema_mismatch", probability=True
            )
        digest_rows.append((position, ranking, probability, evaluable))
    if evaluable_count != risk_validation.evaluable_n_rows:
        raise _source_error("evaluable_scope_mismatch")
    metadata["task15_evidence_status"] = "provided"
    payload = json.dumps(
        [
            (position, str(score), str(probability), evaluable)
            for position, score, probability, evaluable in digest_rows
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    metadata["task15_evidence_fingerprint"] = sha256(
        payload.encode("ascii")
    ).hexdigest()
    return values, metadata


def _loss_evidence(
    data: pd.DataFrame,
    config: LifecycleMonitoringConfig,
    probabilities: pd.Series,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build private, position-aligned loss readiness facts without aggregation."""
    evidence = pd.DataFrame(
        {
            "row_position": pd.Series(range(len(data)), dtype="int64"),
            "exposure": pd.Series([pd.NA] * len(data), dtype="Float64"),
            "loss_fraction": pd.Series([pd.NA] * len(data), dtype="Float64"),
            "observed_loss": pd.Series([pd.NA] * len(data), dtype="Float64"),
            "observed_mature": pd.Series([pd.NA] * len(data), dtype="boolean"),
            "probability": probabilities.copy(deep=True),
        }
    )
    metadata = {
        "exposure_source": "not_requested",
        "observed_loss_source": "not_requested",
        "loss_fraction_source": "not_requested",
    }

    def numeric(value: object, *, minimum: float, maximum: float | None) -> object:
        if (
            value is None
            or value is pd.NA
            or value is pd.NaT
            or (type(value) in {float, np.float64} and np.isnan(value))
        ):
            return pd.NA
        if not _finite_real(value):
            raise _error("input schema is invalid", "loss_evidence_invalid")
        result = float(value)
        if result < minimum or (maximum is not None and result > maximum):
            raise _error("input schema is invalid", "loss_evidence_invalid")
        return result

    if config.exposure_column is not None:
        metadata["exposure_source"] = "dataframe"
        for position, value in enumerate(data[config.exposure_column]):
            evidence.loc[position, "exposure"] = numeric(value, minimum=0, maximum=None)
    if config.loss_fraction is not None:
        metadata["loss_fraction_source"] = (
            "dataframe" if type(config.loss_fraction) is str else "scalar"
        )
        source = (
            data[config.loss_fraction]
            if type(config.loss_fraction) is str
            else [config.loss_fraction] * len(data)
        )
        for position, value in enumerate(source):
            evidence.loc[position, "loss_fraction"] = numeric(
                value, minimum=0, maximum=1
            )
    if config.observed_loss_column is not None:
        metadata["observed_loss_source"] = "dataframe"
        for position, value in enumerate(data[config.observed_loss_column]):
            evidence.loc[position, "observed_loss"] = numeric(
                value, minimum=0, maximum=None
            )
            if config.observed_loss_is_mature_snapshot:
                evidence.loc[position, "observed_mature"] = True
            elif config.observed_loss_available_time_column is not None:
                available = data[config.observed_loss_available_time_column].iat[
                    position
                ]
                evidence.loc[position, "observed_mature"] = (
                    available is not None
                    and available is not pd.NA
                    and available is not pd.NaT
                    and _datetime(available, "datetime_type_invalid")
                    <= config.analysis_as_of
                )
            else:
                raise _error("config is invalid", "observed_loss_maturity_declaration")
    return evidence, metadata


def _classify_loss_evidence(
    evidence: pd.DataFrame,
    config: LifecycleMonitoringConfig,
    probability_source: str,
) -> pd.DataFrame:
    """Classify independent expected and observed readiness by physical row."""
    classified = pd.DataFrame(
        {
            "row_position": evidence["row_position"].copy(deep=True),
            "expected_status": pd.Series(["available"] * len(evidence), dtype="string"),
            "expected_reason": pd.Series(["computed"] * len(evidence), dtype="string"),
            "observed_status": pd.Series(["available"] * len(evidence), dtype="string"),
            "observed_reason": pd.Series(["computed"] * len(evidence), dtype="string"),
        }
    )
    for position, row in evidence.iterrows():
        if probability_source != "task15":
            expected = ("not_applicable", "source_not_requested")
        elif pd.isna(row["probability"]):
            expected = ("unavailable", "probability_unavailable")
        elif config.exposure_column is None or config.loss_fraction is None:
            expected = ("not_applicable", "source_not_requested")
        elif pd.isna(row["exposure"]):
            expected = ("unavailable", "exposure_unavailable")
        elif pd.isna(row["loss_fraction"]):
            expected = ("unavailable", "exposure_unavailable")
        else:
            expected = ("available", "computed")
        if config.observed_loss_column is None:
            observed = ("not_applicable", "source_not_requested")
        elif not bool(row["observed_mature"]):
            observed = ("not_verifiable", "observed_loss_not_mature")
        elif pd.isna(row["observed_loss"]):
            observed = ("unavailable", "observed_loss_unavailable")
        else:
            observed = ("available", "computed")
        classified.loc[position, ["expected_status", "expected_reason"]] = expected
        classified.loc[position, ["observed_status", "observed_reason"]] = observed
    return classified


def _loss_summary_rows(
    config: LifecycleMonitoringConfig,
    evidence: pd.DataFrame,
    classified: pd.DataFrame,
    observation_history: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Aggregate only the private 4a readiness facts into approved loss rows."""
    records = evidence.merge(
        classified, on="row_position", how="inner", validate="one_to_one"
    )
    monitoring: list[dict[str, object]] = []
    lifecycle: list[dict[str, object]] = []
    scenario_positions = {
        candidate.scenario_key: position
        for position, candidate in enumerate(config.scenarios)
    }

    def state(rows: pd.DataFrame, prefix: str) -> tuple[str, str]:
        if prefix == "exposure":
            if config.exposure_column is None:
                return "not_applicable", "source_not_requested"
            if rows["exposure"].isna().any():
                return "unavailable", "exposure_unavailable"
            return "available", "computed"
        failures = rows.loc[rows[f"{prefix}_status"] != "available"]
        if failures.empty:
            return "available", "computed"
        first = failures.iloc[0]
        return str(first[f"{prefix}_status"]), str(first[f"{prefix}_reason"])

    def add_monitor(
        metric: str,
        rows: pd.DataFrame,
        prefix: str,
        *,
        rate: bool = False,
        scenario: object = pd.NA,
    ) -> None:
        status, reason = state(rows, prefix)
        if prefix == "observed":
            observed_status, observed_reason = state(rows, "observed")
            exposure_status, exposure_reason = state(rows, "exposure")
            if observed_status != "available":
                status, reason = observed_status, observed_reason
            elif exposure_status != "available":
                status, reason = "not_verifiable", exposure_reason
        if status == "available":
            if prefix == "expected":
                amounts = rows["probability"] * rows["exposure"] * rows["loss_fraction"]
            elif metric == "exposure_sum":
                amounts = rows["exposure"]
            else:
                amounts = rows["observed_loss"]
            numerator = float(amounts.sum())
            denominator = float(rows["exposure"].sum()) if rate else pd.NA
            value = numerator / denominator if rate and denominator != 0 else numerator
            if rate and denominator == 0:
                status, reason, value = "undefined", "zero_denominator", pd.NA
        else:
            numerator, denominator, value = pd.NA, pd.NA, pd.NA
        monitoring.append(
            {
                **_summary_row(
                    scope_key="scenario" if scenario is not pd.NA else "overall",
                    scope_position=(
                        scenario_positions[scenario]
                        if scenario is not pd.NA
                        else pd.NA
                    ),
                    scenario_key=scenario,
                    rule_key=pd.NA,
                    metric=metric,
                    numerator=numerator,
                    denominator=denominator,
                    support_n=len(rows),
                    support_unit="observation",
                    unit="fraction" if rate else "exposure_unit",
                    status=status,
                    reason=reason,
                ),
                "metric_value": value,
            }
        )

    all_rows = records.sort_values("row_position", kind="stable")
    add_monitor("exposure_sum", all_rows, "exposure")
    add_monitor("observed_loss_sum", all_rows, "observed")
    add_monitor("observed_loss_rate", all_rows, "observed", rate=True)
    primary = observation_history.set_index("row_position")[
        ["primary_scenario_key", "primary_rule_key"]
    ]
    for scenario in config.scenarios:
        rule_keys = {rule.rule_key for rule in scenario.rules}
        positions = [
            position
            for position, row in primary.iterrows()
            if not pd.isna(row["primary_scenario_key"])
            and row["primary_scenario_key"] == scenario.scenario_key
            and not pd.isna(row["primary_rule_key"])
            and row["primary_rule_key"] in rule_keys
        ]
        scoped = all_rows.loc[all_rows["row_position"].isin(positions)]
        add_monitor(
            "expected_loss_sum", scoped, "expected", scenario=scenario.scenario_key
        )
        add_monitor(
            "expected_loss_rate",
            scoped,
            "expected",
            rate=True,
            scenario=scenario.scenario_key,
        )
    status, reason = state(all_rows, "observed")
    exposure_status, exposure_reason = state(all_rows, "exposure")
    if status == "available" and exposure_status != "available":
        status, reason = "not_verifiable", exposure_reason
    if status == "available":
        numerator = float(all_rows["observed_loss"].sum())
        denominator = float(all_rows["exposure"].sum())
        rate_status = "available" if denominator != 0 else "undefined"
        rate_reason = "computed" if denominator != 0 else "zero_denominator"
    else:
        numerator = denominator = pd.NA
        rate_status = status
        rate_reason = reason
    for metric, rate in (("observed_loss_sum", False), ("observed_loss_rate", True)):
        lifecycle.append(
            _lifecycle_summary_row(
                scope_key="overall",
                scope_position=pd.NA,
                from_state_key=pd.NA,
                to_state_key=pd.NA,
                metric=metric,
                numerator=numerator,
                denominator=denominator if rate else pd.NA,
                support_n=len(all_rows),
                support_unit="observation",
                unit="fraction" if rate else "exposure_unit",
                status=rate_status if rate else status,
                reason=rate_reason if rate else reason,
            )
        )
    return monitoring, lifecycle


def _scenario_loss_summaries(
    config: LifecycleMonitoringConfig,
    evidence: pd.DataFrame,
    classification: pd.DataFrame,
    observation_history: pd.DataFrame,
) -> list[dict[str, object]]:
    """Project frozen loss facts to scenario and scenario-rule monitoring scopes."""
    base_rows, _ = _loss_summary_rows(
        config, evidence, classification, observation_history
    )
    rows: list[dict[str, object]] = []
    non_expected = {"exposure_sum", "observed_loss_sum", "observed_loss_rate"}

    def add(
        source: dict[str, object],
        *,
        scope_key: str,
        scope_position: int,
        scenario_key: str,
        rule_key: object,
    ) -> None:
        row = dict(source)
        row["scope_key"] = scope_key
        row["scope_position"] = scope_position
        row["scenario_key"] = scenario_key
        row["rule_key"] = rule_key
        row["finding_key"] = f"monitoring:{scope_key}:{row['metric']}"
        rows.append(row)

    for ordinal, scenario in enumerate(config.scenarios):
        for source in base_rows:
            if source["metric"] in non_expected:
                add(
                    source,
                    scope_key="scenario",
                    scope_position=ordinal,
                    scenario_key=scenario.scenario_key,
                    rule_key=pd.NA,
                )
        for rule in scenario.rules:
            for source in base_rows:
                if source["metric"] in non_expected:
                    add(
                        source,
                        scope_key="scenario_rule",
                        scope_position=ordinal,
                        scenario_key=scenario.scenario_key,
                        rule_key=rule.rule_key,
                    )
            primary = observation_history.loc[
                (observation_history["primary_scenario_key"] == scenario.scenario_key)
                & (observation_history["primary_rule_key"] == rule.rule_key)
            ]
            positions = primary["row_position"]
            rule_rows, _ = _loss_summary_rows(
                config,
                evidence.loc[evidence["row_position"].isin(positions)],
                classification.loc[
                    classification["row_position"].isin(positions)
                ],
                primary,
            )
            for source in rule_rows:
                if (
                    source["metric"] in {"expected_loss_sum", "expected_loss_rate"}
                    and source["scenario_key"] == scenario.scenario_key
                ):
                    add(
                        source,
                        scope_key="scenario_rule",
                        scope_position=ordinal,
                        scenario_key=scenario.scenario_key,
                        rule_key=rule.rule_key,
                    )
    return rows


def _audit_metadata(
    data: pd.DataFrame, data_audit: DataAuditResult | None
) -> dict[str, str]:
    """Validate optional Task 16 diagnostics without granting business authority."""
    metadata = {
        "task16_evidence_status": "not_provided",
        "task16_config_fingerprint": "not_provided",
        "task16_snapshot_identity": "unverified",
    }
    if data_audit is None:
        return metadata
    if type(data_audit) is not DataAuditResult:
        raise _source_error("task16_schema_mismatch")
    if data_audit.n_rows != len(data):
        raise _source_error("task16_row_scope_mismatch")
    if data_audit.n_columns != len(data.columns):
        raise _source_error("task16_column_scope_mismatch")
    fingerprint = data_audit.config_fingerprint
    if (
        type(fingerprint) is not str
        or len(fingerprint) != 64
        or not all(character in "0123456789abcdef" for character in fingerprint)
    ):
        raise _source_error("task16_schema_mismatch")
    if (
        not isinstance(data_audit.provenance, pd.DataFrame)
        or "provenance_key" not in data_audit.provenance
    ):
        raise _source_error("task16_schema_mismatch")
    profile = data_audit.column_profile
    if not isinstance(profile, pd.DataFrame) or not {
        "side",
        "column",
        "column_position",
    }.issubset(profile.columns):
        raise _source_error("task16_schema_mismatch")
    current = profile.loc[profile["side"] == "current"]
    declared = current.sort_values("column_position", kind="stable")["column"].tolist()
    if declared != list(data.columns):
        raise _source_error("task16_column_scope_mismatch")
    metadata["task16_evidence_status"] = "provided"
    metadata["task16_config_fingerprint"] = fingerprint
    return metadata


def _event_resource_gates(
    config: LifecycleMonitoringConfig,
    entities: list[int],
    event_count: int,
) -> None:
    """Reject event outputs and scans before notification/event matching."""
    if event_count * len(config.scenarios) > 500_000:
        raise _error("resource limit exceeded", "event_match_rows")
    notifications_upper = len(entities) * sum(
        rule.enabled for scenario in config.scenarios for rule in scenario.rules
    )
    operations = len(config.scenarios) * (notifications_upper + event_count)
    if operations > 11_000_000:
        raise _error("resource limit exceeded", "event_match_scan_operations")


def _monitoring_metric_count(scope_key: str) -> int:
    """Count the closed monitoring matrix entries for one approved scope."""
    return sum(
        _monitoring_metric_allowed(scope_key, metric)
        for metric in _COMPARISON_METRIC_ORDER
    )


def _summary_row_projections(
    config: LifecycleMonitoringConfig, facts: _ScopeFacts
) -> dict[str, int]:
    """Calculate final summary bounds with Python integers before materialization."""
    scenario_count = int(len(config.scenarios))
    rule_count = int(sum(len(scenario.rules) for scenario in config.scenarios))
    alert_level_count = int(
        sum(
            len({rule.alert_level for rule in scenario.rules})
            for scenario in config.scenarios
        )
    )
    segment_count = int(len(facts.segment_groups))
    time_count = int(len(facts.time_groups))
    cohort_count = int(len(facts.cohort_groups))
    vintage_count = int(len(facts.vintage_groups))
    state_count = int(len(config.states))
    transition_count = int(len(_TRANSITION_SCOPE_INVENTORY))
    segment_time_count = int(len(facts.segment_time_groups))
    cohort_time_count = int(len(facts.cohort_time_groups))
    vintage_state_count = vintage_count * state_count
    monitoring = (
        _monitoring_metric_count("overall")
        + scenario_count * _monitoring_metric_count("scenario")
        + rule_count * _monitoring_metric_count("scenario_rule")
        + alert_level_count * _monitoring_metric_count("scenario_alert_level")
        + scenario_count
        * (
            segment_count * _monitoring_metric_count("scenario_segment")
            + time_count * _monitoring_metric_count("scenario_time")
            + cohort_count * _monitoring_metric_count("scenario_cohort")
            + vintage_count * _monitoring_metric_count("scenario_vintage")
            + state_count * _monitoring_metric_count("scenario_state")
            + transition_count * _monitoring_metric_count("scenario_transition")
        )
        + segment_time_count * _monitoring_metric_count("segment_time")
        + cohort_time_count * _monitoring_metric_count("cohort_time")
        + vintage_state_count * _monitoring_metric_count("vintage_state")
    )
    transition_pair_count = state_count * state_count
    lifecycle = (
        17
        + 7 * state_count
        + 8 * transition_pair_count
        + 17 * (segment_time_count + cohort_time_count)
        + 7 * vintage_state_count
    )
    reference = next(
        scenario
        for scenario in config.scenarios
        if scenario.scenario_key == config.reference_scenario_key
    )
    reference_rules = {rule.rule_key for rule in reference.rules}
    reference_levels = {rule.alert_level for rule in reference.rules}
    comparison = 0
    for scenario in config.scenarios:
        if scenario.scenario_key == config.reference_scenario_key:
            continue
        shared_rules = len(
            reference_rules.intersection(rule.rule_key for rule in scenario.rules)
        )
        shared_levels = len(
            reference_levels.intersection(
                rule.alert_level for rule in scenario.rules
            )
        )
        comparison += (
            _monitoring_metric_count("scenario")
            + shared_rules * _monitoring_metric_count("scenario_rule")
            + shared_levels * _monitoring_metric_count("scenario_alert_level")
            + segment_count * _monitoring_metric_count("scenario_segment")
            + time_count * _monitoring_metric_count("scenario_time")
            + cohort_count * _monitoring_metric_count("scenario_cohort")
            + vintage_count * _monitoring_metric_count("scenario_vintage")
            + state_count * _monitoring_metric_count("scenario_state")
            + transition_count * _monitoring_metric_count("scenario_transition")
        )
    return {
        "monitoring_summary_rows": int(monitoring),
        "lifecycle_summary_rows": int(lifecycle),
        "scenario_comparison_rows": int(comparison),
    }


def _summary_projection_gates(projections: dict[str, int]) -> None:
    """Enforce final summary gates in their frozen key precedence."""
    for key in (
        "monitoring_summary_rows",
        "lifecycle_summary_rows",
        "scenario_comparison_rows",
    ):
        if projections[key] > 200_000:
            raise _error("resource limit exceeded", key)


def _summary_resource_gates(
    config: LifecycleMonitoringConfig, facts: _ScopeFacts
) -> dict[str, int]:
    """Project and reject final summary outputs before rule/state evaluation."""
    projections = _summary_row_projections(config, facts)
    _summary_projection_gates(projections)
    return projections


def _declared_events(
    data: pd.DataFrame,
    config: LifecycleMonitoringConfig,
    entities: list[int],
    aware: bool,
) -> list[dict[str, object]]:
    """Return visible distinct future events without retaining raw event payloads."""
    if config.event_time_column is None:
        return []
    events: dict[tuple[int, datetime | pd.Timestamp], dict[str, object]] = {}
    for position in range(len(data)):
        value = data[config.event_time_column].iat[position]
        if value is None or value is pd.NA or value is pd.NaT:
            continue
        event_time = _datetime(value, "datetime_type_invalid")
        if _aware(event_time) != aware:
            raise _error("input schema is invalid", "datetime_awareness_mismatch")
        if event_time > config.analysis_as_of:
            continue
        key = (entities[position], event_time)
        prior = events.get(key)
        if prior is None:
            events[key] = {
                "entity_position": entities[position],
                "event_time": event_time,
                "event_row_position": position,
                "duplicate_source_row_count": 0,
            }
        else:
            prior["duplicate_source_row_count"] = (
                int(prior["duplicate_source_row_count"]) + 1
            )
    ordered = sorted(
        events.values(),
        key=lambda event: (
            int(event["entity_position"]),
            event["event_time"],
            int(event["event_row_position"]),
        ),
    )
    ordinal_by_entity: dict[int, int] = {}
    for event in ordered:
        entity = int(event["entity_position"])
        event["event_ordinal"] = ordinal_by_entity.get(entity, 0)
        ordinal_by_entity[entity] = int(event["event_ordinal"]) + 1
    return ordered


def _within_horizon(
    start: datetime | pd.Timestamp,
    event_time: datetime | pd.Timestamp,
    config: LifecycleMonitoringConfig,
) -> bool:
    """Apply the frozen open-start and configurable closed-end event horizon."""
    assert config.prediction_horizon is not None
    end = start + config.prediction_horizon
    return start < event_time and (
        event_time <= end if config.horizon_end_inclusive else event_time < end
    )


def _event_evidence(
    data: pd.DataFrame,
    config: LifecycleMonitoringConfig,
    entities: list[int],
    events: list[dict[str, object]],
    notifications: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[object], list[object]]:
    """Populate event matches with per-entity/scenario linear horizon scans."""
    if config.event_time_column is None:
        return [], ["not_applicable"] * len(data), [pd.NA] * len(data)
    assert config.prediction_horizon is not None
    event_times_by_entity: dict[int, list[datetime | pd.Timestamp]] = {}
    for event in events:
        event_times_by_entity.setdefault(int(event["entity_position"]), []).append(
            event["event_time"]
        )
    observation_maturity: list[object] = []
    observation_event: list[object] = []
    for position in range(len(data)):
        time = data[config.observation_time_column].iat[position]
        mature = time + config.prediction_horizon <= config.analysis_as_of
        observation_maturity.append("mature" if mature else "immature")
        observation_event.append(
            any(
                _within_horizon(time, event_time, config)
                for event_time in event_times_by_entity.get(entities[position], [])
            )
        )
    event_mature: dict[tuple[int, int], bool] = {}
    for event in events:
        entity = int(event["entity_position"])
        event_time = event["event_time"]
        event_mature[(entity, int(event["event_ordinal"]))] = any(
            observation_maturity[position] == "mature"
            and _within_horizon(
                data[config.observation_time_column].iat[position], event_time, config
            )
            for position in range(len(data))
            if entities[position] == entity
        )
    for note in notifications:
        mature = (
            note["notification_time"] + config.prediction_horizon
            <= config.analysis_as_of
        )
        note["maturity_status"] = "mature" if mature else "immature"
        note["first_matched_event_ordinal"] = pd.NA
        note["matched_event_count"] = 0

    matches: list[dict[str, object]] = []
    events_by_entity: dict[int, list[dict[str, object]]] = {}
    notes_by_entity_scenario: dict[tuple[int, str], list[dict[str, object]]] = {}
    for event in events:
        events_by_entity.setdefault(int(event["entity_position"]), []).append(event)
    for note in notifications:
        notes_by_entity_scenario.setdefault(
            (int(note["entity_position"]), str(note["scenario_key"])), []
        ).append(note)
    for scenario in config.scenarios:
        for entity, entity_events in events_by_entity.items():
            notes = sorted(
                notes_by_entity_scenario.get((entity, scenario.scenario_key), []),
                key=lambda note: (
                    note["notification_time"],
                    str(note["scenario_key"]),
                    int(note["entity_position"]),
                    str(note["rule_key"]),
                    int(note["notification_ordinal"]),
                ),
            )
            active: deque[dict[str, object]] = deque()
            next_note = 0
            for event in entity_events:
                event_time = event["event_time"]
                while (
                    next_note < len(notes)
                    and notes[next_note]["notification_time"] < event_time
                ):
                    candidate = notes[next_note]
                    if candidate["maturity_status"] == "mature":
                        active.append(candidate)
                    next_note += 1
                while active and not _within_horizon(
                    active[0]["notification_time"], event_time, config
                ):
                    active.popleft()
                mature = event_mature[(entity, int(event["event_ordinal"]))]
                owner = active[0] if mature and active else None
                if owner is not None:
                    owner["matched_event_count"] = int(owner["matched_event_count"]) + 1
                    if pd.isna(owner["first_matched_event_ordinal"]):
                        owner["first_matched_event_ordinal"] = event["event_ordinal"]
                matches.append(
                    {
                        "scenario_key": scenario.scenario_key,
                        "entity_position": entity,
                        "event_ordinal": event["event_ordinal"],
                        "event_row_position": event["event_row_position"],
                        "event_time": event_time,
                        "duplicate_source_row_count": event[
                            "duplicate_source_row_count"
                        ],
                        "event_status": "mature" if mature else "not_eligible",
                        "match_status": "captured"
                        if owner is not None
                        else "not_captured",
                        "captured": owner is not None,
                        "capturing_rule_key": pd.NA
                        if owner is None
                        else owner["rule_key"],
                        "capturing_episode_ordinal": pd.NA
                        if owner is None
                        else owner["episode_ordinal"],
                        "capturing_notification_ordinal": (
                            pd.NA if owner is None else owner["notification_ordinal"]
                        ),
                        "capturing_notification_row_position": (
                            pd.NA if owner is None else owner["row_position"]
                        ),
                        "notification_time": pd.NaT
                        if owner is None
                        else owner["notification_time"],
                        "lead_time_seconds": (
                            pd.NA
                            if owner is None
                            else (
                                event_time - owner["notification_time"]
                            ).total_seconds()
                        ),
                        "candidate_notification_count": len(active) if mature else 0,
                        "status": "available",
                        "reason": "event_captured" if owner is not None else "computed",
                        "finding_key": (
                            f"monitoring:event:{scenario.scenario_key}:"
                            f"{entity}:{event['event_ordinal']}"
                        ),
                    }
                )
    matches.sort(
        key=lambda row: (
            int(row["entity_position"]),
            row["event_time"],
            int(row["event_ordinal"]),
            str(row["scenario_key"]),
        )
    )
    return matches, observation_maturity, observation_event


def _summary_row(
    *,
    scope_key: str,
    scope_position: object,
    scenario_key: object,
    rule_key: object,
    metric: str,
    numerator: object,
    denominator: object,
    support_n: int,
    support_unit: str,
    unit: str,
    status: str = "available",
    reason: str = "computed",
    mature_n: int = 0,
    censored_n: int = 0,
) -> dict[str, object]:
    """Construct one frozen monitoring-summary schema row."""
    if status == "available":
        if unit in {"fraction", "count/entity"} or metric.endswith("_mean"):
            metric_value = float(numerator) / float(denominator)
        elif metric.endswith("_median"):
            metric_value = numerator
        else:
            metric_value = numerator
    else:
        metric_value = pd.NA
    return {
        "scope_key": scope_key,
        "scope_position": scope_position,
        "scenario_key": scenario_key,
        "rule_key": rule_key,
        "metric": metric,
        "metric_value": metric_value,
        "numerator": numerator,
        "denominator": denominator,
        "support_n": support_n,
        "support_unit": support_unit,
        "mature_n": mature_n,
        "censored_n": censored_n,
        "unit": unit,
        "status": status,
        "reason": reason,
        "finding_key": f"monitoring:{scope_key}:{metric}",
    }


def _monitoring_summaries(
    config: LifecycleMonitoringConfig,
    rule_evaluations: list[dict[str, object]],
    notifications: list[dict[str, object]],
    episodes: list[dict[str, object]],
    event_matches: list[dict[str, object]],
    observation_history: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate completed warning, episode, and event facts without recomputation."""
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, object, object, object, list[dict[str, object]]]] = [
        ("overall", pd.NA, pd.NA, pd.NA, rule_evaluations)
    ]
    for ordinal, scenario in enumerate(config.scenarios):
        evaluations = [
            row
            for row in rule_evaluations
            if row["scenario_key"] == scenario.scenario_key
        ]
        scopes.append(("scenario", ordinal, scenario.scenario_key, pd.NA, evaluations))
        for rule in scenario.rules:
            scopes.append(
                (
                    "scenario_rule",
                    ordinal,
                    scenario.scenario_key,
                    rule.rule_key,
                    [row for row in evaluations if row["rule_key"] == rule.rule_key],
                )
            )
    for scope_key, scope_position, scenario_key, rule_key, evaluations in scopes:
        active = [row for row in evaluations if row["path_status"] == "evaluated"]
        current_hits = [row for row in active if row["truth"] == "true"]
        hit_positions = {int(row["row_position"]) for row in current_hits}
        evaluable_positions = {int(row["row_position"]) for row in active}
        scoped_entities = {int(row["entity_position"]) for row in active}
        warned_entities = {int(row["entity_position"]) for row in current_hits}
        persistent = [row for row in current_hits if row["episode_status"] == "active"]
        scoped_notes = [
            note
            for note in notifications
            if scenario_key is pd.NA or note["scenario_key"] == scenario_key
            if rule_key is pd.NA or note["rule_key"] == rule_key
        ]
        scoped_episodes = [
            episode
            for episode in episodes
            if scenario_key is pd.NA or episode["scenario_key"] == scenario_key
            if rule_key is pd.NA or episode["rule_key"] == rule_key
        ]
        durations = [float(episode["duration_seconds"]) for episode in scoped_episodes]
        for metric, numerator, denominator, support, support_unit, unit in (
            (
                "warning_hit_count",
                len(current_hits),
                pd.NA,
                len(active),
                "rule_evaluation",
                "count",
            ),
            (
                "warning_observation_rate",
                len(hit_positions),
                len(evaluable_positions),
                len(evaluable_positions),
                "observation",
                "fraction",
            ),
            (
                "warned_entity_count",
                len(warned_entities),
                pd.NA,
                len(scoped_entities),
                "entity",
                "count",
            ),
            (
                "warned_entity_rate",
                len(warned_entities),
                len(scoped_entities),
                len(scoped_entities),
                "entity",
                "fraction",
            ),
            (
                "persistent_warning_count",
                len({int(row["row_position"]) for row in persistent}),
                pd.NA,
                len(evaluable_positions),
                "observation",
                "count",
            ),
            (
                "persistent_warning_rate",
                len({int(row["row_position"]) for row in persistent}),
                len(evaluable_positions),
                len(evaluable_positions),
                "observation",
                "fraction",
            ),
            (
                "notification_count",
                len(scoped_notes),
                pd.NA,
                len(scoped_notes),
                "notification",
                "count",
            ),
            (
                "notifications_per_entity",
                len(scoped_notes),
                len(scoped_entities),
                len(scoped_entities),
                "entity",
                "count/entity",
            ),
            (
                "episode_count",
                len(scoped_episodes),
                pd.NA,
                len(scoped_episodes),
                "episode",
                "count",
            ),
            (
                "open_episode_count",
                sum(bool(row["is_unresolved"]) for row in scoped_episodes),
                pd.NA,
                len(scoped_episodes),
                "episode",
                "count",
            ),
            (
                "resolved_episode_count",
                sum(not bool(row["is_unresolved"]) for row in scoped_episodes),
                pd.NA,
                len(scoped_episodes),
                "episode",
                "count",
            ),
            (
                "episode_duration_mean",
                sum(durations),
                len(durations),
                len(durations),
                "episode",
                "seconds",
            ),
            (
                "episode_duration_median",
                float(np.median(durations)) if durations else pd.NA,
                len(durations),
                len(durations),
                "episode",
                "seconds",
            ),
        ):
            zero = denominator is not pd.NA and denominator == 0
            rows.append(
                _summary_row(
                    scope_key=scope_key,
                    scope_position=scope_position,
                    scenario_key=scenario_key,
                    rule_key=rule_key,
                    metric=metric,
                    numerator=numerator,
                    denominator=denominator,
                    support_n=support,
                    support_unit=support_unit,
                    unit=unit,
                    status="undefined" if zero else "available",
                    reason="zero_denominator" if zero else "computed",
                )
            )
        if scope_key == "scenario":
            hits_by_position: dict[int, list[dict[str, object]]] = {}
            for hit in current_hits:
                hits_by_position.setdefault(int(hit["row_position"]), []).append(hit)
            overlap = sum(len(hits) >= 2 for hits in hits_by_position.values())
            conflict = sum(
                sum(
                    int(hit["alert_rank"])
                    == max(int(candidate["alert_rank"]) for candidate in hits)
                    for hit in hits
                )
                >= 2
                for hits in hits_by_position.values()
            )
            for metric, numerator in (
                ("overlap_count", overlap),
                ("conflict_count", conflict),
            ):
                rows.append(
                    _summary_row(
                        scope_key=scope_key,
                        scope_position=scope_position,
                        scenario_key=scenario_key,
                        rule_key=pd.NA,
                        metric=metric,
                        numerator=numerator,
                        denominator=pd.NA,
                        support_n=len(evaluable_positions),
                        support_unit="observation",
                        unit="count",
                    )
                )
        scoped_matches = [
            match
            for match in event_matches
            if scenario_key is pd.NA or match["scenario_key"] == scenario_key
        ]
        mature_events = [
            match for match in scoped_matches if match["event_status"] == "mature"
        ]
        captured_events = [match for match in mature_events if bool(match["captured"])]
        mature_notes = [
            note for note in scoped_notes if note["maturity_status"] == "mature"
        ]
        censored_notes = [
            note for note in scoped_notes if note["maturity_status"] != "mature"
        ]
        matched_notes = [
            note for note in mature_notes if int(note["matched_event_count"]) > 0
        ]
        unmatched_notes = [
            note for note in mature_notes if int(note["matched_event_count"]) == 0
        ]
        scoped_observations = observation_history
        if rule_key is not pd.NA:
            scoped_observations = [
                row
                for row in scoped_observations
                if row["primary_rule_key"] == rule_key
            ]
        scenario_rule_keys = {
            rule.rule_key
            for scenario in config.scenarios
            if scenario_key is pd.NA or scenario.scenario_key == scenario_key
            for rule in scenario.rules
        }
        mature_observations = [
            row for row in scoped_observations if row["maturity_status"] == "mature"
        ]
        mature_no_event = [
            row for row in mature_observations if row["event_within_horizon"] is False
        ]
        mature_warned = [
            row
            for row in mature_observations
            if row["primary_rule_key"] in scenario_rule_keys
        ]
        mature_warned_event = [
            row for row in mature_warned if row["event_within_horizon"] is True
        ]
        notification_positions = {int(note["row_position"]) for note in mature_notes}
        false_positive_positions = {
            int(row["row_position"])
            for row in mature_no_event
            if int(row["row_position"]) in notification_positions
        }
        lead_times = [float(match["lead_time_seconds"]) for match in captured_events]
        if config.event_time_column is None:
            for metric, support_unit, unit in (
                ("captured_event_count", "event", "count"),
                ("event_recall", "event", "fraction"),
                ("notification_precision", "notification", "fraction"),
                ("false_alert_share", "notification", "fraction"),
                ("false_positive_rate", "observation", "fraction"),
                ("lead_time_mean", "event", "seconds"),
                ("lead_time_median", "event", "seconds"),
                ("warning_to_event_rate", "observation", "fraction"),
            ):
                rows.append(
                    _summary_row(
                        scope_key=scope_key,
                        scope_position=scope_position,
                        scenario_key=scenario_key,
                        rule_key=rule_key,
                        metric=metric,
                        numerator=pd.NA,
                        denominator=pd.NA,
                        support_n=0,
                        support_unit=support_unit,
                        unit=unit,
                        status="not_applicable",
                        reason="source_not_requested",
                    )
                )
            continue
        censored_events = len(scoped_matches) - len(mature_events)
        event_metrics = (
            (
                "captured_event_count",
                len(captured_events),
                pd.NA,
                len(mature_events),
                "event",
                "count",
                len(mature_events),
                censored_events,
            ),
            (
                "event_recall",
                len(captured_events),
                len(mature_events),
                len(mature_events),
                "event",
                "fraction",
                len(mature_events),
                censored_events,
            ),
            (
                "notification_precision",
                len(matched_notes),
                len(mature_notes),
                len(mature_notes),
                "notification",
                "fraction",
                len(mature_notes),
                len(censored_notes),
            ),
            (
                "false_alert_share",
                len(unmatched_notes),
                len(mature_notes),
                len(mature_notes),
                "notification",
                "fraction",
                len(mature_notes),
                len(censored_notes),
            ),
            (
                "false_positive_rate",
                len(false_positive_positions),
                len(mature_no_event),
                len(mature_no_event),
                "observation",
                "fraction",
                len(mature_no_event),
                len(scoped_observations) - len(mature_observations),
            ),
            (
                "lead_time_mean",
                sum(lead_times),
                len(lead_times),
                len(lead_times),
                "event",
                "seconds",
                len(lead_times),
                censored_events,
            ),
            (
                "lead_time_median",
                float(np.median(lead_times)) if lead_times else pd.NA,
                len(lead_times),
                len(lead_times),
                "event",
                "seconds",
                len(lead_times),
                censored_events,
            ),
            (
                "warning_to_event_rate",
                len(mature_warned_event),
                len(mature_warned),
                len(mature_warned),
                "observation",
                "fraction",
                len(mature_warned),
                len(scoped_observations) - len(mature_observations),
            ),
        )
        for (
            metric,
            numerator,
            denominator,
            support,
            support_unit,
            unit,
            mature_n,
            censored_n,
        ) in event_metrics:
            zero = denominator is not pd.NA and denominator == 0
            rows.append(
                _summary_row(
                    scope_key=scope_key,
                    scope_position=scope_position,
                    scenario_key=scenario_key,
                    rule_key=rule_key,
                    metric=metric,
                    numerator=numerator,
                    denominator=denominator,
                    support_n=support,
                    support_unit=support_unit,
                    unit=unit,
                    status="undefined" if zero else "available",
                    reason="zero_denominator" if zero else "computed",
                    mature_n=mature_n,
                    censored_n=censored_n,
                )
            )
    return rows


def _lifecycle_summary_row(
    *,
    scope_key: str,
    scope_position: object,
    from_state_key: object,
    to_state_key: object,
    metric: str,
    numerator: object,
    denominator: object,
    support_n: int,
    support_unit: str,
    unit: str,
    status: str = "available",
    reason: str = "computed",
) -> dict[str, object]:
    """Construct one frozen lifecycle-summary row from existing state facts."""
    if status == "available":
        if unit == "fraction" or metric.endswith("_mean"):
            metric_value = float(numerator) / float(denominator)
        else:
            metric_value = numerator
    else:
        metric_value = pd.NA
    return {
        "scope_key": scope_key,
        "scope_position": scope_position,
        "from_state_key": from_state_key,
        "to_state_key": to_state_key,
        "metric": metric,
        "metric_value": metric_value,
        "numerator": numerator,
        "denominator": denominator,
        "support_n": support_n,
        "support_unit": support_unit,
        "unit": unit,
        "status": status,
        "reason": reason,
        "finding_key": f"monitoring:lifecycle:{scope_key}:{metric}",
    }


def _lifecycle_summaries(
    config: LifecycleMonitoringConfig,
    state_history: list[dict[str, object]],
    state_transitions: list[dict[str, object]],
    *,
    durations: dict[int, float] | None = None,
) -> list[dict[str, object]]:
    """Aggregate frozen effective-state and transition facts without replaying them."""
    rows: list[dict[str, object]] = []
    verifiable_history = [row for row in state_history if row["status"] == "available"]
    evaluable_entities = {int(row["entity_position"]) for row in verifiable_history}
    ordered_states = sorted(
        config.states, key=lambda state: (state.state_rank, state.state_key)
    )
    state_scopes: list[tuple[str, object, object, list[dict[str, object]]]] = [
        ("overall", pd.NA, pd.NA, verifiable_history)
    ]
    for position, state in enumerate(ordered_states):
        state_scopes.append(
            (
                "state",
                position,
                state.state_key,
                [
                    row
                    for row in verifiable_history
                    if row["effective_state_key"] == state.state_key
                ],
            )
        )
    if durations is None:
        durations = _state_durations(config, verifiable_history)
    for scope_key, scope_position, state_key, scoped_history in state_scopes:
        scoped_entities = {int(row["entity_position"]) for row in scoped_history}
        scoped_durations = [
            durations[int(row["row_position"])] for row in scoped_history
        ]
        denominator = len(verifiable_history)
        entity_denominator = len(evaluable_entities)
        for metric, numerator, denominator_value, support, support_unit, unit in (
            (
                "state_observation_count",
                len(scoped_history),
                pd.NA,
                len(scoped_history),
                "observation",
                "count",
            ),
            (
                "state_observation_rate",
                len(scoped_history),
                denominator,
                denominator,
                "observation",
                "fraction",
            ),
            (
                "entity_state_count",
                len(scoped_entities),
                pd.NA,
                len(scoped_entities),
                "entity",
                "count",
            ),
            (
                "entity_state_rate",
                len(scoped_entities),
                entity_denominator,
                entity_denominator,
                "entity",
                "fraction",
            ),
            (
                "time_in_state_mean",
                sum(scoped_durations),
                len(scoped_durations),
                len(scoped_durations),
                "observation",
                "seconds",
            ),
        ):
            zero = denominator_value is not pd.NA and denominator_value == 0
            rows.append(
                _lifecycle_summary_row(
                    scope_key=scope_key,
                    scope_position=scope_position,
                    from_state_key=pd.NA,
                    to_state_key=state_key,
                    metric=metric,
                    numerator=numerator,
                    denominator=denominator_value,
                    support_n=support,
                    support_unit=support_unit,
                    unit=unit,
                    status="undefined" if zero else "available",
                    reason="zero_denominator" if zero else "computed",
                )
            )
    valid = [
        row for row in state_transitions if row["transition_kind"] in {"stay", "change"}
    ]
    adverse = set(config.adverse_state_keys)
    valid_from_adverse = [row for row in valid if row["from_state_key"] in adverse]
    transition_scopes: list[
        tuple[str, object, object, object, list[dict[str, object]]]
    ] = [("overall", pd.NA, pd.NA, pd.NA, valid)]
    pairs = sorted(
        {(row["from_state_key"], row["effective_to_state_key"]) for row in valid}
    )
    for position, (from_key, to_key) in enumerate(pairs):
        transition_scopes.append(
            (
                "transition",
                position,
                from_key,
                to_key,
                [
                    row
                    for row in valid
                    if row["from_state_key"] == from_key
                    and row["effective_to_state_key"] == to_key
                ],
            )
        )
    for scope_key, scope_position, from_key, to_key, scoped in transition_scopes:
        valid_denominator = len(valid)
        for metric, numerator, denominator, support, unit in (
            ("transition_count", len(scoped), pd.NA, len(scoped), "count"),
            (
                "transition_rate",
                len(scoped),
                valid_denominator,
                valid_denominator,
                "fraction",
            ),
            (
                "roll_forward_count",
                sum(row["transition_direction"] == "roll_forward" for row in scoped),
                pd.NA,
                len(scoped),
                "count",
            ),
            (
                "roll_forward_rate",
                sum(row["transition_direction"] == "roll_forward" for row in scoped),
                valid_denominator,
                valid_denominator,
                "fraction",
            ),
            (
                "roll_back_count",
                sum(row["transition_direction"] == "roll_back" for row in scoped),
                pd.NA,
                len(scoped),
                "count",
            ),
            (
                "roll_back_rate",
                sum(row["transition_direction"] == "roll_back" for row in scoped),
                valid_denominator,
                valid_denominator,
                "fraction",
            ),
            (
                "cure_count",
                sum(bool(row["is_cure"]) for row in scoped),
                pd.NA,
                len(scoped),
                "count",
            ),
            (
                "cure_rate",
                sum(
                    bool(row["is_cure"])
                    for row in scoped
                    if row["from_state_key"] in adverse
                ),
                len(valid_from_adverse),
                len(valid_from_adverse),
                "fraction",
            ),
        ):
            zero = denominator is not pd.NA and denominator == 0
            rows.append(
                _lifecycle_summary_row(
                    scope_key=scope_key,
                    scope_position=scope_position,
                    from_state_key=from_key,
                    to_state_key=to_key,
                    metric=metric,
                    numerator=numerator,
                    denominator=denominator,
                    support_n=support,
                    support_unit="transition",
                    unit=unit,
                    status="undefined" if zero else "available",
                    reason="zero_denominator" if zero else "computed",
                )
            )
    for metric, kind in (("entry_count", "entry"), ("reentry_count", "reentry")):
        count = sum(row["transition_kind"] == kind for row in state_transitions)
        rows.append(
            _lifecycle_summary_row(
                scope_key="overall",
                scope_position=pd.NA,
                from_state_key=pd.NA,
                to_state_key=pd.NA,
                metric=metric,
                numerator=count,
                denominator=pd.NA,
                support_n=count,
                support_unit="transition",
                unit="count",
            )
        )
    return rows


def _state_durations(
    config: LifecycleMonitoringConfig, state_history: list[dict[str, object]]
) -> dict[int, float]:
    """Compute frozen state durations once so group filters cannot alter them."""
    durations: dict[int, float] = {}
    by_entity: dict[int, list[dict[str, object]]] = {}
    for row in state_history:
        by_entity.setdefault(int(row["entity_position"]), []).append(row)
    for entity_rows in by_entity.values():
        entity_rows.sort(
            key=lambda row: (row["observation_time"], int(row["row_position"]))
        )
        for index, row in enumerate(entity_rows):
            end = (
                entity_rows[index + 1]["observation_time"]
                if index + 1 < len(entity_rows)
                else config.analysis_as_of
            )
            durations[int(row["row_position"])] = (
                end - row["observation_time"]
            ).total_seconds()
    return durations


_EVENT_METRICS = frozenset(
    {
        "captured_event_count",
        "event_recall",
        "notification_precision",
        "false_alert_share",
        "false_positive_rate",
        "lead_time_mean",
        "lead_time_median",
        "warning_to_event_rate",
    }
)

_WARNING_METRICS = (
    "warning_hit_count",
    "warning_observation_rate",
    "warned_entity_count",
    "warned_entity_rate",
    "persistent_warning_count",
    "persistent_warning_rate",
    "notification_count",
    "notifications_per_entity",
    "episode_count",
    "open_episode_count",
    "resolved_episode_count",
    "episode_duration_mean",
    "episode_duration_median",
)
_OVERLAP_METRICS = frozenset({"overlap_count", "conflict_count"})
_LOSS_METRICS = frozenset(
    {
        "exposure_sum",
        "expected_loss_sum",
        "expected_loss_rate",
        "observed_loss_sum",
        "observed_loss_rate",
    }
)
_COMPARISON_SCOPES = (
    "scenario",
    "scenario_rule",
    "scenario_alert_level",
    "scenario_segment",
    "scenario_time",
    "scenario_cohort",
    "scenario_vintage",
    "scenario_state",
    "scenario_transition",
)
_COMPARISON_SCOPE_SET = frozenset(_COMPARISON_SCOPES)
_EVENT_SCOPES = frozenset(
    {
        "overall",
        "scenario",
        "scenario_rule",
        "scenario_segment",
        "scenario_time",
        "scenario_cohort",
        "scenario_vintage",
    }
)
_OVERLAP_SCOPES = frozenset({"scenario", "scenario_segment", "scenario_time"})
_COMPARISON_METRIC_ORDER = (
    "warning_hit_count",
    "warning_observation_rate",
    "warned_entity_count",
    "warned_entity_rate",
    "persistent_warning_count",
    "persistent_warning_rate",
    "notification_count",
    "notifications_per_entity",
    "overlap_count",
    "conflict_count",
    "episode_count",
    "open_episode_count",
    "resolved_episode_count",
    "episode_duration_mean",
    "episode_duration_median",
    "captured_event_count",
    "event_recall",
    "notification_precision",
    "false_alert_share",
    "false_positive_rate",
    "lead_time_mean",
    "lead_time_median",
    "warning_to_event_rate",
    "exposure_sum",
    "expected_loss_sum",
    "expected_loss_rate",
    "observed_loss_sum",
    "observed_loss_rate",
)
_COMPARISON_METRIC_ORDER_INDEX = {
    metric: position for position, metric in enumerate(_COMPARISON_METRIC_ORDER)
}
_COMPARISON_SCOPE_ORDER_INDEX = {
    scope: position for position, scope in enumerate(_COMPARISON_SCOPES)
}


def _monitoring_metric_allowed(scope_key: str, metric: str) -> bool:
    """Apply the frozen monitoring metric × scope matrix without expansion."""
    if metric in _WARNING_METRICS:
        return True
    if metric in _OVERLAP_METRICS:
        return scope_key in _OVERLAP_SCOPES
    if metric in _EVENT_METRICS:
        return scope_key in _EVENT_SCOPES
    if metric in {"exposure_sum", "observed_loss_sum", "observed_loss_rate"}:
        return True
    if metric in {"expected_loss_sum", "expected_loss_rate"}:
        return scope_key.startswith("scenario")
    return False


def _normalized_comparison_identity(
    row: dict[str, object],
) -> tuple[str, str, int | None, str | None] | None:
    """Return the AM-04 closed comparison identity for one source summary row."""
    scope_key = row["scope_key"]
    metric = row["metric"]
    if (
        scope_key not in _COMPARISON_SCOPE_SET
        or metric not in _COMPARISON_METRIC_ORDER_INDEX
        or not _monitoring_metric_allowed(scope_key, metric)
    ):
        return None
    if scope_key in {"scenario", "scenario_rule"}:
        scope_position: int | None = None
    else:
        source_position = row["scope_position"]
        if pd.isna(source_position):
            return None
        scope_position = int(source_position)
    rule_key = row["rule_key"] if scope_key == "scenario_rule" else pd.NA
    if scope_key == "scenario_rule" and pd.isna(rule_key):
        return None
    return (
        metric,
        scope_key,
        scope_position,
        None if pd.isna(rule_key) else rule_key,
    )


_COMPARISON_FAILURE_PRECEDENCE = (
    "not_applicable",
    "inactive",
    "censored",
    "not_verifiable",
    "unavailable",
    "undefined",
)


def _comparison_failure(
    reference: dict[str, object], comparator: dict[str, object]
) -> tuple[str, str]:
    """Choose a fixed source-status precedence without traversal dependence."""
    for status in _COMPARISON_FAILURE_PRECEDENCE:
        for source in (reference, comparator):
            if source["status"] == status:
                return status, source["reason"]
    return "not_verifiable", "support_not_comparable"


def _source_values_match(
    reference: dict[str, object], comparator: dict[str, object]
) -> bool:
    """Require the frozen support universe before deriving a comparison delta."""
    return all(
        reference[column] == comparator[column]
        for column in ("support_n", "support_unit", "mature_n", "censored_n", "unit")
    )


def _numeric_value(value: object) -> float | None:
    """Preserve valid zero values while treating only nullable values as missing."""
    return None if pd.isna(value) else float(value)


def _difference(left: object, right: object) -> object:
    """Return right minus left only when both source quantities are present."""
    left_value = _numeric_value(left)
    right_value = _numeric_value(right)
    if left_value is None or right_value is None:
        return pd.NA
    return right_value - left_value


def _shared_value(left: object, right: object) -> object:
    """Retain an exact common scalar, otherwise expose no fabricated common value."""
    if pd.isna(left) and pd.isna(right):
        return pd.NA
    if pd.isna(left) or pd.isna(right):
        return pd.NA
    return left if left == right else pd.NA


def _scenario_comparison_rows(
    config: LifecycleMonitoringConfig, monitoring_summary: pd.DataFrame
) -> list[dict[str, object]]:
    """Project frozen monitoring-summary facts into AM-04 scenario comparisons."""
    records = monitoring_summary.to_dict("records")
    reference_key = config.reference_scenario_key
    scenario_order = {
        scenario.scenario_key: position
        for position, scenario in enumerate(config.scenarios)
    }
    reference_scenario = next(
        scenario
        for scenario in config.scenarios
        if scenario.scenario_key == reference_key
    )
    reference_priority = {
        rule.rule_key: (rule.priority, rule.rule_key)
        for rule in reference_scenario.rules
    }

    def source_units(
        scenario_key: str,
    ) -> dict[tuple[str, str, int | None, str | None], dict[str, object]]:
        units: dict[tuple[str, str, int | None, str | None], dict[str, object]] = {}
        for row in records:
            if pd.isna(row["scenario_key"]) or row["scenario_key"] != scenario_key:
                continue
            identity = _normalized_comparison_identity(row)
            if identity is not None:
                units[identity] = row
        return units

    reference_units = source_units(reference_key)
    rows: list[dict[str, object]] = []
    for comparator in config.scenarios:
        if comparator.scenario_key == reference_key:
            continue
        comparator_units = source_units(comparator.scenario_key)
        shared_identities = [
            identity
            for identity in reference_units
            if identity in comparator_units
        ]
        shared_identities.sort(
            key=lambda identity: (
                _COMPARISON_METRIC_ORDER_INDEX[identity[0]],
                _COMPARISON_SCOPE_ORDER_INDEX[identity[1]],
                -1 if identity[2] is None else identity[2],
                reference_priority.get(identity[3], (0, ""))
                if identity[1] == "scenario_rule"
                else (0, ""),
            )
        )
        for metric, scope_key, scope_position, rule_key in shared_identities:
            reference = reference_units[(metric, scope_key, scope_position, rule_key)]
            comparator_row = comparator_units[
                (metric, scope_key, scope_position, rule_key)
            ]
            normal = (
                reference["status"] == "available"
                and comparator_row["status"] == "available"
                and _source_values_match(reference, comparator_row)
                and _numeric_value(reference["metric_value"]) is not None
                and _numeric_value(comparator_row["metric_value"]) is not None
            )
            if normal:
                reference_value = _numeric_value(reference["metric_value"])
                comparator_value = _numeric_value(comparator_row["metric_value"])
                status, reason = "available", "computed"
                numerator = _difference(
                    reference["numerator"], comparator_row["numerator"]
                )
                denominator = _shared_value(
                    reference["denominator"], comparator_row["denominator"]
                )
                support_n = reference["support_n"]
                support_unit = reference["support_unit"]
                delta = comparator_value - reference_value
            elif (
                reference["status"] == "available"
                and comparator_row["status"] == "available"
            ):
                status, reason = "not_verifiable", "support_not_comparable"
                reference_value = comparator_value = delta = numerator = denominator = (
                    pd.NA
                )
                support_n, support_unit = 0, pd.NA
            else:
                status, reason = _comparison_failure(reference, comparator_row)
                reference_value = comparator_value = delta = numerator = denominator = (
                    pd.NA
                )
                support_n, support_unit = 0, pd.NA
            rows.append(
                {
                    "reference_scenario_key": reference_key,
                    "comparator_scenario_key": comparator.scenario_key,
                    "metric": metric,
                    "scope_key": scope_key,
                    "scope_position": pd.NA
                    if scope_position is None
                    else scope_position,
                    "rule_key": pd.NA if rule_key is None else rule_key,
                    "reference_value": reference_value,
                    "comparator_value": comparator_value,
                    "delta": delta,
                    "numerator": numerator,
                    "denominator": denominator,
                    "support_n": support_n,
                    "support_unit": support_unit,
                    "status": status,
                    "reason": reason,
                    "finding_key": "monitoring:scenario_comparison",
                }
            )
    rows.sort(
        key=lambda row: (
            scenario_order[row["reference_scenario_key"]],
            scenario_order[row["comparator_scenario_key"]],
            _COMPARISON_METRIC_ORDER_INDEX[row["metric"]],
            _COMPARISON_SCOPE_ORDER_INDEX[row["scope_key"]],
            -1 if pd.isna(row["scope_position"]) else int(row["scope_position"]),
            reference_priority.get(row["rule_key"], (0, ""))
            if row["scope_key"] == "scenario_rule"
            else (0, ""),
        )
    )
    return rows


def _episode_positions(
    episodes: list[dict[str, object]], observation_history: list[dict[str, object]]
) -> dict[int, int]:
    """Associate an episode with its private start-row identity for scoping."""
    start_positions = {
        (int(row["entity_position"]), row["observation_time"]): int(
            row["row_position"]
        )
        for row in observation_history
    }
    return {
        index: start_positions[
            (int(episode["entity_position"]), episode["episode_start_time"])
        ]
        for index, episode in enumerate(episodes)
    }


def _scenario_alert_level_groups(
    config: LifecycleMonitoringConfig, scenario: WarningScenario
) -> tuple[tuple[int, frozenset[str]], ...]:
    """Select only configured alert levels used by one scenario's rules."""
    alert_order = {
        key: position
        for position, (key, _) in enumerate(
            sorted(config.alert_level_ranks, key=lambda item: (-item[1], item[0]))
        )
    }
    by_level: dict[str, list[str]] = {}
    for rule in scenario.rules:
        by_level.setdefault(rule.alert_level, []).append(rule.rule_key)
    return tuple(
        (alert_order[level], frozenset(by_level[level]))
        for level in sorted(by_level, key=lambda key: alert_order[key])
    )


def _scenario_state_groups(
    config: LifecycleMonitoringConfig,
    state_history: list[dict[str, object]],
) -> tuple[_ScopeGroup, ...]:
    """Build actual state memberships with config-global state ordinals."""
    groups: list[_ScopeGroup] = []
    ordered_states = sorted(
        config.states, key=lambda state: (state.state_rank, state.state_key)
    )
    for ordinal, state in enumerate(ordered_states):
        positions = frozenset(
            int(row["row_position"])
            for row in state_history
            if row["status"] == "available"
            and row["effective_state_key"] == state.state_key
        )
        if positions:
            groups.append(_ScopeGroup(ordinal, positions))
    return tuple(groups)


_TRANSITION_SCOPE_INVENTORY = (
    ("entry", "not_applicable"),
    ("reentry", "not_applicable"),
    ("stay", "flat"),
    ("change", "roll_forward"),
    ("change", "roll_back"),
    ("change", "flat"),
    ("invalid", "not_applicable"),
)
_TRANSITION_SCOPE_ORDER = {
    key: position for position, key in enumerate(_TRANSITION_SCOPE_INVENTORY)
}


def _scenario_transition_groups(
    state_transitions: list[dict[str, object]],
) -> tuple[_ScopeGroup, ...]:
    """Build actual kind/direction memberships with global frozen ordinals."""
    memberships: dict[tuple[str, str], list[int]] = {}
    for transition in state_transitions:
        key = (
            transition["transition_kind"],
            transition["transition_direction"],
        )
        if key not in _TRANSITION_SCOPE_ORDER:
            continue
        memberships.setdefault(key, []).append(int(transition["to_row_position"]))
    return tuple(
        _ScopeGroup(_TRANSITION_SCOPE_ORDER[key], frozenset(memberships[key]))
        for key in _TRANSITION_SCOPE_INVENTORY
        if key in memberships
    )


def _scoped_monitoring_summaries(
    config: LifecycleMonitoringConfig,
    facts: _ScopeFacts,
    rule_evaluations: list[dict[str, object]],
    notifications: list[dict[str, object]],
    episodes: list[dict[str, object]],
    event_matches: list[dict[str, object]],
    observation_history: pd.DataFrame,
    loss_evidence: pd.DataFrame,
    loss_classification: pd.DataFrame,
    vintage_state_groups: tuple[_ScopeGroup, ...],
    state_history: list[dict[str, object]],
    state_transitions: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Reuse the frozen metric engine on approved anonymous row subsets only."""
    rows: list[dict[str, object]] = []
    observations = observation_history.to_dict("records")
    episode_positions = _episode_positions(episodes, observations)

    def add(
        group: _ScopeGroup,
        scope_key: str,
        scenario_key: str | None,
        *,
        include_events: bool,
        rule_keys: frozenset[str] | None = None,
        enforce_matrix: bool = False,
    ) -> None:
        positions = group.row_positions
        evaluations = [
            row
            for row in rule_evaluations
            if int(row["row_position"]) in positions
            and (scenario_key is None or row["scenario_key"] == scenario_key)
            and (rule_keys is None or row["rule_key"] in rule_keys)
        ]
        notes = [
            row
            for row in notifications
            if int(row["row_position"]) in positions
            and (scenario_key is None or row["scenario_key"] == scenario_key)
            and (rule_keys is None or row["rule_key"] in rule_keys)
        ]
        scoped_episodes = [
            episode
            for index, episode in enumerate(episodes)
            if episode_positions[index] in positions
            and (scenario_key is None or episode["scenario_key"] == scenario_key)
            and (rule_keys is None or episode["rule_key"] in rule_keys)
        ]
        matches = [
            match
            for match in event_matches
            if int(match["event_row_position"]) in positions
        ]
        scoped_observations = [
            row for row in observations if int(row["row_position"]) in positions
        ]
        source_scope = "scenario" if scenario_key is not None else "overall"
        generated = _monitoring_summaries(
            config,
            evaluations,
            notes,
            scoped_episodes,
            matches,
            scoped_observations,
        )
        for row in generated:
            if row["scope_key"] != source_scope:
                continue
            if scenario_key is not None and row["scenario_key"] != scenario_key:
                continue
            if not include_events and row["metric"] in _EVENT_METRICS:
                continue
            if enforce_matrix and not _monitoring_metric_allowed(
                scope_key, row["metric"]
            ):
                continue
            row = dict(row)
            row["scope_key"] = scope_key
            row["scope_position"] = group.scope_position
            row["finding_key"] = f"monitoring:{scope_key}:{row['metric']}"
            rows.append(row)
        scoped_observation = observation_history.loc[
            observation_history["row_position"].isin(positions)
        ]
        if rule_keys is not None:
            scoped_observation = scoped_observation.loc[
                scoped_observation["primary_rule_key"].isin(rule_keys)
            ]
        scoped_positions = scoped_observation["row_position"]
        evidence = loss_evidence.loc[
            loss_evidence["row_position"].isin(scoped_positions)
        ]
        classification = loss_classification.loc[
            loss_classification["row_position"].isin(scoped_positions)
        ]
        loss_rows, _ = _loss_summary_rows(
            config,
            evidence,
            classification,
            scoped_observation,
        )
        for row in loss_rows:
            is_expected = row["metric"] in {"expected_loss_sum", "expected_loss_rate"}
            if is_expected:
                if scenario_key is None or row["scenario_key"] != scenario_key:
                    continue
            elif row["scope_key"] != "overall":
                continue
            if enforce_matrix and not _monitoring_metric_allowed(
                scope_key, row["metric"]
            ):
                continue
            row = dict(row)
            row["scope_key"] = scope_key
            row["scope_position"] = group.scope_position
            row["finding_key"] = f"monitoring:{scope_key}:{row['metric']}"
            rows.append(row)

    all_positions = frozenset(int(row["row_position"]) for row in observations)
    for scenario in config.scenarios:
        for ordinal, rule_keys in _scenario_alert_level_groups(config, scenario):
            add(
                _ScopeGroup(ordinal, all_positions),
                "scenario_alert_level",
                scenario.scenario_key,
                include_events=False,
                rule_keys=rule_keys,
                enforce_matrix=True,
            )
        for group in facts.segment_groups:
            add(
                group,
                "scenario_segment",
                scenario.scenario_key,
                include_events=True,
                enforce_matrix=True,
            )
        for group in facts.time_groups:
            add(
                group,
                "scenario_time",
                scenario.scenario_key,
                include_events=True,
                enforce_matrix=True,
            )
        for group in facts.cohort_groups:
            add(
                group,
                "scenario_cohort",
                scenario.scenario_key,
                include_events=True,
                enforce_matrix=True,
            )
        for group in facts.vintage_groups:
            add(
                group,
                "scenario_vintage",
                scenario.scenario_key,
                include_events=True,
                enforce_matrix=True,
            )
        for group in _scenario_state_groups(config, state_history):
            add(
                group,
                "scenario_state",
                scenario.scenario_key,
                include_events=False,
                enforce_matrix=True,
            )
        for group in _scenario_transition_groups(state_transitions):
            add(
                group,
                "scenario_transition",
                scenario.scenario_key,
                include_events=False,
                enforce_matrix=True,
            )
    for scope_key, groups in (
        ("segment_time", facts.segment_time_groups),
        ("cohort_time", facts.cohort_time_groups),
        ("vintage_state", vintage_state_groups),
    ):
        for group in groups:
            add(group, scope_key, None, include_events=False)
    return rows


def _vintage_state_groups(
    config: LifecycleMonitoringConfig,
    facts: _ScopeFacts,
    state_history: list[dict[str, object]],
) -> tuple[tuple[_ScopeGroup, str], ...]:
    """Create only actual vintage × effective-state memberships in frozen order."""
    available = [row for row in state_history if row["status"] == "available"]
    ordered_states = sorted(
        config.states, key=lambda state: (state.state_rank, state.state_key)
    )
    groups: list[tuple[_ScopeGroup, str]] = []
    for vintage in facts.vintage_groups:
        for state in ordered_states:
            positions = frozenset(
                int(row["row_position"])
                for row in available
                if int(row["row_position"]) in vintage.row_positions
                and row["effective_state_key"] == state.state_key
            )
            if positions:
                groups.append((_ScopeGroup(len(groups), positions), state.state_key))
    return tuple(groups)


def _scoped_lifecycle_summaries(
    config: LifecycleMonitoringConfig,
    facts: _ScopeFacts,
    state_history: list[dict[str, object]],
    state_transitions: list[dict[str, object]],
    loss_evidence: pd.DataFrame,
    loss_classification: pd.DataFrame,
    observation_history: pd.DataFrame,
) -> tuple[list[dict[str, object]], tuple[_ScopeGroup, ...]]:
    """Expand only approved lifecycle scopes without changing base calculations."""
    rows: list[dict[str, object]] = []
    durations = _state_durations(
        config, [row for row in state_history if row["status"] == "available"]
    )
    vintage_pairs = _vintage_state_groups(config, facts, state_history)

    def add(group: _ScopeGroup, scope_key: str, state_key: str | None = None) -> None:
        positions = group.row_positions
        history = [
            row for row in state_history if int(row["row_position"]) in positions
        ]
        transitions = [
            row
            for row in state_transitions
            if int(row["to_row_position"]) in positions
        ]
        generated = _lifecycle_summaries(
            config, history, transitions, durations=durations
        )
        source_scope = "state" if state_key is not None else "overall"
        for row in generated:
            if row["scope_key"] != source_scope:
                continue
            if state_key is not None and row["to_state_key"] != state_key:
                continue
            row = dict(row)
            row["scope_key"] = scope_key
            row["scope_position"] = group.scope_position
            row["finding_key"] = f"monitoring:lifecycle:{scope_key}:{row['metric']}"
            rows.append(row)
        evidence = loss_evidence.loc[loss_evidence["row_position"].isin(positions)]
        classification = loss_classification.loc[
            loss_classification["row_position"].isin(positions)
        ]
        _, loss_rows = _loss_summary_rows(
            config,
            evidence,
            classification,
            observation_history.loc[observation_history["row_position"].isin(positions)],
        )
        for row in loss_rows:
            row = dict(row)
            row["scope_key"] = scope_key
            row["scope_position"] = group.scope_position
            row["finding_key"] = f"monitoring:lifecycle:{scope_key}:{row['metric']}"
            rows.append(row)

    for scope_key, groups in (
        ("segment_time", facts.segment_time_groups),
        ("cohort_time", facts.cohort_time_groups),
    ):
        for group in groups:
            add(group, scope_key)
    vintage_groups: list[_ScopeGroup] = []
    for group, state_key in vintage_pairs:
        add(group, "vintage_state", state_key)
        vintage_groups.append(group)
    return rows, tuple(vintage_groups)


def _rule_frame(
    data: pd.DataFrame,
    config: LifecycleMonitoringConfig,
    condition: MonitoringCondition,
    entities: list[int],
    source_values: dict[str, pd.Series] | None = None,
) -> tuple[pd.DataFrame, _ConditionNode]:
    """Build an immutable position-indexed kernel input for one condition."""
    values = data.loc[:, list(config.condition_feature_columns)].copy(deep=False)
    signal_names: list[str] = []

    def visit(node: MonitoringCondition) -> MonitoringCondition:
        if node.kind != "atomic":
            return MonitoringCondition(
                node.kind, children=tuple(visit(c) for c in node.children)
            )
        name = f"__task18_signal_{len(signal_names)}"
        if node.left_kind in _DERIVED:
            signal_names.append(name)
            values[name] = _derived_signal(data, config, node, entities).array
            return MonitoringCondition(
                "atomic", node.operator, "column", name, node.right_kind, node.right
            )
        if node.left_kind in {"ranking_score", "event_probability"}:
            signal_names.append(name)
            if source_values is None:
                values[name] = pd.Series([pd.NA] * len(data), dtype="Float64")
            else:
                values[name] = source_values[node.left_kind].array
            return MonitoringCondition(
                "atomic", node.operator, "column", name, node.right_kind, node.right
            )
        return node

    rewritten = visit(condition)
    if rewritten.left_kind != "column" and rewritten.kind == "atomic":
        values["__task18_signal"] = pd.Series([pd.NA] * len(data), dtype="Float64")
    return values.reset_index(drop=True), _compile_condition(rewritten)


def _derived_signal(
    data: pd.DataFrame,
    config: LifecycleMonitoringConfig,
    condition: MonitoringCondition,
    entities: list[int],
) -> pd.Series:
    """Compute closed prior-only numeric signals with one per-call index."""
    result: list[object] = [pd.NA] * len(data)
    source = condition.left
    if source is None:
        return pd.Series(result, dtype="Float64")
    if condition.left_kind == "peer_deviation":
        return _peer_deviation(data, config, source)
    ordered = _ordered_positions(data, config, entities)
    window = (
        config.recent_window
        if condition.window == "recent"
        else config.history_window
    )
    by_entity: dict[int, list[int]] = {}
    for position in ordered:
        by_entity.setdefault(entities[position], []).append(position)
    for positions in by_entity.values():
        times = [data[config.observation_time_column].iat[p] for p in positions]
        values = [
            float(data[source].iat[p]) if _finite_real(data[source].iat[p]) else None
            for p in positions
        ]
        finite_positions = [
            index for index, value in enumerate(values) if value is not None
        ]
        prefix_count = [0]
        prefix_sum = [0.0]
        prefix_x = [0.0]
        prefix_y = [0.0]
        prefix_x2 = [0.0]
        prefix_xy = [0.0]
        origin = times[0]
        for time, value in zip(times, values, strict=True):
            x = (time - origin).total_seconds()
            present = value is not None
            prefix_count.append(prefix_count[-1] + int(present))
            prefix_sum.append(prefix_sum[-1] + (value if present else 0.0))
            prefix_x.append(prefix_x[-1] + (x if present else 0.0))
            prefix_y.append(prefix_y[-1] + (value if present else 0.0))
            prefix_x2.append(prefix_x2[-1] + (x * x if present else 0.0))
            prefix_xy.append(prefix_xy[-1] + (x * value if present else 0.0))
        for index, position in enumerate(positions):
            current = values[index]
            if current is None:
                continue
            start = times[index] - window
            left = (
                bisect_left(times, start)
                if config.history_start_inclusive
                else bisect_right(times, start)
            )
            right = index  # strict-prior: never include the current row
            finite_right = bisect_left(finite_positions, right)
            count = prefix_count[right] - prefix_count[left]
            if count == 0:
                continue
            if condition.left_kind in {"prior_value", "change"}:
                prior = values[finite_positions[finite_right - 1]]
                result[position] = (
                    prior
                    if condition.left_kind == "prior_value"
                    else current - prior
                )
            elif condition.left_kind == "history_mean":
                result[position] = (
                    prefix_sum[right] - prefix_sum[left]
                ) / count
            elif condition.left_kind == "trend" and count >= 2:
                sum_x = prefix_x[right] - prefix_x[left]
                sum_y = prefix_y[right] - prefix_y[left]
                sum_x2 = prefix_x2[right] - prefix_x2[left]
                sum_xy = prefix_xy[right] - prefix_xy[left]
                denominator = sum_x2 - (sum_x * sum_x / count)
                if denominator:
                    result[position] = (
                        sum_xy - (sum_x * sum_y / count)
                    ) / denominator
    return pd.Series(result, dtype="Float64")


def _peer_deviation(
    data: pd.DataFrame, config: LifecycleMonitoringConfig, source: str
) -> pd.Series:
    """Return current value minus a frozen, strict-prior peer mean.

    The immutable per-call baseline is grouped once.  Every query therefore
    performs only scalar eligibility checks and a dictionary lookup; no query
    rescans the input frame.
    """
    result: list[object] = [pd.NA] * len(data)
    end = config.peer_reference_end
    groups = config.peer_group_columns
    if end is None or not groups:
        return pd.Series(result, dtype="Float64")
    start = config.peer_reference_start
    peer_keys: list[tuple[object, ...] | None] = []
    grouped_values: dict[tuple[object, ...], list[float]] = {}
    for position in range(len(data)):
        group = tuple(data[column].iat[position] for column in groups)
        if any(not _scalar(item) for item in group):
            raise _error("input schema is invalid", "unsupported_dtype")
        if any(_scope_missing(item) for item in group):
            peer_keys.append(None)
            continue
        key = tuple(_peer_identity(item) for item in group)
        peer_keys.append(key)
        observed = data[config.observation_time_column].iat[position]
        available = data[config.available_time_column].iat[position]
        if (
            observed > end
            or available > end
            or (start is not None and observed < start)
        ):
            continue
        value = data[source].iat[position]
        if _finite_real(value):
            grouped_values.setdefault(key, []).append(float(value))
    for position in range(len(data)):
        current_time = data[config.observation_time_column].iat[position]
        current = data[source].iat[position]
        if end >= current_time or not _finite_real(current):
            continue
        key = peer_keys[position]
        if key is None:
            continue
        values = grouped_values.get(key, ())
        if len(values) >= 2:
            result[position] = float(current) - (sum(values) / len(values))
    return pd.Series(result, dtype="Float64")


def _peer_identity(value: object) -> tuple[object, ...]:
    """Return a deterministic key for an already validated peer scalar."""
    value_type = type(value)
    if value_type is bool:
        return ("bool", value)
    if value_type is int:
        return ("int", value)
    if value_type is float:
        return ("float", value)
    if value_type is str:
        return ("str", value)
    if value_type is date:
        return ("date", value.toordinal())
    if value_type is datetime:
        offset = value.utcoffset()
        if offset is None:
            return (
                "datetime-naive",
                value.year,
                value.month,
                value.day,
                value.hour,
                value.minute,
                value.second,
                value.microsecond,
                value.fold,
            )
        return ("datetime-aware", value.timestamp())
    if value_type is pd.Timestamp:
        return ("timestamp", value.value)
    raise _error("input schema is invalid", "unsupported_dtype")


def _finite_real(value: object) -> bool:
    """Accept exact non-bool finite numeric scalars without coercion."""
    return (
        type(value) in {int, float, np.int64, np.float64}
        and type(value) is not bool
        and (type(value) not in {float, np.float64} or isfinite(value))
    )


def _apply_persistence(
    evaluations: list[dict[str, object]], config: LifecycleMonitoringConfig
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Apply the bounded per-rule warning state machine without recomputation."""
    notifications: list[dict[str, object]] = []
    episodes: list[dict[str, object]] = []
    memory: dict[tuple[int, int, int], dict[str, object]] = {}
    order = sorted(
        evaluations,
        key=lambda r: (
            int(r["entity_position"]),
            int(r["scenario_order"]),
            int(r["rule_order"]),
            r["observation_time"],
            int(r["row_position"]),
        ),
    )
    ranks = dict(config.alert_level_ranks)
    for row in order:
        entity, scenario_order, rule_order = (
            int(row["entity_position"]),
            int(row["scenario_order"]),
            int(row["rule_order"]),
        )
        rule = config.scenarios[scenario_order].rules[rule_order]
        state = memory.setdefault(
            (entity, scenario_order, rule_order),
            {
                "true": 0,
                "false": 0,
                "open": False,
                "ordinal": -1,
                "start": None,
                "hits": 0,
                "notes": 0,
                "last_emitted": None,
                "suppressed": 0,
                "previous_time": None,
            },
        )
        truth, time = row["truth"], row["observation_time"]
        previous_time = state["previous_time"]
        state["previous_time"] = time
        consecutive = (
            previous_time is None
            or config.expected_observation_interval is None
            or time - previous_time <= config.expected_observation_interval
        )
        row["notification_status"] = "not_emitted"
        if not consecutive:
            state["true"], state["false"] = 0, 0
            row["true_streak"], row["false_streak"] = 0, 0
            row["episode_status"] = "active" if state["open"] else "clear"
            continue
        if truth == "true":
            state["true"], state["false"] = int(state["true"]) + 1, 0
        elif truth == "false":
            state["false"], state["true"] = int(state["false"]) + 1, 0
        else:
            state["true"], state["false"] = 0, 0
        row["true_streak"], row["false_streak"] = state["true"], state["false"]
        row["episode_status"] = "clear"
        if (
            truth == "true"
            and not state["open"]
            and int(state["true"]) >= rule.persistence_observations
        ):
            state.update(
                open=True,
                ordinal=int(state["ordinal"]) + 1,
                start=time,
                hits=1,
                notes=1,
                last_emitted=time,
                suppressed=0,
            )
            row["episode_status"], row["notification_status"] = "active", "emitted"
            notifications.append(
                _notification(
                    row,
                    int(state["ordinal"]),
                    0,
                    "episode_reopen" if int(state["ordinal"]) else "episode_open",
                )
            )
        elif (
            state["open"]
            and truth == "false"
            and int(state["false"]) >= rule.resolution_observations
        ):
            state["open"] = False
            row["episode_status"] = "resolved"
            episodes.append(_episode(row, state, rule, resolved=True))
        elif state["open"] and truth == "true":
            row["episode_status"] = "active"
            state["hits"] = int(state["hits"]) + 1
            if time - state["last_emitted"] >= rule.cooldown:
                ordinal = int(state["notes"])
                state["notes"] = ordinal + 1
                state["last_emitted"] = time
                row["notification_status"] = "emitted"
                notifications.append(
                    _notification(row, int(state["ordinal"]), ordinal, "repeated")
                )
            else:
                row["notification_status"] = "suppressed"
                state["suppressed"] = int(state["suppressed"]) + 1
        elif truth == "true":
            row["episode_status"] = "pending"
        elif state["open"] and truth == "false":
            row["episode_status"] = "pending"
        elif state["open"]:
            row["episode_status"] = "active"
    for (entity, scenario_order, rule_order), state in sorted(memory.items()):
        if state["open"]:
            scenario, rule = (
                config.scenarios[scenario_order],
                config.scenarios[scenario_order].rules[rule_order],
            )
            row = {
                "entity_position": entity,
                "scenario_key": scenario.scenario_key,
                "rule_key": rule.rule_key,
                "alert_rank": ranks[rule.alert_level],
                "observation_time": config.analysis_as_of,
            }
            episodes.append(_episode(row, state, rule, resolved=False))
    notifications.sort(
        key=lambda row: (
            row["notification_time"],
            int(row["entity_position"]),
            str(row["scenario_key"]),
            str(row["rule_key"]),
            int(row["notification_ordinal"]),
        )
    )
    episodes.sort(
        key=lambda row: (
            int(row["entity_position"]),
            str(row["scenario_key"]),
            str(row["rule_key"]),
            int(row["episode_ordinal"]),
        )
    )
    return notifications, episodes


def _notification(
    row: dict[str, object], episode: int, ordinal: int, kind: str
) -> dict[str, object]:
    return {
        "entity_position": row["entity_position"],
        "scenario_key": row["scenario_key"],
        "rule_key": row["rule_key"],
        "episode_ordinal": episode,
        "notification_ordinal": ordinal,
        "row_position": row["row_position"],
        "notification_time": row["observation_time"],
        "alert_level": row["alert_level"],
        "alert_rank": row["alert_rank"],
        "notification_kind": kind,
        "is_repeated": kind == "repeated",
        "first_matched_event_ordinal": pd.NA,
        "matched_event_count": 0,
        "maturity_status": "not_applicable",
        "status": "available",
        "reason": "computed",
        "finding_key": row["finding_key"],
    }


def _episode(
    row: dict[str, object],
    state: dict[str, object],
    rule: EarlyWarningRule,
    *,
    resolved: bool,
) -> dict[str, object]:
    end = row["observation_time"] if resolved else pd.NaT
    duration = (row["observation_time"] - state["start"]).total_seconds()
    return {
        "entity_position": row["entity_position"],
        "scenario_key": row["scenario_key"],
        "rule_key": row["rule_key"],
        "episode_ordinal": state["ordinal"],
        "alert_level": rule.alert_level,
        "alert_rank": row["alert_rank"],
        "episode_start_time": state["start"],
        "episode_end_time": end,
        "duration_seconds": duration,
        "raw_hit_count": state["hits"],
        "notification_count": state["notes"],
        "suppressed_notification_count": state["suppressed"],
        "is_reopen": state["ordinal"] > 0,
        "is_unresolved": not resolved,
        "status": "available",
        "reason": "episode_resolved" if resolved else "episode_active",
        "finding_key": f"entity:{row['entity_position']}:episode:{state['ordinal']}",
    }


def _state_condition_results(
    data: pd.DataFrame,
    config: LifecycleMonitoringConfig,
    entities: list[int],
    source_values: dict[str, pd.Series] | None = None,
) -> dict[str, object]:
    """Evaluate eligible state candidates through the one Task 16 kernel path."""
    inventory = {config.default_state_key, config.unknown_state_key}
    results: dict[str, object] = {}
    for state in config.states:
        if not state.enabled or state.state_key in inventory:
            continue
        values, node = _rule_frame(
            data, config, state.condition, entities, source_values
        )
        results[state.state_key] = _evaluate_condition(values, node)
    return results


def _apply_states(
    data: pd.DataFrame,
    config: LifecycleMonitoringConfig,
    entities: list[int],
    evaluations: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Assign candidates and effective states without changing warning evidence."""
    by_key = {state.state_key: state for state in config.states}
    inventory = {config.default_state_key, config.unknown_state_key}
    candidates = [
        state
        for state in config.states
        if state.enabled and state.state_key not in inventory
    ]
    histories: list[dict[str, object]] = []
    transitions: list[dict[str, object]] = []
    memory: dict[int, dict[str, object]] = {}
    for position in _ordered_positions(data, config, entities):
        entity = entities[position]
        time = data[config.observation_time_column].iat[position]
        true_states = [
            state
            for state in candidates
            if str(evaluations[state.state_key].truth.iat[position]) == "true"
        ]
        unknown = any(
            str(evaluations[state.state_key].truth.iat[position]) == "unknown"
            for state in candidates
        )
        if true_states:
            candidate = min(
                true_states, key=lambda state: (state.priority, state.state_key)
            )
            state_status, state_reason = "available", "computed"
            candidate_is_unknown = False
        elif unknown:
            candidate = by_key[config.unknown_state_key]
            state_status, state_reason = "not_verifiable", "unknown_condition"
            candidate_is_unknown = True
        else:
            candidate = by_key[config.default_state_key]
            state_status, state_reason = "available", "default_state_applied"
            candidate_is_unknown = False

        prior = memory.get(entity)
        prior_time = prior["time"] if prior is not None else None
        consecutive = prior is not None and (
            config.expected_observation_interval is None
            or time - prior_time <= config.expected_observation_interval
        )
        if prior is None:
            effective = candidate
            kind, direction, allowed = "entry", "not_applicable", True
            transition_status, transition_reason = "available", "entry_observation"
            from_position: object = pd.NA
            from_key: object = pd.NA
            from_rank: object = pd.NA
            cure = False
        else:
            previous = by_key[prior["key"]]
            from_position, from_key, from_rank = (
                prior["row_position"],
                previous.state_key,
                previous.state_rank,
            )
            if not consecutive:
                effective = candidate
                kind, direction, allowed = "reentry", "not_applicable", True
                transition_status, transition_reason = (
                    "available",
                    "reentry_observation",
                )
                cure = previous.state_key in config.adverse_state_keys and (
                    candidate.state_key in config.cure_state_keys
                )
            elif candidate.state_key == previous.state_key:
                effective = candidate
                kind, direction, allowed = "stay", "flat", True
                transition_status, transition_reason = "available", "transition_allowed"
                cure = False
            elif previous.terminal:
                effective = previous
                kind, direction, allowed = "invalid", "not_applicable", False
                transition_status, transition_reason = (
                    "not_verifiable",
                    "terminal_state_exit",
                )
                cure = False
            elif (
                config.allowed_transitions
                and (previous.state_key, candidate.state_key)
                not in config.allowed_transitions
            ):
                effective = previous
                kind, direction, allowed = "invalid", "not_applicable", False
                transition_status, transition_reason = (
                    "not_verifiable",
                    "transition_not_allowed",
                )
                cure = False
            else:
                effective = candidate
                kind, allowed = "change", True
                if candidate.state_rank > previous.state_rank:
                    direction = "roll_forward"
                elif candidate.state_rank < previous.state_rank:
                    direction = "roll_back"
                else:
                    direction = "flat"
                transition_status, transition_reason = "available", "transition_allowed"
                cure = previous.state_key in config.adverse_state_keys and (
                    candidate.state_key in config.cure_state_keys
                )
            if state_status == "not_verifiable" and kind != "invalid":
                transition_status, transition_reason = state_status, state_reason

        histories.append(
            {
                "row_position": position,
                "entity_position": entity,
                "observation_time": time,
                "candidate_state_key": (
                    pd.NA if candidate_is_unknown else candidate.state_key
                ),
                "candidate_state_rank": (
                    pd.NA if candidate_is_unknown else candidate.state_rank
                ),
                "candidate_state_priority": (
                    pd.NA if candidate_is_unknown else candidate.priority
                ),
                "effective_state_key": effective.state_key,
                "effective_state_rank": effective.state_rank,
                "matching_state_count": len(true_states),
                "status": state_status,
                "reason": state_reason,
                "finding_key": f"state:{position}",
            }
        )
        transitions.append(
            {
                "entity_position": entity,
                "from_row_position": from_position,
                "to_row_position": position,
                "transition_time": time,
                "from_state_key": from_key,
                "candidate_to_state_key": (
                    pd.NA if candidate_is_unknown else candidate.state_key
                ),
                "effective_to_state_key": effective.state_key,
                "from_rank": from_rank,
                "candidate_to_rank": (
                    pd.NA if candidate_is_unknown else candidate.state_rank
                ),
                "effective_to_rank": effective.state_rank,
                "transition_kind": kind,
                "transition_direction": direction,
                "is_allowed": allowed,
                "is_consecutive": pd.NA if prior is None else consecutive,
                "is_cure": cure,
                "exposure": pd.NA,
                "observed_loss": pd.NA,
                "status": transition_status,
                "reason": transition_reason,
                "finding_key": (
                    "transition:"
                    f"{-1 if from_key is pd.NA else _state_ordinal(config, from_key)}:"
                    f"{_state_ordinal(config, effective.state_key)}"
                ),
            }
        )
        if kind != "invalid":
            memory[entity] = {
                "key": effective.state_key,
                "time": time,
                "row_position": position,
            }
        else:
            memory[entity]["time"] = time
            memory[entity]["row_position"] = position
    histories.sort(key=lambda row: int(row["row_position"]))
    transitions.sort(
        key=lambda row: (
            int(row["entity_position"]),
            row["transition_time"],
            int(row["to_row_position"]),
        )
    )
    return histories, transitions


def _state_ordinal(config: LifecycleMonitoringConfig, state_key: object) -> int:
    """Return the deterministic configuration ordinal for a state key."""
    for ordinal, state in enumerate(config.states):
        if state.state_key == state_key:
            return ordinal
    return -1


def monitor_lifecycle(
    data: pd.DataFrame,
    config: LifecycleMonitoringConfig,
    *,
    risk_validation: BinaryRiskValidationResult | None = None,
    data_audit: DataAuditResult | None = None,
) -> LifecycleMonitoringResult:
    """Run caller-declared offline post-loan warning and lifecycle monitoring.

    The final approved contract evaluates only evidence available at each
    observation, produces simulated notifications and lifecycle evidence, and
    never sends an actual notification, performs a production action, or
    optimizes a scenario. Inputs and optional Task 15/16 evidence remain
    unchanged; no current clock, timezone guess, or external state is used.

    Parameters
    ----------
    data
        Required prepared DataFrame. Its physical row order defines identity.
    config
        Required immutable configuration with explicit roles and ``analysis_as_of``.
    risk_validation, data_audit
        Optional frozen Task 15 and Task 16 evidence; later phases consume them
        only through their owner contracts.

    Returns
    -------
    LifecycleMonitoringResult
        Deterministic typed, privacy-preserving evidence tables.

    Raises
    ------
    ValueError
        For stable lifecycle config, schema, condition, alignment, or resource
        errors. Missing evidence is not silently fabricated.

    Examples
    --------
    >>> # result = monitor_lifecycle(prepared_frame, config)
    """
    checked = _validate_config(config)
    _validate_scenarios(checked)
    entity_positions, aware = _validate_data(data, checked)
    scope_facts = _scope_facts(data, checked, entity_positions, aware)
    _warning_resource_gates(checked, entity_positions)
    _state_resource_gates(checked, entity_positions)
    events = _declared_events(data, checked, entity_positions, aware)
    _event_resource_gates(checked, entity_positions, len(events))
    summary_projections = _summary_resource_gates(checked, scope_facts)
    source_values, source_metadata = _risk_source_values(data, checked, risk_validation)
    loss_evidence, loss_metadata = _loss_evidence(
        data, checked, source_values["event_probability"]
    )
    loss_classification = _classify_loss_evidence(
        loss_evidence, checked, source_metadata["probability_source"]
    )
    audit_metadata = _audit_metadata(data, data_audit)
    rule_evaluations: list[dict[str, object]] = []
    alert_ranks = dict(checked.alert_level_ranks)
    for scenario_order, scenario in enumerate(checked.scenarios):
        for rule_order, rule in enumerate(scenario.rules):
            values, node = _rule_frame(
                data, checked, rule.condition, entity_positions, source_values
            )
            evaluation = _evaluate_condition(values, node)
            for position in range(len(data)):
                entity = entity_positions[position]
                time = data[checked.observation_time_column].iat[position]
                active = (
                    rule.enabled
                    and (rule.effective_from is None or time >= rule.effective_from)
                    and (rule.expires_at is None or time < rule.expires_at)
                )
                truth = (
                    str(evaluation.truth.iat[position]) if active else "not_evaluated"
                )
                status = str(evaluation.status.iat[position]) if active else "inactive"
                reason = (
                    str(evaluation.reason.iat[position]) if active else "rule_inactive"
                )
                rule_evaluations.append(
                    {
                        "row_position": position,
                        "entity_position": entity,
                        "observation_time": time,
                        "scenario_key": scenario.scenario_key,
                        "scenario_order": scenario_order,
                        "rule_key": rule.rule_key,
                        "rule_order": rule_order,
                        "alert_level": rule.alert_level,
                        "alert_rank": alert_ranks[rule.alert_level],
                        "path_status": "evaluated" if active else "not_evaluated",
                        "truth": truth,
                        "true_streak": 0,
                        "false_streak": 0,
                        "episode_status": "not_evaluated",
                        "notification_status": "not_emitted",
                        "status": status,
                        "reason": reason,
                        "finding_key": f"rule:{scenario_order}:{rule_order}",
                    }
                )
    notifications, episodes = _apply_persistence(rule_evaluations, checked)
    event_matches, observation_maturity, observation_events = _event_evidence(
        data, checked, entity_positions, events, notifications
    )
    rule_evaluations.sort(
        key=lambda row: (
            int(row["row_position"]),
            int(row["scenario_order"]),
            checked.scenarios[int(row["scenario_order"])]
            .rules[int(row["rule_order"])]
            .priority,
            str(row["rule_key"]),
        )
    )
    state_evaluations = _state_condition_results(
        data, checked, entity_positions, source_values
    )
    state_history, state_transitions = _apply_states(
        data, checked, entity_positions, state_evaluations
    )
    tables = {name: _empty_table(name, aware) for name in _TABLE_SCHEMAS}
    observation = _empty_table("observation_history", aware)
    observation["row_position"] = pd.Series(range(len(data)), dtype="int64")
    observation["entity_position"] = pd.Series(entity_positions, dtype="int64")
    observation["observation_time"] = data[checked.observation_time_column].reset_index(
        drop=True
    )
    timestamp_dtype = "datetime64[ns, UTC]" if aware else "datetime64[ns]"
    observation["observation_time"] = observation["observation_time"].astype(
        timestamp_dtype
    )
    observation["observation_status"] = "available"
    observation["observation_reason"] = "computed"
    observation["is_consecutive"] = pd.Series(
        _consecutive_positions(data, checked, entity_positions), dtype="boolean"
    )
    observation["cohort_position"] = pd.Series(
        scope_facts.cohort_positions, dtype="Int64"
    )
    observation["period_index"] = pd.Series(
        scope_facts.period_indices, dtype="Int64"
    )
    observation["active_rule_count"] = 0
    observation["emitted_notification_count"] = 0
    observation["maturity_status"] = observation_maturity
    observation["event_within_horizon"] = pd.Series(
        observation_events,
        dtype="boolean",
    )
    observation["state_status"] = "not_applicable"
    observation["state_reason"] = "source_not_requested"
    state_by_position = {int(row["row_position"]): row for row in state_history}
    for position in range(len(data)):
        hits = [
            row
            for row in rule_evaluations
            if row["row_position"] == position
            and row["truth"] == "true"
            and row["episode_status"] == "active"
        ]
        if hits:
            primary = sorted(
                hits,
                key=lambda row: (
                    -int(row["alert_rank"]),
                    int(row["rule_order"]),
                    str(row["rule_key"]),
                ),
            )[0]
            observation.loc[position, "primary_scenario_key"] = primary["scenario_key"]
            observation.loc[position, "primary_rule_key"] = primary["rule_key"]
            observation.loc[position, "primary_alert_level"] = primary["alert_level"]
            observation.loc[position, "primary_alert_rank"] = primary["alert_rank"]
        observation.loc[position, "active_rule_count"] = len(hits)
        observation.loc[position, "emitted_notification_count"] = sum(
            note["row_position"] == position for note in notifications
        )
        state = state_by_position[position]
        observation.loc[position, "effective_state_key"] = state["effective_state_key"]
        observation.loc[position, "effective_state_rank"] = state[
            "effective_state_rank"
        ]
        observation.loc[position, "state_status"] = state["status"]
        observation.loc[position, "state_reason"] = state["reason"]
    monitoring_summary = _monitoring_summaries(
        checked,
        rule_evaluations,
        notifications,
        episodes,
        event_matches,
        observation.to_dict("records"),
    )
    lifecycle_summary = _lifecycle_summaries(checked, state_history, state_transitions)
    loss_monitoring, loss_lifecycle = _loss_summary_rows(
        checked, loss_evidence, loss_classification, observation
    )
    monitoring_summary.extend(loss_monitoring)
    monitoring_summary.extend(
        _scenario_loss_summaries(
            checked,
            loss_evidence,
            loss_classification,
            observation,
        )
    )
    lifecycle_summary.extend(loss_lifecycle)
    scoped_lifecycle, vintage_state_groups = _scoped_lifecycle_summaries(
        checked,
        scope_facts,
        state_history,
        state_transitions,
        loss_evidence,
        loss_classification,
        observation,
    )
    lifecycle_summary.extend(scoped_lifecycle)
    monitoring_summary.extend(
        _scoped_monitoring_summaries(
            checked,
            scope_facts,
            rule_evaluations,
            notifications,
            episodes,
            event_matches,
            observation,
            loss_evidence,
            loss_classification,
            vintage_state_groups,
            state_history,
            state_transitions,
        )
    )
    tables["observation_history"] = observation
    for name, rows in (
        ("rule_evaluations", rule_evaluations),
        ("notifications", notifications),
        ("alert_episodes", episodes),
        ("event_matches", event_matches),
        ("state_history", state_history),
        ("state_transitions", state_transitions),
        ("monitoring_summary", monitoring_summary),
        ("lifecycle_summary", lifecycle_summary),
    ):
        tables[name] = _typed_table(name, rows, aware)
    tables["scenario_comparison"] = _typed_table(
        "scenario_comparison",
        _scenario_comparison_rows(checked, tables["monitoring_summary"]),
        aware,
    )
    fingerprint = _fingerprint(checked)
    aware_time_model = "aware_utc" if aware else "naive"
    scenario_inventory = [
        {
            "ordinal": ordinal,
            "scenario_key": scenario.scenario_key,
            "scenario_kind": scenario.scenario_kind,
        }
        for ordinal, scenario in enumerate(checked.scenarios)
    ]
    rule_inventory = [
        {
            "scenario_ordinal": scenario_ordinal,
            "rule_ordinal": rule_ordinal,
            "rule_key": rule.rule_key,
            "priority": rule.priority,
            "alert_level": rule.alert_level,
            "enabled": rule.enabled,
        }
        for scenario_ordinal, scenario in enumerate(checked.scenarios)
        for rule_ordinal, rule in enumerate(scenario.rules)
    ]
    alert_inventory = [
        {"level": level, "rank": rank}
        for level, rank in checked.alert_level_ranks
    ]
    state_inventory = [
        {
            "ordinal": ordinal,
            "state_key": state.state_key,
            "rank": state.state_rank,
            "priority": state.priority,
            "terminal": state.terminal,
            "enabled": state.enabled,
        }
        for ordinal, state in enumerate(checked.states)
    ]
    transition_inventory = {
        "allowed": list(checked.allowed_transitions),
        "adverse": list(checked.adverse_state_keys),
        "cure": list(checked.cure_state_keys),
    }
    history_windows = {
        "recent_seconds": checked.recent_window.total_seconds(),
        "history_seconds": checked.history_window.total_seconds(),
        "history_start_inclusive": checked.history_start_inclusive,
        "expected_observation_interval_seconds": (
            None
            if checked.expected_observation_interval is None
            else checked.expected_observation_interval.total_seconds()
        ),
        "peer_reference_start": _canonical_encode(checked.peer_reference_start),
        "peer_reference_end": _canonical_encode(checked.peer_reference_end),
    }
    provenance_items = [
        ("contract_version", "task18-v1"),
        ("monitoring_fingerprint", fingerprint),
        ("row_identity", "physical_row_position"),
        ("entity_identity", "anonymous_first_appearance"),
        ("time_model", aware_time_model),
        ("analysis_as_of", _canonical_json(checked.analysis_as_of)),
        ("history_windows", _canonical_json(history_windows)),
        (
            "horizon_policy",
            "not_requested"
            if checked.prediction_horizon is None
            else _canonical_json(
                {
                    "seconds": checked.prediction_horizon.total_seconds(),
                    "end_inclusive": checked.horizon_end_inclusive,
                }
            ),
        ),
        ("scenario_inventory", _canonical_json(scenario_inventory)),
        ("rule_inventory", _canonical_json(rule_inventory)),
        ("alert_level_inventory", _canonical_json(alert_inventory)),
        ("state_inventory", _canonical_json(state_inventory)),
        ("transition_policy", _canonical_json(transition_inventory)),
        ("event_source", "dataframe" if checked.event_time_column else "not_requested"),
        ("score_source", source_metadata["score_source"]),
        ("probability_source", source_metadata["probability_source"]),
        ("task15_evidence_status", source_metadata["task15_evidence_status"]),
        ("task15_evidence_fingerprint", source_metadata["task15_evidence_fingerprint"]),
        ("task16_evidence_status", audit_metadata["task16_evidence_status"]),
        ("task16_config_fingerprint", audit_metadata["task16_config_fingerprint"]),
        ("task16_snapshot_identity", audit_metadata["task16_snapshot_identity"]),
        ("exposure_source", loss_metadata["exposure_source"]),
        ("observed_loss_source", loss_metadata["observed_loss_source"]),
        (
            "scope_inventory",
            "anonymous_segment_cohort_vintage"
            if (
                checked.segment_columns
                or checked.cohort_column is not None
                or checked.cohort_time_column is not None
            )
            else "overall_only",
        ),
        (
            "resource_usage",
            _canonical_json(summary_projections),
        ),
    ]
    provenance = pd.DataFrame(
        {
            "provenance_key": pd.Series(
                [key for key, _ in provenance_items],
                dtype="string",
            ),
            "provenance_value": pd.Series(
                [value for _, value in provenance_items],
                dtype="string",
            ),
            "status": pd.Series(["available"] * len(provenance_items), dtype="string"),
            "reason": pd.Series(["computed"] * len(provenance_items), dtype="string"),
            "finding_key": pd.Series(
                ["monitoring:provenance"] * len(provenance_items), dtype="string"
            ),
        }
    )
    tables["provenance"] = provenance
    return LifecycleMonitoringResult(
        checked.monitoring_key,
        checked.monitoring_version,
        fingerprint,
        len(data),
        len(set(entity_positions)),
        len(data),
        len(checked.scenarios),
        sum(len(scenario.rules) for scenario in checked.scenarios),
        sum(rule.enabled for scenario in checked.scenarios for rule in scenario.rules),
        len(checked.states),
        tables["observation_history"],
        tables["rule_evaluations"],
        tables["notifications"],
        tables["alert_episodes"],
        tables["event_matches"],
        tables["state_history"],
        tables["state_transitions"],
        tables["monitoring_summary"],
        tables["scenario_comparison"],
        tables["lifecycle_summary"],
        tables["provenance"],
        (),
        (
            "offline_monitoring_not_executed",
            "historical_comparison_not_causal",
            "caller_defined_states_and_alert_levels",
            "entity_linkage_depends_on_prepared_input",
        ),
    )
