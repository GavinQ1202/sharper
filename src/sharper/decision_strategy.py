"""Deterministic, offline pre-loan decision-strategy simulation.

This module implements the approved Task 17 contract.  It deliberately keeps
policy declaration, validation, simulation, and evidence assembly separate from
production decision execution, model fitting, and threshold optimisation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from sharper._condition_kernel import (
    _ConditionNode,
    _ConditionOperand,
    _evaluate_condition,
    _is_allowed_scalar,
    _normalize,
)
from sharper.data_audit import _TABLE_SCHEMAS as _AUDIT_TABLE_SCHEMAS
from sharper.data_audit import DataAuditResult
from sharper.risk_validation import (
    BinaryRiskValidationResult,
    _validate_binary_risk_validation_result,
)

_ROLE_ORDER = (
    "selected",
    "limited",
    "rejected",
    "review",
    "request_information",
    "other",
)
_ROLE_SET = frozenset(_ROLE_ORDER)
_SAFE_ACTION_ROLES = frozenset({"rejected", "review", "request_information"})
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
_CONSTRAINT_METRICS = (
    "action_count",
    "action_rate",
    "selected_rate",
    "rejected_rate",
    "review_count",
    "review_rate",
    "request_information_rate",
    "selected_exposure_sum",
    "expected_loss_sum",
    "expected_loss_rate",
    "observed_loss_sum",
    "observed_loss_rate",
    "selected_event_rate",
)
_STATUS = frozenset(
    {
        "available",
        "unavailable",
        "undefined",
        "not_applicable",
        "not_verifiable",
        "inactive",
    }
)
_KEY_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_SCORE_COLUMN = "__sharper_task17_ranking_score__"
_PROBABILITY_COLUMN = "__sharper_task17_event_probability__"
_SCHEMA_VERSION = "task17-decision-strategy-v1"
_TASK15_VERSION = "task15-binary-risk-validation-v1"
_KERNEL_VERSION = "task16-condition-kernel-v1"

_ROW_COLUMNS = (
    "row_position",
    "decision_status",
    "decision_reason",
    "base_action_name",
    "final_action_name",
    "applied_rule_key",
    "matched_rule_count",
    "unknown_rule_count",
    "overlap_rule_count",
    "conflict_rule_count",
    "override_applied",
    "historical_mapping_status",
)
_RULE_EVALUATION_COLUMNS = (
    "row_position",
    "rule_key",
    "phase",
    "priority",
    "rule_order",
    "path_status",
    "truth",
    "status",
    "reason",
    "is_applied",
    "is_overlap",
    "is_conflict",
)
_RULE_SUMMARY_COLUMNS = (
    "scope_type",
    "scope_column",
    "scope_ordinal",
    "time_slice_ordinal",
    "phase",
    "priority",
    "rule_key",
    "action_key",
    "action_role",
    "metric_key",
    "metric_value",
    "numerator",
    "denominator",
    "support_n_rows",
    "unit",
    "status",
    "reason",
    "finding_key",
)
_ACTION_SUMMARY_COLUMNS = (
    "scope_type",
    "scope_column",
    "scope_ordinal",
    "time_slice_ordinal",
    "action_key",
    "action_role",
    "metric_key",
    "metric_value",
    "numerator",
    "denominator",
    "support_n_rows",
    "unit",
    "status",
    "reason",
    "finding_key",
)
_BUSINESS_SUMMARY_COLUMNS = _ACTION_SUMMARY_COLUMNS
_CONSTRAINT_COLUMNS = (
    "constraint_key",
    "metric",
    "operator",
    "threshold",
    "action_name",
    "action_role",
    "actual_value",
    "status",
    "reason",
    "support_n",
    "gap",
    "violation_magnitude",
    "finding_key",
)
_TRANSITION_COLUMNS = (
    "historical_action_name",
    "simulated_action_name",
    "row_count",
    "row_rate",
    "status",
    "reason",
    "finding_key",
)
_PROVENANCE_COLUMNS = (
    "provenance_key",
    "provenance_value",
    "status",
    "reason",
    "finding_key",
)
_RULE_METRICS = (
    "evaluated_count",
    "hit_count",
    "hit_rate",
    "unknown_count",
    "unknown_rate",
    "not_evaluated_count",
    "applied_count",
    "sole_hit_count",
    "overlap_count",
    "overlap_rate",
    "conflict_count",
    "incremental_action_count",
    "leave_one_out_changed_action_count",
    "captured_event_count",
    "target_capture_rate",
)
_ACTION_METRICS = (
    "action_count",
    "action_rate",
    "evaluable_event_count",
    "event_count",
    "event_rate",
    "exposure_sum",
    "expected_loss_sum",
    "assumption_based_observed_event_loss_sum",
    "assumed_action_value_sum",
    "assumed_action_cost_sum",
    "assumption_based_payoff_sum",
)
_BUSINESS_METRICS = (
    "row_count",
    "decided_rate",
    "ranking_score_mean",
    "ranking_score_min",
    "ranking_score_max",
    "event_probability_mean",
    "event_probability_min",
    "event_probability_max",
    "observed_event_count",
    "observed_event_rate",
    "selected_rate",
    "rejected_rate",
    "review_capacity_rate",
    "unknown_action_rate",
    "exposure_sum",
    "expected_loss_sum",
    "expected_loss_rate",
    "assumption_based_observed_event_loss_sum",
    "actual_observed_loss_sum",
    "actual_observed_loss_rate",
    "assumed_action_value_sum",
    "assumed_action_cost_sum",
    "assumption_based_payoff_sum",
    "selected_event_rate",
    "historical_mapped_rate",
)


@dataclass(frozen=True)
class StrategyCondition:
    """Declare one closed, pure-data strategy condition tree.

    Summary
    -------
    A shallow-frozen, pure-data declaration consumed by the private Task 16
    condition kernel only through :func:`simulate_decision_strategy`.

    Attributes
    ----------
    kind
        Required. ``atomic``, ``and``, ``or``, or ``not``.
    operator
        Default: ``None``. Boolean nodes have no operator; an atomic node must
        declare one approved comparison or missingness operator.
    left_kind
        Default: ``None``. Boolean nodes request no left source; atomic nodes must
        declare ``column``, ``ranking_score``, or ``event_probability``.
    left
        Default: ``None``. Score sources and Boolean nodes have no column name;
        a column source must declare its column name.
    right_kind
        Default: ``None``. Missingness operators and Boolean nodes have no right
        source; other atomic operators declare ``literal`` or ``column``.
    right
        Default: ``None``. No right operand is supplied for missingness operators
        or Boolean nodes; other atomic operators supply closed pure-data evidence.
    children
        Default: ``()``. No child conditions are supplied; Boolean nodes must add
        the arity required by their kind.

    Validation / Errors
    -------------------
    Construction performs no whole-tree validation. The simulation entry validates
    every node before row access and raises ``ValueError`` for invalid shape,
    literals, source declarations, or inherited Task 16 resource budgets.

    Missing / Unavailable behavior
    ------------------------------
    Missing or incompatible operands use the private Task 16 kernel's three-valued
    truth and route an action-path unknown to the declared safe unknown action.

    Side effects / Immutability
    ---------------------------
    Instances are shallow frozen and consumed read-only. Conditions perform no file,
    network, production-decision, or optimization side effect.

    Examples
    --------
    >>> StrategyCondition("atomic", "ge", "column", "score", "literal", 0.5)
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
    left_kind: Literal["column", "ranking_score", "event_probability"] | None = None
    left: str | None = None
    right_kind: Literal["literal", "column"] | None = None
    right: object | None = None
    children: tuple[StrategyCondition, ...] = ()


@dataclass(frozen=True)
class DecisionRule:
    """Declare one ordered eligibility or decision rule.

    Summary
    -------
    A shallow-frozen rule declaration evaluated only by the public simulation entry.

    Attributes
    ----------
    rule_key
        Required. Stable caller-owned rule identity.
    phase
        Required. ``eligibility`` or ``decision`` execution phase.
    priority
        Required. Ascending integer priority within the phase.
    condition
        Required. Closed :class:`StrategyCondition` tree.
    action_name
        Required. Caller-declared action key produced when the rule applies.
    stop_on_hit
        Default: ``True``. The first applicable hit stops later rule application in
        the same ordered path while diagnostics remain available.
    enabled
        Default: ``True``. The rule participates when its effective window is active.
    effective_from
        Default: ``None``. The rule inherits the strategy start time.
    expires_at
        Default: ``None``. The rule inherits the strategy expiry, including no expiry.
    description_key
        Default: ``None``. No safe description metadata key is retained.

    Validation / Errors
    -------------------
    Construction performs no cross-field validation. The public simulation entry
    validates keys, priorities, action roles, effective bounds, and the complete
    condition tree and raises ``ValueError`` for invalid declarations.

    Missing / Unavailable behavior
    ------------------------------
    A disabled or out-of-window rule is inactive, not unknown, and contributes no
    row-level action while retaining typed inactive summary evidence.

    Side effects / Immutability
    ---------------------------
    Instances are shallow frozen. Simulation does not mutate the rule or execute a
    production approval.

    Examples
    --------
    >>> DecisionRule("income", "decision", 10, condition, "review")  # doctest: +SKIP
    """

    rule_key: str
    phase: Literal["eligibility", "decision"]
    priority: int
    condition: StrategyCondition
    action_name: str
    stop_on_hit: bool = True
    enabled: bool = True
    effective_from: datetime | None = None
    expires_at: datetime | None = None
    description_key: str | None = None


@dataclass(frozen=True)
class DecisionConstraint:
    """Declare one non-optimising strategy evidence constraint.

    Summary
    -------
    A shallow-frozen comparison declaration that measures, but never changes, a
    simulated strategy.

    Attributes
    ----------
    constraint_key
        Required. Stable caller-owned constraint identity.
    metric
        Required. One frozen Task 17 constraint metric.
    operator
        Required. ``le`` or ``ge`` comparison direction.
    threshold
        Required. Caller-declared business threshold.
    action_name
        Default: ``None``. No action-key scope is requested unless the metric
        requires one.
    action_role
        Default: ``None``. No closed-role scope is requested unless the metric
        requires one.
    minimum_support
        Default: ``1``. At least one supporting row is required before comparison.

    Validation / Errors
    -------------------
    Cross-field scope, threshold, operator, and support validation occurs in
    :func:`simulate_decision_strategy`; invalid declarations raise ``ValueError``.

    Missing / Unavailable behavior
    ------------------------------
    Missing sources produce typed not-applicable or not-verifiable evidence;
    insufficient support is undefined. Constraints never repair missing evidence.

    Side effects / Immutability
    ---------------------------
    Instances are shallow frozen. A constraint only measures frozen simulated
    actions and never changes rules, cutoffs, or row decisions.

    Examples
    --------
    >>> DecisionConstraint("review_cap", "review_rate", "le", 0.2, action_role="review")
    """

    constraint_key: str
    metric: Literal[
        "action_count",
        "action_rate",
        "selected_rate",
        "rejected_rate",
        "review_count",
        "review_rate",
        "request_information_rate",
        "selected_exposure_sum",
        "expected_loss_sum",
        "expected_loss_rate",
        "observed_loss_sum",
        "observed_loss_rate",
        "selected_event_rate",
    ]
    operator: Literal["le", "ge"]
    threshold: float
    action_name: str | None = None
    action_role: (
        Literal[
            "selected", "rejected", "review", "request_information", "limited", "other"
        ]
        | None
    ) = None
    minimum_support: int = 1


@dataclass(frozen=True)
class DecisionStrategyConfig:
    """Configure one deterministic offline strategy simulation.

    Summary
    -------
    A shallow-frozen declaration of rules, actions, scopes, and optional evidence;
    it never learns, optimizes, or executes a production decision.

    Attributes
    ----------
    strategy_key
        Required. Stable caller-owned strategy identity.
    strategy_version
        Required. Stable caller-owned strategy version.
    effective_from
        Required. Explicit inclusive strategy start; no current clock is read.
    expires_at
        Required. ``None`` means the strategy has no exclusive expiry bound.
    evaluation_time
        Required. Explicit evaluation time used for all effective-window checks.
    rules
        Required. Ordered tuple of :class:`DecisionRule` declarations; ``()`` is a
        valid explicitly supplied inventory with no rules.
    default_action_name
        Required. Action used when no active decision rule applies.
    unknown_action_name
        Required. Safe action used when an action-path condition is unknown.
    action_role_mapping
        Required. Complete tuple mapping every declared action to one closed role.
    constraints
        Default: ``()``. No evidence-only constraints are evaluated.
    ranking_score_column
        Default: ``None``. No DataFrame ranking-score column is requested; Task 17
        has no DataFrame event-probability entry.
    ranking_score_direction
        Default: ``None``. No DataFrame ranking direction is declared; a declared
        ranking column requires an explicit direction and is never a probability.
    historical_action_column
        Default: ``None``. No historical action column is requested.
    historical_action_mapping
        Default: ``()``. No historical raw-value-to-action mappings are supplied.
    historical_policy_version
        Default: ``None``. No sanitized historical policy version is retained.
    exposure_column
        Default: ``None``. No row-level exposure evidence is requested.
    loss_fraction
        Default: ``None``. No constant or DataFrame loss-fraction evidence is
        requested.
    action_assumptions
        Default: ``()``. No action value or cost assumptions are supplied.
    exposure_unit
        Default: ``None``. No common opaque exposure/loss unit is declared.
    segment_columns
        Default: ``()``. No segment or segment-time stability scopes are produced.
    time_slice_column
        Default: ``None``. No time-slice or segment-time scopes are produced.

    Validation / Errors
    -------------------
    Construction does not validate cross-field relationships. The simulation entry
    validates the whole config before row evaluation and raises stable ``ValueError``
    prefixes for invalid config, input, conditions, alignment, or resource limits.

    Missing / Unavailable behavior
    ------------------------------
    Optional undeclared sources remain typed not-applicable evidence. Unknown rule
    truth routes to ``unknown_action_name`` rather than the default action.

    Side effects / Immutability
    ---------------------------
    Instances are shallow frozen and are consumed read-only. The configuration
    describes offline simulation only, not approval execution or optimization.

    Examples
    --------
    >>> DecisionStrategyConfig(  # doctest: +SKIP
    ...     "policy", "v1", start, None, at, (), "select", "review",
    ...     (("select", "selected"), ("review", "review")),
    ... )
    """

    strategy_key: str
    strategy_version: str
    effective_from: datetime
    expires_at: datetime | None
    evaluation_time: datetime
    rules: tuple[DecisionRule, ...]
    default_action_name: str
    unknown_action_name: str
    action_role_mapping: tuple[
        tuple[
            str,
            Literal[
                "selected",
                "rejected",
                "review",
                "request_information",
                "limited",
                "other",
            ],
        ],
        ...,
    ]
    constraints: tuple[DecisionConstraint, ...] = ()
    ranking_score_column: str | None = None
    ranking_score_direction: Literal["higher_risk", "lower_risk"] | None = None
    historical_action_column: str | None = None
    historical_action_mapping: tuple[tuple[object, str], ...] = ()
    historical_policy_version: str | None = None
    exposure_column: str | None = None
    loss_fraction: float | str | None = None
    action_assumptions: tuple[tuple[str, float, float], ...] = ()
    exposure_unit: str | None = None
    segment_columns: tuple[str, ...] = ()
    time_slice_column: str | None = None


@dataclass(frozen=True)
class DecisionStrategyResult:
    """Contain one sanitized offline decision-strategy result.

    Summary
    -------
    A shallow-frozen result container owning newly allocated, typed evidence tables.

    Attributes
    ----------
    strategy_key
        Required. Sanitized strategy identity.
    strategy_version
        Required. Sanitized strategy version.
    strategy_fingerprint
        Required. Deterministic lowercase SHA-256 configuration fingerprint.
    input_n_rows
        Required. Exact input row count.
    decided_n_rows
        Required. Exact count of rows assigned a simulated action.
    unavailable_n_rows
        Required. Exact count of rows without an active simulated action.
    requested_rule_count
        Required. Exact requested rule count.
    active_rule_count
        Required. Exact effective rule count.
    requested_constraint_count
        Required. Exact requested constraint count.
    row_decisions
        Required. Newly allocated typed row-decision table.
    rule_evaluations
        Required. Newly allocated typed diagnostic rule-evaluation table.
    rule_summary
        Required. Newly allocated typed rule-summary table.
    action_summary
        Required. Newly allocated typed action-summary table.
    business_summary
        Required. Newly allocated typed business-summary table.
    constraint_summary
        Required. Newly allocated typed constraint-summary table.
    historical_transitions
        Required. Newly allocated typed paired historical-transition table.
    provenance
        Required. Newly allocated typed sanitized-provenance table.
    warnings
        Required. Deterministically ordered tuple; ``()`` means no warning applies.
    limitations
        Required. Deterministically ordered tuple; ``()`` means no limitation applies.

    Validation / Errors
    -------------------
    Results are created only after complete public validation. Invalid declarations
    raise before construction; callers should not construct this result as a validator.

    Missing / Unavailable behavior
    ------------------------------
    Tables preserve nullable dtypes and independent status/reason fields. Missing
    evidence is not converted to zero, and inactive rows have no final action.

    Side effects / Immutability
    ---------------------------
    The dataclass is shallow frozen and owns newly allocated tables. It retains no
    input DataFrame, config, condition, estimator, or Figure and executes no action.

    Examples
    --------
    >>> result = simulate_decision_strategy(frame, config)  # doctest: +SKIP
    >>> result.decided_n_rows + result.unavailable_n_rows == result.input_n_rows
    True
    """

    strategy_key: str
    strategy_version: str
    strategy_fingerprint: str
    input_n_rows: int
    decided_n_rows: int
    unavailable_n_rows: int
    requested_rule_count: int
    active_rule_count: int
    requested_constraint_count: int
    row_decisions: pd.DataFrame
    rule_evaluations: pd.DataFrame
    rule_summary: pd.DataFrame
    action_summary: pd.DataFrame
    business_summary: pd.DataFrame
    constraint_summary: pd.DataFrame
    historical_transitions: pd.DataFrame
    provenance: pd.DataFrame
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]


def _config_error(key: str) -> ValueError:
    return ValueError(f"decision strategy config is invalid: {key}")


def _input_error(key: str) -> ValueError:
    return ValueError(f"decision strategy input schema is invalid: {key}")


def _condition_error(key: str) -> ValueError:
    return ValueError(f"decision condition is invalid: {key}")


def _alignment_error(key: str) -> ValueError:
    return ValueError(f"strategy source alignment: {key}")


def _resource_error(key: str) -> ValueError:
    return ValueError(f"decision strategy resource limit exceeded: {key}")


def _key(value: object, key: str) -> str:
    if type(value) is not str:
        raise _config_error(key)
    if len(value) > 64:
        raise _resource_error(key)
    if _KEY_RE.fullmatch(value) is None:
        raise _config_error(key)
    return value


def _is_real(value: object) -> bool:
    return type(value) is not bool and isinstance(value, Real)


def _aware(value: datetime) -> bool:
    try:
        return value.tzinfo is not None and value.utcoffset() is not None
    except (TypeError, ValueError):
        return False


def _safe_scalar(value: object, *, config: bool = False) -> object:
    if not _is_allowed_scalar(value):
        raise (
            _config_error("unsupported_scalar_type")
            if config
            else _input_error("unsupported_scalar_type")
        )
    try:
        return _normalize(value, specification=config)
    except ValueError as exc:
        raise (
            _config_error("unsupported_scalar_type")
            if config
            else _input_error("unsupported_scalar_type")
        ) from exc


def _is_missing_safe(value: object) -> bool:
    value = _safe_scalar(value)
    return (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or (type(value) is float and np.isnan(value))
    )


def _scalar_family(value: object) -> str:
    value = _safe_scalar(value, config=True)
    if value is None or value is pd.NA or value is pd.NaT:
        return "missing"
    if type(value) is bool:
        return "bool"
    if type(value) is int:
        return "int"
    if type(value) is float:
        return "float"
    if type(value) is str:
        return "str"
    if type(value) is date:
        return "date"
    if type(value) in (datetime, pd.Timestamp):
        return "datetime"
    raise _config_error("unsupported_scalar_type")


def _scalar_identity(value: object) -> tuple[str, object]:
    value = _safe_scalar(value)
    if _is_missing_safe(value):
        return ("missing", 0)
    family = _scalar_family(value)
    if family == "float":
        if not isfinite(value):
            return ("float", value.hex())
        return ("float", value.hex())
    if family == "datetime":
        return ("datetime", value.isoformat())
    if family == "date":
        return ("date", value.isoformat())
    return (family, value)


def _canonical_scalar(value: object) -> object:
    value = _safe_scalar(value, config=True)
    if value is None or value is pd.NA or value is pd.NaT:
        return {"type": "missing"}
    family = _scalar_family(value)
    if family == "float":
        if not isfinite(value):
            raise _config_error("nonfinite_literal")
        payload: object = value.hex()
    elif family in {"date", "datetime"}:
        payload = value.isoformat()
    else:
        payload = value
    return {"type": family, "value": payload}


def _canonical_condition(condition: StrategyCondition) -> dict[str, object]:
    def condition_scalar(value: object) -> object:
        try:
            return _canonical_scalar(value)
        except ValueError as exc:
            key = str(exc).rsplit(":", 1)[-1].strip()
            raise _condition_error(key) from exc

    return {
        "kind": condition.kind,
        "operator": condition.operator,
        "left_kind": condition.left_kind,
        "left": condition.left,
        "right_kind": condition.right_kind,
        "right": (
            [condition_scalar(item) for item in condition.right]
            if type(condition.right) is tuple
            else (
                condition_scalar(condition.right)
                if condition.right is not None
                else None
            )
        ),
        "children": [_canonical_condition(child) for child in condition.children],
    }


def _fingerprint(config: DecisionStrategyConfig) -> str:
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "strategy_key": config.strategy_key,
        "strategy_version": config.strategy_version,
        "effective_from": config.effective_from.isoformat(),
        "expires_at": config.expires_at.isoformat() if config.expires_at else None,
        "evaluation_time": config.evaluation_time.isoformat(),
        "rules": [
            {
                "rule_key": rule.rule_key,
                "phase": rule.phase,
                "priority": rule.priority,
                "condition": _canonical_condition(rule.condition),
                "action": rule.action_name,
                "stop": rule.stop_on_hit,
                "enabled": rule.enabled,
                "effective_from": (
                    rule.effective_from.isoformat() if rule.effective_from else None
                ),
                "expires_at": rule.expires_at.isoformat() if rule.expires_at else None,
                "description_key": rule.description_key,
            }
            for rule in config.rules
        ],
        "default_action": config.default_action_name,
        "unknown_action": config.unknown_action_name,
        "roles": list(config.action_role_mapping),
        "constraints": [
            {
                "constraint_key": item.constraint_key,
                "metric": item.metric,
                "operator": item.operator,
                "threshold": float(item.threshold),
                "action_name": item.action_name,
                "action_role": item.action_role,
                "minimum_support": item.minimum_support,
            }
            for item in config.constraints
        ],
        "ranking_score_column": config.ranking_score_column,
        "ranking_score_direction": config.ranking_score_direction,
        "historical_action_column": config.historical_action_column,
        "historical_action_mapping": [
            [_canonical_scalar(raw), action]
            for raw, action in config.historical_action_mapping
        ],
        "historical_policy_version": config.historical_policy_version,
        "exposure_column": config.exposure_column,
        "loss_fraction": (
            config.loss_fraction
            if type(config.loss_fraction) is str or config.loss_fraction is None
            else _canonical_scalar(config.loss_fraction)
        ),
        "action_assumptions": [
            [action, float(value), float(cost)]
            for action, value, cost in config.action_assumptions
        ],
        "exposure_unit": config.exposure_unit,
        "segment_columns": list(config.segment_columns),
        "time_slice_column": config.time_slice_column,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _frame(
    rows: list[dict[str, object]],
    columns: tuple[str, ...],
    dtypes: dict[str, str],
) -> pd.DataFrame:
    result = pd.DataFrame(rows, columns=columns)
    for column in columns:
        dtype = dtypes[column]
        if dtype == "object":
            result[column] = pd.Series(result[column].tolist(), dtype="object")
        else:
            result[column] = pd.array(result[column], dtype=dtype)
    result.index = pd.RangeIndex(len(result))
    return result


_ROW_DTYPES = {
    **{column: "string" for column in _ROW_COLUMNS},
    "row_position": "int64",
    "matched_rule_count": "int64",
    "unknown_rule_count": "int64",
    "overlap_rule_count": "int64",
    "conflict_rule_count": "int64",
    "override_applied": "boolean",
}
_RULE_EVALUATION_DTYPES = {
    **{column: "string" for column in _RULE_EVALUATION_COLUMNS},
    "row_position": "int64",
    "priority": "int64",
    "rule_order": "int64",
    "is_applied": "boolean",
    "is_overlap": "boolean",
    "is_conflict": "boolean",
}
_SUMMARY_DTYPES = {
    **{column: "string" for column in _RULE_SUMMARY_COLUMNS},
    "scope_ordinal": "Int64",
    "time_slice_ordinal": "Int64",
    "priority": "int64",
    "metric_value": "Float64",
    "numerator": "Float64",
    "denominator": "Float64",
    "support_n_rows": "int64",
}
_ACTION_SUMMARY_DTYPES = {
    key: value
    for key, value in _SUMMARY_DTYPES.items()
    if key in _ACTION_SUMMARY_COLUMNS
}
_CONSTRAINT_DTYPES = {
    **{column: "string" for column in _CONSTRAINT_COLUMNS},
    "threshold": "float64",
    "actual_value": "Float64",
    "support_n": "int64",
    "gap": "Float64",
    "violation_magnitude": "Float64",
}
_TRANSITION_DTYPES = {
    **{column: "string" for column in _TRANSITION_COLUMNS},
    "row_count": "int64",
    "row_rate": "Float64",
}
_PROVENANCE_DTYPES = {column: "string" for column in _PROVENANCE_COLUMNS}


def _validate_top_level(data: object, config: object) -> None:
    if type(config) is not DecisionStrategyConfig:
        raise _config_error("config_type")
    if type(data) is not pd.DataFrame:
        raise _input_error("data_not_dataframe")
    if len(data) > 100_000:
        raise _resource_error("input_rows")


def _validate_datetime_window(
    start: object, end: object, evaluation: object | None, key: str
) -> None:
    if type(start) is not datetime or (end is not None and type(end) is not datetime):
        raise _config_error(key)
    if evaluation is not None and type(evaluation) is not datetime:
        raise _config_error("evaluation_time")
    values = [value for value in (start, end, evaluation) if value is not None]
    if any(_aware(value) != _aware(values[0]) for value in values[1:]):
        raise _config_error(key)
    if end is not None and not start < end:
        raise _config_error(key)


def _validate_config(
    config: DecisionStrategyConfig,
) -> tuple[dict[str, str], dict[str, int]]:
    _key(config.strategy_key, "strategy_key")
    _key(config.strategy_version, "strategy_version")
    _validate_datetime_window(
        config.effective_from,
        config.expires_at,
        config.evaluation_time,
        "strategy_window",
    )
    if type(config.rules) is not tuple or any(
        type(x) is not DecisionRule for x in config.rules
    ):
        raise _config_error("rules")
    if len(config.rules) > 100:
        raise _resource_error("all_rules")
    if sum(rule.phase == "eligibility" for rule in config.rules) > 50:
        raise _resource_error("eligibility_rules")
    if sum(rule.phase == "decision" for rule in config.rules) > 50:
        raise _resource_error("decision_rules")
    if type(config.constraints) is not tuple or any(
        type(x) is not DecisionConstraint for x in config.constraints
    ):
        raise _config_error("constraints")
    if len(config.constraints) > 50:
        raise _resource_error("constraints")
    if type(config.action_role_mapping) is not tuple:
        raise _config_error("action_role_mapping")
    if len(config.action_role_mapping) > 50:
        raise _resource_error("action_role_mappings")
    mapping: dict[str, str] = {}
    action_order: dict[str, int] = {}
    for ordinal, item in enumerate(config.action_role_mapping):
        if type(item) is not tuple or len(item) != 2:
            raise _config_error("action_role_mapping")
        action, role = item
        action = _key(action, "action_key")
        if type(role) is not str or role not in _ROLE_SET or action in mapping:
            raise _config_error("action_role_mapping")
        mapping[action] = role
        action_order[action] = ordinal
    default = _key(config.default_action_name, "default_action_name")
    unknown = _key(config.unknown_action_name, "unknown_action_name")
    if default not in mapping or unknown not in mapping:
        raise _config_error("required_action_role_unmapped")
    if mapping[unknown] not in _SAFE_ACTION_ROLES:
        raise _config_error("unknown_action_role")
    seen_keys: set[str] = set()
    seen_priorities: set[tuple[str, int]] = set()
    action_inventory = {default, unknown}
    for rule in config.rules:
        rule_key = _key(rule.rule_key, "rule_key")
        if rule_key in seen_keys:
            raise _config_error("duplicate_rule_key")
        seen_keys.add(rule_key)
        if type(rule.phase) is not str or rule.phase not in {"eligibility", "decision"}:
            raise _config_error("rule_phase")
        if type(rule.priority) is not int or not 0 <= rule.priority <= 9999:
            raise _config_error("rule_priority")
        priority_key = (rule.phase, rule.priority)
        if priority_key in seen_priorities:
            raise _config_error("duplicate_rule_priority")
        seen_priorities.add(priority_key)
        if type(rule.stop_on_hit) is not bool or type(rule.enabled) is not bool:
            raise _config_error("rule_flags")
        action = _key(rule.action_name, "rule_action")
        action_inventory.add(action)
        if action not in mapping:
            raise _config_error("required_action_role_unmapped")
        if rule.phase == "eligibility" and mapping[action] not in _SAFE_ACTION_ROLES:
            raise _config_error("eligibility_action_role")
        if rule.description_key is not None:
            _key(rule.description_key, "description_key")
        start = rule.effective_from or config.effective_from
        end = rule.expires_at if rule.expires_at is not None else config.expires_at
        _validate_datetime_window(start, end, config.evaluation_time, "rule_window")
        if (
            rule.effective_from is not None
            and rule.effective_from < config.effective_from
        ):
            raise _config_error("rule_window")
        if (
            config.expires_at is not None
            and rule.expires_at is not None
            and rule.expires_at > config.expires_at
        ):
            raise _config_error("rule_window")
    if set(mapping) != action_inventory:
        raise _config_error("action_role_mapping")
    _validate_sources_config(config)
    if any(action not in mapping for _, action in config.historical_action_mapping):
        raise _config_error("historical_action_mapping")
    _validate_assumptions(config, mapping)
    _validate_constraints(config, mapping)
    return mapping, action_order


def _validate_sources_config(config: DecisionStrategyConfig) -> None:
    for name in (
        "ranking_score_column",
        "historical_action_column",
        "exposure_column",
        "time_slice_column",
    ):
        value = getattr(config, name)
        if value is not None and (type(value) is not str or not value):
            raise _config_error(name)
    if config.ranking_score_column is None:
        if config.ranking_score_direction is not None:
            raise _config_error("score_direction_without_column")
    elif config.ranking_score_direction not in {"higher_risk", "lower_risk"}:
        raise _config_error("ranking_score_direction")
    if type(config.historical_action_mapping) is not tuple:
        raise _config_error("historical_action_mapping")
    if len(config.historical_action_mapping) > 50:
        raise _resource_error("historical_action_mappings")
    if (config.historical_action_column is None) != (
        len(config.historical_action_mapping) == 0
    ):
        raise _config_error("historical_action_mapping")
    seen: set[tuple[str, object]] = set()
    for item in config.historical_action_mapping:
        if type(item) is not tuple or len(item) != 2:
            raise _config_error("historical_action_mapping")
        raw, action = item
        identity = _scalar_identity_config(raw)
        if identity[0] == "missing" or identity in seen:
            raise _config_error("historical_action_mapping")
        seen.add(identity)
        _key(action, "historical_action_mapping")
    if config.historical_policy_version is not None:
        _key(config.historical_policy_version, "historical_policy_version")
    if type(config.loss_fraction) is str and not config.loss_fraction:
        raise _config_error("loss_fraction")
    if type(config.segment_columns) is not tuple or any(
        type(column) is not str or not column for column in config.segment_columns
    ):
        raise _config_error("segment_columns")
    if len(config.segment_columns) > 4:
        raise _resource_error("segment_columns")
    if len(set(config.segment_columns)) != len(config.segment_columns):
        raise _config_error("segment_columns")
    if config.time_slice_column in config.segment_columns:
        raise _config_error("scope_column_conflict")
    occupied = {
        value
        for value in (
            config.ranking_score_column,
            config.historical_action_column,
            config.exposure_column,
            config.loss_fraction if type(config.loss_fraction) is str else None,
        )
        if value is not None
    }
    if occupied & (set(config.segment_columns) | {config.time_slice_column}):
        raise _config_error("scope_column_conflict")


def _scalar_identity_config(value: object) -> tuple[str, object]:
    normalized = _safe_scalar(value, config=True)
    if normalized is None or normalized is pd.NA or normalized is pd.NaT:
        return ("missing", 0)
    family = _scalar_family(normalized)
    if family == "float":
        if not isfinite(normalized):
            raise _config_error("nonfinite_literal")
        return (family, normalized.hex())
    if family in {"date", "datetime"}:
        return (family, normalized.isoformat())
    return (family, normalized)


def _validate_assumptions(
    config: DecisionStrategyConfig, mapping: dict[str, str]
) -> None:
    if type(config.action_assumptions) is not tuple:
        raise _config_error("action_assumptions")
    if len(config.action_assumptions) > 50:
        raise _resource_error("action_assumptions")
    if config.action_assumptions:
        seen: set[str] = set()
        for item in config.action_assumptions:
            if type(item) is not tuple or len(item) != 3:
                raise _config_error("action_assumptions")
            action, value, cost = item
            action = _key(action, "action_assumptions")
            if action in seen or action not in mapping:
                raise _config_error("action_assumptions")
            seen.add(action)
            if not _is_real(value) or not isfinite(float(value)):
                raise _config_error("action_assumptions")
            if not _is_real(cost) or not isfinite(float(cost)) or float(cost) < 0:
                raise _config_error("action_assumptions")
        if seen != set(mapping):
            raise _config_error("action_assumptions")
    if config.loss_fraction is not None and type(config.loss_fraction) is not str:
        if (
            not _is_real(config.loss_fraction)
            or not isfinite(float(config.loss_fraction))
            or not 0 <= float(config.loss_fraction) <= 1
        ):
            raise _config_error("loss_fraction")
    if config.exposure_unit is not None:
        _key(config.exposure_unit, "exposure_unit")
    business_declared = (
        bool(config.action_assumptions)
        or config.exposure_column is not None
        or config.loss_fraction is not None
    )
    if business_declared and config.exposure_unit is None:
        raise _config_error("exposure_unit")


def _validate_constraints(
    config: DecisionStrategyConfig, mapping: dict[str, str]
) -> None:
    seen: set[str] = set()
    role_requirements = {
        "selected_rate": "selected",
        "rejected_rate": "rejected",
        "review_count": "review",
        "review_rate": "review",
        "request_information_rate": "request_information",
        "selected_exposure_sum": "selected",
        "selected_event_rate": "selected",
    }
    for item in config.constraints:
        key = _key(item.constraint_key, "constraint_key")
        if key in seen:
            raise _config_error("duplicate_constraint_key")
        seen.add(key)
        if item.metric not in _CONSTRAINT_METRICS or item.operator not in {"le", "ge"}:
            raise _config_error("constraint_metric")
        if (
            not _is_real(item.threshold)
            or not isfinite(float(item.threshold))
            or float(item.threshold) < 0
        ):
            raise _config_error("constraint_threshold")
        if (
            item.metric
            in {
                "action_rate",
                "selected_rate",
                "rejected_rate",
                "review_rate",
                "request_information_rate",
                "selected_event_rate",
                "expected_loss_rate",
            }
            and float(item.threshold) > 1
        ):
            raise _config_error("constraint_threshold")
        if type(item.minimum_support) is not int or item.minimum_support < 1:
            raise _config_error("constraint_minimum_support")
        if item.metric in {"action_count", "action_rate"}:
            if item.action_name not in mapping or item.action_role is not None:
                raise _config_error("constraint_scope")
        elif item.metric in role_requirements:
            if (
                item.action_name is not None
                or item.action_role != role_requirements[item.metric]
            ):
                raise _config_error("constraint_scope")
        elif item.action_name is not None or item.action_role is not None:
            raise _config_error("constraint_scope")


def _validate_input_columns(data: pd.DataFrame, config: DecisionStrategyConfig) -> None:
    if data.columns.has_duplicates:
        raise _input_error("duplicate_columns")
    if any(type(column) is not str for column in data.columns):
        raise _input_error("non_string_columns")
    if _SCORE_COLUMN in data.columns or _PROBABILITY_COLUMN in data.columns:
        raise _input_error("reserved_column_conflict")
    required = set(config.segment_columns)
    required.update(
        value
        for value in (
            config.ranking_score_column,
            config.historical_action_column,
            config.exposure_column,
            config.loss_fraction if type(config.loss_fraction) is str else None,
            config.time_slice_column,
        )
        if value is not None
    )
    for rule in config.rules:
        _collect_condition_columns(rule.condition, required)
    if any(column not in data.columns for column in required):
        raise _input_error("missing_column")


def _collect_condition_columns(condition: StrategyCondition, output: set[str]) -> None:
    if type(condition) is not StrategyCondition:
        raise _condition_error("node_type")
    if condition.kind == "atomic":
        if condition.left_kind == "column" and type(condition.left) is str:
            output.add(condition.left)
        if condition.right_kind == "column" and type(condition.right) is str:
            output.add(condition.right)
    elif type(condition.children) is tuple:
        for child in condition.children:
            _collect_condition_columns(child, output)


def _validate_numeric_column(
    data: pd.DataFrame,
    column: str,
    *,
    nonnegative: bool,
    fraction: bool = False,
    allow_missing: bool = False,
) -> np.ndarray:
    series = data[column]
    if (
        is_bool_dtype(series.dtype)
        or is_complex_dtype(series.dtype)
        or not is_numeric_dtype(series.dtype)
    ):
        raise _input_error("unsupported_numeric_column")
    array = series.to_numpy(dtype="float64", na_value=np.nan)
    missing = np.isnan(array)
    if np.isinf(array).any() or (not allow_missing and missing.any()):
        raise _input_error("nonfinite_numeric_column")
    finite = ~missing
    if nonnegative and (array[finite] < 0).any():
        raise _input_error("negative_numeric_column")
    if fraction and ((array[finite] < 0) | (array[finite] > 1)).any():
        raise _input_error("loss_fraction_range")
    return array


def _validate_data_audit(result: DataAuditResult | None, data: pd.DataFrame) -> None:
    if result is None:
        return
    if type(result) is not DataAuditResult:
        raise _alignment_error("task16_result_type")
    if result.n_rows != len(data) or result.n_columns != len(data.columns):
        raise _alignment_error("task16_shape")
    for name, schema in _AUDIT_TABLE_SCHEMAS.items():
        table = getattr(result, name)
        if type(table) is not pd.DataFrame or tuple(table.columns) != tuple(
            column for column, _ in schema
        ):
            raise _alignment_error("task16_schema")
        if tuple(str(table[column].dtype) for column, _ in schema) != tuple(
            dtype for _, dtype in schema
        ):
            raise _alignment_error("task16_schema")
    if type(result.warnings) is not tuple or type(result.limitations) is not tuple:
        raise _alignment_error("task16_schema")
    if type(result.config_fingerprint) is not str or not re.fullmatch(
        r"[0-9a-f]{64}", result.config_fingerprint
    ):
        raise _alignment_error("task16_provenance")


def _exact_position(value: object, n_rows: int) -> int:
    if type(value) is not int or not 0 <= value < n_rows:
        raise _alignment_error("row_position")
    return value


def _exact_count(value: object, key: str) -> int:
    if type(value) is not int or value < 0:
        raise _alignment_error(key)
    return value


def _task15_optional_number(
    value: object,
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    nonfinite_unavailable: bool = False,
) -> float | None:
    """Validate a consumed Task 15 numeric cell without protocol dispatch."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if not _is_allowed_scalar(value):
        raise _alignment_error(key)
    normalized = _normalize(value, specification=False)
    if type(normalized) not in (int, float) or type(normalized) is bool:
        raise _alignment_error(key)
    if type(normalized) is float and normalized != normalized:
        return None
    if not isfinite(normalized):
        if nonfinite_unavailable:
            return None
        raise _alignment_error(key)
    number = float(normalized)
    if minimum is not None and number < minimum:
        raise _alignment_error(key)
    if maximum is not None and number > maximum:
        raise _alignment_error(key)
    return number


def _task15_exact_support(value: object, key: str) -> int:
    if not _is_allowed_scalar(value):
        raise _alignment_error(key)
    normalized = _normalize(value, specification=False)
    if type(normalized) is not int or normalized < 0:
        raise _alignment_error(key)
    return normalized


def _task15_optional_unit(value: object, key: str) -> str | None:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if type(value) is not str or not value:
        raise _alignment_error(key)
    return value


def _validate_risk(
    result: BinaryRiskValidationResult | None, data: pd.DataFrame
) -> dict[str, object] | None:
    if result is None:
        return None
    try:
        _validate_binary_risk_validation_result(result)
    except (TypeError, ValueError) as exc:
        raise _alignment_error("task15_schema") from exc
    n_rows = len(data)
    if result.input_n_rows != n_rows:
        raise _alignment_error("input_n_rows")
    validation_mode = result.validation_mode
    time_modes = {"time_holdout", "time_forward"}
    non_time_modes = {
        "stratified_holdout",
        "stratified_kfold",
        "group_holdout",
        "group_kfold",
    }
    if validation_mode not in time_modes | non_time_modes:
        raise _alignment_error("validation_mode")
    if result.score_direction != "higher_positive_event_risk":
        raise _alignment_error("score_direction")
    if type(result.positive_label) not in (str, int, bool):
        raise _alignment_error("positive_label_provenance")
    predictions = result.predictions
    excluded = result.excluded_rows
    prediction_positions = tuple(
        _exact_position(value, n_rows) for value in predictions["row_position"].tolist()
    )
    if prediction_positions != tuple(sorted(prediction_positions)) or len(
        set(prediction_positions)
    ) != len(prediction_positions):
        raise _alignment_error("prediction_positions")
    excluded_positions = tuple(
        _exact_position(value, n_rows) for value in excluded["row_position"].tolist()
    )
    if len(set(excluded_positions)) != len(excluded_positions):
        raise _alignment_error("excluded_positions")
    if excluded_positions != tuple(sorted(excluded_positions)):
        raise _alignment_error("excluded_positions")
    fold_validation: list[int] = []
    fold_evaluable: list[int] = []
    membership: dict[int, object] = {}
    fold_ids: list[int] = []
    fold_maturity: list[tuple[int, int, int, int, int]] = []
    for row in result.folds.to_dict("records"):
        fold_id = row["fold_id"]
        if type(fold_id) is not int:
            raise _alignment_error("fold_id")
        fold_ids.append(fold_id)
        validation = row["validation_row_positions"]
        evaluable = row["evaluable_validation_row_positions"]
        if type(validation) is not tuple or type(evaluable) is not tuple:
            raise _alignment_error("fold_positions_type")
        valid_tuple = tuple(_exact_position(x, n_rows) for x in validation)
        eval_tuple = tuple(_exact_position(x, n_rows) for x in evaluable)
        if (
            valid_tuple != tuple(sorted(valid_tuple))
            or eval_tuple != tuple(sorted(eval_tuple))
            or len(set(valid_tuple)) != len(valid_tuple)
            or len(set(eval_tuple)) != len(eval_tuple)
            or not set(eval_tuple) <= set(valid_tuple)
        ):
            raise _alignment_error("fold_positions")
        for position in valid_tuple:
            if position in membership:
                raise _alignment_error("fold_position_duplicate")
            membership[position] = fold_id
        fold_validation.extend(valid_tuple)
        fold_evaluable.extend(eval_tuple)
        validation_n = _exact_count(row["validation_n"], "maturity_count_mismatch")
        evaluable_validation_n = _exact_count(
            row["evaluable_validation_n"], "maturity_count_mismatch"
        )
        validation_mature_n = _exact_count(
            row["validation_mature_n"], "maturity_count_mismatch"
        )
        validation_excluded_n = _exact_count(
            row["validation_excluded_n"], "maturity_count_mismatch"
        )
        immature_validation_n = _exact_count(
            row["immature_validation_n"], "maturity_count_mismatch"
        )
        if validation_n != len(valid_tuple) or evaluable_validation_n != len(
            eval_tuple
        ):
            raise _alignment_error("maturity_count_mismatch")
        fold_maturity.append(
            (
                validation_n,
                evaluable_validation_n,
                validation_mature_n,
                validation_excluded_n,
                immature_validation_n,
            )
        )
    if fold_ids != list(range(len(fold_ids))):
        raise _alignment_error("fold_id")
    fold_validation_tuple = tuple(sorted(fold_validation))
    fold_evaluable_tuple = tuple(sorted(fold_evaluable))
    if prediction_positions != fold_validation_tuple:
        raise _alignment_error("prediction_fold_union")
    evaluable_flags = predictions["is_evaluable"].tolist()
    if any(type(flag) not in (bool, np.bool_) for flag in evaluable_flags):
        raise _alignment_error("prediction_is_evaluable")
    evaluable_positions = tuple(
        position
        for position, flag in zip(prediction_positions, evaluable_flags, strict=True)
        if bool(flag)
    )
    if evaluable_positions != fold_evaluable_tuple:
        raise _alignment_error("evaluable_fold_union")
    for row in predictions.to_dict("records"):
        position = row["row_position"]
        if type(row["fold_id"]) is not int:
            raise _alignment_error("prediction_fold_id")
        if membership.get(position) != row["fold_id"]:
            raise _alignment_error("prediction_fold_id")
    for (
        validation_n,
        evaluable_validation_n,
        validation_mature_n,
        validation_excluded_n,
        immature_validation_n,
    ) in fold_maturity:
        if validation_mode in time_modes:
            if (
                validation_mature_n != evaluable_validation_n
                or validation_excluded_n != immature_validation_n
                or validation_excluded_n != validation_n - evaluable_validation_n
            ):
                raise _alignment_error("time_mode_maturity_mismatch")
        elif any(
            count != 0
            for count in (
                validation_mature_n,
                validation_excluded_n,
                immature_validation_n,
            )
        ):
            raise _alignment_error("non_time_maturity_count_nonzero")
    if set(prediction_positions) & set(excluded_positions):
        raise _alignment_error("predicted_excluded_overlap")
    if sorted((*prediction_positions, *excluded_positions)) != list(range(n_rows)):
        raise _alignment_error("source_scope_coverage")
    missing_target_n = int((excluded["reason"] == "missing_target").sum())
    if (
        result.predicted_n_rows != len(prediction_positions)
        or result.evaluable_n_rows != len(evaluable_positions)
        or result.eligible_n_rows != n_rows - missing_target_n
    ):
        raise _alignment_error("top_level_counts")
    return {
        "prediction_positions": prediction_positions,
        "evaluable_positions": evaluable_positions,
    }


def _validate_condition_trees(config: DecisionStrategyConfig) -> None:
    columns: set[str] = {_SCORE_COLUMN, _PROBABILITY_COLUMN}
    for rule in config.rules:
        _collect_condition_columns(rule.condition, columns)
    empty = pd.DataFrame(
        {column: pd.Series(dtype="float64") for column in sorted(columns)}
    )
    for rule in config.rules:
        start = rule.effective_from or config.effective_from
        end = rule.expires_at if rule.expires_at is not None else config.expires_at
        node = _compile_condition(
            rule.condition,
            version=config.strategy_version,
            effective_from=start,
            expires_at=end,
            data_score_direction=config.ranking_score_direction,
        )
        try:
            _evaluate_condition(empty, node, evaluation_time=config.evaluation_time)
        except ValueError as exc:
            raise _translate_kernel_error(exc) from exc


def _compile_condition(
    condition: StrategyCondition,
    *,
    version: str,
    effective_from: datetime | None,
    expires_at: datetime | None,
    data_score_direction: str | None,
    root: bool = True,
    active: set[int] | None = None,
) -> _ConditionNode:
    if type(condition) is not StrategyCondition:
        raise _condition_error("node_type")
    active = set() if active is None else active
    if id(condition) in active:
        raise _condition_error("condition_cycle")
    active.add(id(condition))
    try:
        if type(condition.children) is not tuple:
            raise _condition_error("children")
        if condition.kind == "atomic":
            if (
                condition.children
                or condition.operator not in _OPERATORS
                or condition.left_kind
                not in {"column", "ranking_score", "event_probability"}
            ):
                raise _condition_error("atomic_shape")
            if condition.left_kind == "column":
                if type(condition.left) is not str or not condition.left:
                    raise _condition_error("left_operand")
                left = condition.left
            else:
                if condition.left is not None:
                    raise _condition_error("left_operand")
                left = (
                    _SCORE_COLUMN
                    if condition.left_kind == "ranking_score"
                    else _PROBABILITY_COLUMN
                )
            operator = condition.operator
            if operator in {"is_missing", "is_not_missing"}:
                if condition.right_kind is not None or condition.right is not None:
                    raise _condition_error("right_operand")
                right = None
            else:
                if condition.right_kind not in {"literal", "column"}:
                    raise _condition_error("right_operand")
                if condition.left_kind != "column" and (
                    condition.right_kind != "literal"
                    or operator in {"in", "not_in", "between"}
                ):
                    raise _condition_error("score_operator")
                if condition.right_kind == "column":
                    if type(condition.right) is not str or not condition.right:
                        raise _condition_error("right_operand")
                    right = _ConditionOperand("column", condition.right)
                else:
                    right = _ConditionOperand("literal", condition.right)
            if (
                condition.left_kind == "ranking_score"
                and data_score_direction == "lower_risk"
            ):
                operator = {"gt": "lt", "ge": "le", "lt": "gt", "le": "ge"}.get(
                    operator, operator
                )
            return _ConditionNode(
                "atomic",
                operator,
                _ConditionOperand("column", left),
                right,
                (),
                effective_from if root else None,
                expires_at if root else None,
                version if root else None,
            )
        if (
            condition.kind not in {"and", "or", "not"}
            or condition.operator is not None
            or condition.left_kind is not None
            or condition.left is not None
            or condition.right_kind is not None
            or condition.right is not None
        ):
            raise _condition_error("boolean_shape")
        if (condition.kind in {"and", "or"} and len(condition.children) < 2) or (
            condition.kind == "not" and len(condition.children) != 1
        ):
            raise _condition_error("children")
        children = tuple(
            _compile_condition(
                child,
                version=version,
                effective_from=None,
                expires_at=None,
                data_score_direction=data_score_direction,
                root=False,
                active=active,
            )
            for child in condition.children
        )
        return _ConditionNode(
            condition.kind,
            None,
            None,
            None,
            children,
            effective_from if root else None,
            expires_at if root else None,
            version if root else None,
        )
    finally:
        active.remove(id(condition))


def _translate_kernel_error(exc: ValueError) -> ValueError:
    message = str(exc)
    key = message.rsplit(":", 1)[-1].strip()
    if key in {
        "condition_depth_exceeded",
        "condition_nodes_exceeded",
        "membership_budget_exceeded",
        "string_budget_exceeded",
    }:
        return _resource_error(key)
    if message.startswith("condition evaluation is invalid"):
        return _input_error(key)
    return _condition_error(key)


def _prepare_sources(
    data: pd.DataFrame,
    config: DecisionStrategyConfig,
    risk: BinaryRiskValidationResult | None,
    risk_meta: dict[str, object] | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    working = data.reset_index(drop=True).copy(deep=True)
    n_rows = len(data)
    ranking = np.full(n_rows, np.nan)
    probability = np.full(n_rows, np.nan)
    event = np.full(n_rows, np.nan)
    risk_ranking_values: list[float | None] = []
    risk_probability_values: list[float | None] = []
    if risk is not None:
        risk_ranking_values = [
            _task15_optional_number(value, "ranking_score")
            for value in risk.predictions["ranking_score"].tolist()
        ]
        risk_probability_values = [
            _task15_optional_number(
                value, "event_probability", minimum=0.0, maximum=1.0
            )
            for value in risk.predictions["event_probability"].tolist()
        ]
    risk_has_ranking = any(value is not None for value in risk_ranking_values)
    if config.ranking_score_column is not None and risk_has_ranking:
        raise _config_error("duplicate_score_source")
    if config.ranking_score_column is not None:
        ranking = _validate_numeric_column(
            data, config.ranking_score_column, nonnegative=False
        )
    elif risk is not None and risk_has_ranking:
        for position, value in zip(
            risk_meta["prediction_positions"],
            risk_ranking_values,
            strict=True,
        ):
            if value is None:
                raise _alignment_error("ranking_score")
            ranking[position] = value
    probability_available = False
    if risk is not None and any(value is not None for value in risk_probability_values):
        if risk.probability_provenance is None:
            raise _alignment_error("probability_provenance")
        probability_available = True
        for position, value in zip(
            risk_meta["prediction_positions"],
            risk_probability_values,
            strict=True,
        ):
            if value is None:
                continue
            probability[position] = value
    if risk is not None:
        positive = risk.positive_label
        for row in risk.predictions.to_dict("records"):
            if bool(row["is_evaluable"]):
                target = row["target_value"]
                event[row["row_position"]] = (
                    1.0
                    if type(target) is type(positive) and target == positive
                    else 0.0
                )
    working[_SCORE_COLUMN] = pd.Series(ranking, dtype="float64")
    working[_PROBABILITY_COLUMN] = pd.Series(probability, dtype="float64")
    return (
        working,
        ranking,
        probability,
        event,
        bool(np.isfinite(ranking).any()),
        probability_available,
    )


def _required_condition_sources(condition: StrategyCondition) -> tuple[bool, bool]:
    if condition.kind == "atomic":
        return (
            condition.left_kind == "ranking_score",
            condition.left_kind == "event_probability",
        )
    ranking = False
    probability = False
    for child in condition.children:
        child_ranking, child_probability = _required_condition_sources(child)
        ranking = ranking or child_ranking
        probability = probability or child_probability
    return ranking, probability


def _strategy_active(config: DecisionStrategyConfig) -> bool:
    return config.evaluation_time >= config.effective_from and (
        config.expires_at is None or config.evaluation_time < config.expires_at
    )


def _rule_active(rule: DecisionRule, config: DecisionStrategyConfig) -> bool:
    if not rule.enabled:
        return False
    start = rule.effective_from or config.effective_from
    end = rule.expires_at if rule.expires_at is not None else config.expires_at
    return config.evaluation_time >= start and (
        end is None or config.evaluation_time < end
    )


def _evaluate_rules(
    working: pd.DataFrame,
    config: DecisionStrategyConfig,
    strategy_active: bool,
) -> tuple[list[DecisionRule], dict[str, object]]:
    ordered = sorted(
        config.rules,
        key=lambda rule: (
            0 if rule.phase == "eligibility" else 1,
            rule.priority,
            rule.rule_key,
        ),
    )
    evaluations: dict[str, object] = {}
    for rule in ordered:
        start = rule.effective_from or config.effective_from
        end = rule.expires_at if rule.expires_at is not None else config.expires_at
        node = _compile_condition(
            rule.condition,
            version=config.strategy_version,
            effective_from=start,
            expires_at=end,
            data_score_direction=config.ranking_score_direction,
        )
        try:
            frame = (
                working
                if strategy_active and _rule_active(rule, config)
                else working.iloc[:0]
            )
            evaluations[rule.rule_key] = _evaluate_condition(
                frame, node, evaluation_time=config.evaluation_time
            )
        except ValueError as exc:
            raise _translate_kernel_error(exc) from exc
    return ordered, evaluations


def _assign_rows(
    data: pd.DataFrame,
    config: DecisionStrategyConfig,
    ordered: list[DecisionRule],
    evaluations: dict[str, object],
    mapping: dict[str, str],
    strategy_active: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    active_rules = (
        [rule for rule in ordered if _rule_active(rule, config)]
        if strategy_active
        else []
    )
    for position in range(len(data)):
        if not strategy_active:
            row_rows.append(
                {
                    "row_position": position,
                    "decision_status": "inactive",
                    "decision_reason": "strategy_inactive",
                    "base_action_name": pd.NA,
                    "final_action_name": pd.NA,
                    "applied_rule_key": pd.NA,
                    "matched_rule_count": 0,
                    "unknown_rule_count": 0,
                    "overlap_rule_count": 0,
                    "conflict_rule_count": 0,
                    "override_applied": False,
                    "historical_mapping_status": "not_applicable",
                }
            )
            continue
        phase_path_open = True
        base_action: str | None = None
        final_action: str | None = None
        applied_key: str | None = None
        matched: list[str] = []
        unknown: list[str] = []
        overlap: list[str] = []
        conflicts: list[str] = []
        reached_matches: list[DecisionRule] = []
        path_status: dict[str, str] = {}
        applied: dict[str, bool] = {}
        is_conflict: dict[str, bool] = {}
        eligibility_terminal = False
        for phase in ("eligibility", "decision"):
            if phase == "decision" and eligibility_terminal:
                for rule in active_rules:
                    if rule.phase == phase:
                        path_status[rule.rule_key] = "not_evaluated"
                continue
            phase_rules = [rule for rule in active_rules if rule.phase == phase]
            phase_open = phase_path_open
            for rule in phase_rules:
                result = evaluations[rule.rule_key]
                truth = result.truth.iat[position]
                if truth == "true":
                    matched.append(rule.rule_key)
                elif truth == "unknown":
                    unknown.append(rule.rule_key)
                if not phase_open:
                    path_status[rule.rule_key] = "not_evaluated"
                    if truth == "true":
                        overlap.append(rule.rule_key)
                    continue
                path_status[rule.rule_key] = "evaluated"
                if truth == "unknown":
                    final_action = config.unknown_action_name
                    phase_open = False
                    phase_path_open = False
                    if phase == "eligibility":
                        eligibility_terminal = True
                    continue
                if truth != "true":
                    continue
                reached_matches.append(rule)
                if base_action is None:
                    base_action = rule.action_name
                    final_action = rule.action_name
                    applied_key = rule.rule_key
                    applied[rule.rule_key] = True
                    if phase == "eligibility":
                        eligibility_terminal = True
                    if rule.stop_on_hit:
                        phase_open = False
                        phase_path_open = False
                else:
                    overlap.append(rule.rule_key)
                    if rule.action_name != base_action:
                        conflicts.append(rule.rule_key)
                        is_conflict[rule.rule_key] = True
                        final_action = config.unknown_action_name
                        phase_open = False
                        phase_path_open = False
                    elif rule.stop_on_hit:
                        phase_open = False
                        phase_path_open = False
            if phase == "eligibility" and base_action is not None:
                eligibility_terminal = True
        if final_action is None:
            final_action = config.default_action_name
            reason = "default_action_applied"
        elif conflicts:
            reason = "rule_conflict"
        elif unknown:
            reason = "unknown_condition"
        else:
            reason = "computed"
        for ordinal, rule in enumerate(active_rules):
            result = evaluations[rule.rule_key]
            truth = result.truth.iat[position]
            detail_rows.append(
                {
                    "row_position": position,
                    "rule_key": rule.rule_key,
                    "phase": rule.phase,
                    "priority": rule.priority,
                    "rule_order": ordinal,
                    "path_status": path_status.get(rule.rule_key, "not_evaluated"),
                    "truth": truth,
                    "status": result.status.iat[position],
                    "reason": result.reason.iat[position],
                    "is_applied": applied.get(rule.rule_key, False),
                    "is_overlap": rule.rule_key in overlap,
                    "is_conflict": is_conflict.get(rule.rule_key, False),
                }
            )
        row_rows.append(
            {
                "row_position": position,
                "decision_status": "available",
                "decision_reason": reason,
                "base_action_name": base_action if base_action is not None else pd.NA,
                "final_action_name": final_action,
                "applied_rule_key": applied_key if applied_key is not None else pd.NA,
                "matched_rule_count": len(matched),
                "unknown_rule_count": len(unknown),
                "overlap_rule_count": len(overlap),
                "conflict_rule_count": len(conflicts),
                "override_applied": base_action is not None
                and base_action != final_action,
                "historical_mapping_status": "not_applicable",
            }
        )
    return (
        _frame(row_rows, _ROW_COLUMNS, _ROW_DTYPES),
        _frame(detail_rows, _RULE_EVALUATION_COLUMNS, _RULE_EVALUATION_DTYPES),
    )


@dataclass(frozen=True)
class _Scope:
    scope_type: str
    scope_column: object
    scope_ordinal: object
    time_slice_ordinal: object
    positions: tuple[int, ...]


def _bucket_ordinals(
    data: pd.DataFrame, column: str, limit: int, key: str
) -> tuple[list[tuple[tuple[str, object], tuple[int, ...]]], dict[int, int]]:
    buckets: dict[tuple[str, object], list[int]] = {}
    missing: list[int] = []
    for position in range(len(data)):
        value = data[column].iat[position]
        identity = _scalar_identity(value)
        if identity[0] == "missing":
            missing.append(position)
        else:
            buckets.setdefault(identity, []).append(position)
    if len(buckets) + bool(missing) > limit:
        raise _resource_error(key)
    ordered = [(identity, tuple(positions)) for identity, positions in buckets.items()]
    if missing:
        ordered.append((("missing", 0), tuple(missing)))
    ordinals = {
        position: ordinal
        for ordinal, (_, positions) in enumerate(ordered)
        for position in positions
    }
    return ordered, ordinals


def _scopes(data: pd.DataFrame, config: DecisionStrategyConfig) -> list[_Scope]:
    scopes = [_Scope("overall", pd.NA, pd.NA, pd.NA, tuple(range(len(data))))]
    segment_buckets: dict[str, list[tuple[tuple[str, object], tuple[int, ...]]]] = {}
    for column in config.segment_columns:
        buckets, _ = _bucket_ordinals(data, column, 100, "segment_categories")
        segment_buckets[column] = buckets
        scopes.extend(
            _Scope("segment", column, ordinal, pd.NA, positions)
            for ordinal, (_, positions) in enumerate(buckets)
        )
    time_buckets: list[tuple[tuple[str, object], tuple[int, ...]]] = []
    time_ordinals: dict[int, int] = {}
    if config.time_slice_column is not None:
        time_buckets, time_ordinals = _bucket_ordinals(
            data, config.time_slice_column, 100, "time_slices"
        )
        scopes.extend(
            _Scope("time_slice", config.time_slice_column, pd.NA, ordinal, positions)
            for ordinal, (_, positions) in enumerate(time_buckets)
        )
    if time_buckets:
        for column in config.segment_columns:
            column_name = json.dumps(
                [column, config.time_slice_column],
                ensure_ascii=True,
                separators=(",", ":"),
            )
            for segment_ordinal, (_, segment_positions) in enumerate(
                segment_buckets[column]
            ):
                by_time: dict[int, list[int]] = {}
                for position in segment_positions:
                    by_time.setdefault(time_ordinals[position], []).append(position)
                for time_ordinal in sorted(by_time):
                    scopes.append(
                        _Scope(
                            "segment_time",
                            column_name,
                            segment_ordinal,
                            time_ordinal,
                            tuple(by_time[time_ordinal]),
                        )
                    )
    if len(scopes) - 1 > 1000:
        raise _resource_error("derived_scopes")
    return scopes


def _validate_summary_row_budget(
    scopes: list[_Scope], config: DecisionStrategyConfig, action_count: int
) -> None:
    non_overall = len(scopes) - 1
    rule_rate_count = sum(key.endswith("rate") for key in _RULE_METRICS)
    action_rate_count = sum(key.endswith("rate") for key in _ACTION_METRICS)
    business_stability_count = 11
    rule_rows = len(config.rules) * (
        len(_RULE_METRICS) + non_overall * (len(_RULE_METRICS) + 2 * rule_rate_count)
    )
    action_rows = action_count * (
        len(_ACTION_METRICS)
        + non_overall * (len(_ACTION_METRICS) + 2 * action_rate_count)
    )
    business_rows = 4 * (
        len(_BUSINESS_METRICS)
        + non_overall * (len(_BUSINESS_METRICS) + 2 * business_stability_count)
    )
    if rule_rows + action_rows + business_rows > 100_000:
        raise _resource_error("scope_summary_rows")


def _evidence_arrays(
    data: pd.DataFrame,
    config: DecisionStrategyConfig,
    row_decisions: pd.DataFrame,
    ranking: np.ndarray,
    probability: np.ndarray,
    event: np.ndarray,
) -> dict[str, object]:
    n = len(data)
    exposure = None
    if config.exposure_column is not None:
        exposure = _validate_numeric_column(
            data, config.exposure_column, nonnegative=True, allow_missing=True
        )
    loss_fraction = None
    if type(config.loss_fraction) is str:
        loss_fraction = _validate_numeric_column(
            data,
            config.loss_fraction,
            nonnegative=True,
            fraction=True,
            allow_missing=True,
        )
    elif config.loss_fraction is not None:
        loss_fraction = np.full(n, float(config.loss_fraction), dtype="float64")
    expected = np.full(n, np.nan)
    observed_assumption = np.full(n, np.nan)
    if exposure is not None and loss_fraction is not None:
        components = np.isfinite(exposure) & np.isfinite(loss_fraction)
        mask = np.isfinite(probability) & components
        expected[mask] = probability[mask] * exposure[mask] * loss_fraction[mask]
        mature = np.isfinite(event) & components
        observed_assumption[mature] = (
            event[mature] * exposure[mature] * loss_fraction[mature]
        )
    assumptions = {
        action: (float(value), float(cost))
        for action, value, cost in config.action_assumptions
    }
    assumed_value = np.full(n, np.nan)
    assumed_cost = np.full(n, np.nan)
    payoff = np.full(n, np.nan)
    if assumptions:
        for position, action in enumerate(row_decisions["final_action_name"].tolist()):
            if not pd.isna(action):
                assumed_value[position], assumed_cost[position] = assumptions[action]
                if np.isfinite(expected[position]):
                    payoff[position] = (
                        assumed_value[position]
                        - assumed_cost[position]
                        - expected[position]
                    )
    return {
        "ranking": ranking,
        "probability": probability,
        "event": event,
        "exposure": exposure,
        "loss_fraction": loss_fraction,
        "expected": expected,
        "observed_assumption": observed_assumption,
        "assumed_value": assumed_value,
        "assumed_cost": assumed_cost,
        "payoff": payoff,
        "exposure_declared": exposure is not None,
        "loss_fraction_declared": loss_fraction is not None,
        "probability_available": bool(np.isfinite(probability).any()),
        "event_available": bool(np.isfinite(event).any()),
        "assumptions_declared": bool(assumptions),
    }


def _metric(
    value: float | int | None,
    *,
    numerator: float | int | None = None,
    denominator: float | int | None = None,
    support: int,
    status: str = "available",
    reason: str = "computed",
    unit: object = pd.NA,
) -> dict[str, object]:
    return {
        "metric_value": pd.NA if value is None else float(value),
        "numerator": pd.NA if numerator is None else float(numerator),
        "denominator": pd.NA if denominator is None else float(denominator),
        "support_n_rows": support,
        "unit": unit,
        "status": status,
        "reason": reason,
    }


def _ratio(numerator: float, denominator: float, support: int) -> dict[str, object]:
    if denominator == 0:
        return _metric(
            None,
            numerator=numerator,
            denominator=denominator,
            support=support,
            status="undefined",
            reason="zero_denominator",
        )
    return _metric(
        numerator / denominator,
        numerator=numerator,
        denominator=denominator,
        support=support,
    )


def _unavailable(
    source_declared: bool, reason: str, support: int = 0
) -> dict[str, object]:
    return _metric(
        None,
        support=support,
        status="not_verifiable" if source_declared else "not_applicable",
        reason=reason if source_declared else "source_not_requested",
    )


def _incomplete_component_reason(
    evidence: dict[str, object],
    positions: list[int] | tuple[int, ...],
    *,
    probability: bool,
) -> str | None:
    if probability and any(
        not np.isfinite(evidence["probability"][position]) for position in positions
    ):
        return "probability_unavailable"
    if any(not np.isfinite(evidence["exposure"][position]) for position in positions):
        return "exposure_unavailable"
    if any(
        not np.isfinite(evidence["loss_fraction"][position]) for position in positions
    ):
        return "loss_fraction_unavailable"
    return None


def _scope_prefix(scope: _Scope) -> dict[str, object]:
    return {
        "scope_type": scope.scope_type,
        "scope_column": scope.scope_column,
        "scope_ordinal": scope.scope_ordinal,
        "time_slice_ordinal": scope.time_slice_ordinal,
    }


def _final_action_excluding_rule(
    position: int,
    excluded_rule_key: str,
    config: DecisionStrategyConfig,
    ordered_rules: list[DecisionRule],
    truth_lookup: dict[tuple[int, str], str],
) -> str | None:
    if not _strategy_active(config):
        return None
    active_rules = [
        rule
        for rule in ordered_rules
        if rule.rule_key != excluded_rule_key and _rule_active(rule, config)
    ]
    base_action: str | None = None
    final_action: str | None = None
    eligibility_terminal = False
    for phase in ("eligibility", "decision"):
        if phase == "decision" and eligibility_terminal:
            break
        phase_open = True
        for rule in (item for item in active_rules if item.phase == phase):
            if not phase_open:
                continue
            truth = truth_lookup[(position, rule.rule_key)]
            if truth == "unknown":
                final_action = config.unknown_action_name
                phase_open = False
                if phase == "eligibility":
                    eligibility_terminal = True
                continue
            if truth != "true":
                continue
            if base_action is None:
                base_action = rule.action_name
                final_action = rule.action_name
                if phase == "eligibility":
                    eligibility_terminal = True
                if rule.stop_on_hit:
                    phase_open = False
            elif rule.action_name != base_action:
                final_action = config.unknown_action_name
                phase_open = False
            elif rule.stop_on_hit:
                phase_open = False
    return config.default_action_name if final_action is None else final_action


def _summary_rows(
    scopes: list[_Scope],
    config: DecisionStrategyConfig,
    mapping: dict[str, str],
    action_order: dict[str, int],
    ordered_rules: list[DecisionRule],
    row_decisions: pd.DataFrame,
    rule_evaluations: pd.DataFrame,
    evidence: dict[str, object],
    risk: BinaryRiskValidationResult | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rule_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    business_rows: list[dict[str, object]] = []
    overall_rule: dict[tuple[str, str], dict[str, object]] = {}
    overall_action: dict[tuple[str, str], dict[str, object]] = {}
    overall_business: dict[tuple[object, object, str], dict[str, object]] = {}
    actions = sorted(
        mapping,
        key=lambda action: (
            _ROLE_ORDER.index(mapping[action]),
            action_order[action],
            action,
        ),
    )
    final_actions = row_decisions["final_action_name"].tolist()
    truth_lookup = {
        (int(row.row_position), str(row.rule_key)): str(row.truth)
        for row in rule_evaluations.itertuples(index=False)
    }
    roles = [
        mapping.get(action) if not pd.isna(action) else None for action in final_actions
    ]
    for scope in scopes:
        positions = np.array(scope.positions, dtype=int)
        decided_scope = tuple(
            position
            for position in scope.positions
            if not pd.isna(final_actions[position])
        )
        detail = rule_evaluations.loc[
            rule_evaluations["row_position"].isin(scope.positions)
        ]
        for rule in ordered_rules:
            subset = detail.loc[detail["rule_key"] == rule.rule_key]
            active = _rule_active(rule, config) and _strategy_active(config)
            evaluated = int((subset["path_status"] == "evaluated").sum())
            hit = int((subset["truth"] == "true").sum())
            unknown = int((subset["truth"] == "unknown").sum())
            not_evaluated = int((subset["path_status"] == "not_evaluated").sum())
            applied = int(subset["is_applied"].sum()) if not subset.empty else 0
            overlap = int(subset["is_overlap"].sum()) if not subset.empty else 0
            conflict = int(subset["is_conflict"].sum()) if not subset.empty else 0
            sole_hit = int(
                sum(
                    1
                    for position in scope.positions
                    if row_decisions.at[position, "matched_rule_count"] == 1
                    and not subset.loc[subset["row_position"] == position].empty
                    and subset.loc[subset["row_position"] == position, "truth"].iloc[0]
                    == "true"
                )
            )
            event = evidence["event"]
            mature_positive = (
                int(np.nansum(event[positions] == 1)) if len(positions) else 0
            )
            captured = int(
                sum(
                    1
                    for position in scope.positions
                    if event[position] == 1
                    and not subset.loc[subset["row_position"] == position].empty
                    and subset.loc[subset["row_position"] == position, "truth"].iloc[0]
                    == "true"
                )
            )
            metrics = {
                "evaluated_count": _metric(evaluated, support=len(scope.positions)),
                "hit_count": _metric(hit, support=len(scope.positions)),
                "hit_rate": _ratio(hit, evaluated, evaluated),
                "unknown_count": _metric(unknown, support=len(scope.positions)),
                "unknown_rate": _ratio(unknown, evaluated, evaluated),
                "not_evaluated_count": _metric(
                    not_evaluated, support=len(scope.positions)
                ),
                "applied_count": _metric(applied, support=len(scope.positions)),
                "sole_hit_count": _metric(sole_hit, support=len(scope.positions)),
                "overlap_count": _metric(overlap, support=len(scope.positions)),
                "overlap_rate": _ratio(overlap, evaluated, evaluated),
                "conflict_count": _metric(conflict, support=len(scope.positions)),
                "incremental_action_count": _metric(
                    applied, support=len(scope.positions)
                ),
                "leave_one_out_changed_action_count": _metric(
                    sum(
                        _final_action_excluding_rule(
                            position,
                            rule.rule_key,
                            config,
                            ordered_rules,
                            truth_lookup,
                        )
                        != final_actions[position]
                        for position in scope.positions
                    )
                    if active
                    else 0,
                    support=len(scope.positions),
                ),
            }
            mature_support = (
                int(np.isfinite(event[positions]).sum()) if len(positions) else 0
            )
            if risk is None:
                metrics["captured_event_count"] = _metric(
                    None,
                    support=0,
                    status="not_applicable",
                    reason="source_not_requested",
                )
                metrics["target_capture_rate"] = _metric(
                    None,
                    support=0,
                    status="not_applicable",
                    reason="source_not_requested",
                )
            elif mature_support == 0:
                metrics["captured_event_count"] = _metric(
                    None,
                    support=0,
                    status="not_verifiable",
                    reason="label_not_evaluable",
                )
                metrics["target_capture_rate"] = _metric(
                    None,
                    support=0,
                    status="not_verifiable",
                    reason="label_not_evaluable",
                )
            else:
                metrics["captured_event_count"] = _metric(
                    captured, support=mature_support
                )
                metrics["target_capture_rate"] = _ratio(
                    captured, mature_positive, mature_positive
                )
            if not active:
                metrics = {
                    key: _metric(
                        None, support=0, status="inactive", reason="rule_inactive"
                    )
                    for key in _RULE_METRICS
                }
            for metric_key in _RULE_METRICS:
                item = metrics[metric_key]
                row = {
                    **_scope_prefix(scope),
                    "phase": rule.phase,
                    "priority": rule.priority,
                    "rule_key": rule.rule_key,
                    "action_key": rule.action_name,
                    "action_role": mapping[rule.action_name],
                    "metric_key": metric_key,
                    **item,
                    "finding_key": f"rule:{rule.rule_key}",
                }
                rule_rows.append(row)
                if scope.scope_type == "overall":
                    overall_rule[(rule.rule_key, metric_key)] = item
                elif metric_key.endswith("rate"):
                    baseline = overall_rule[(rule.rule_key, metric_key)]
                    rule_rows.extend(_baseline_rows(row, baseline))
        for action in actions:
            selected_positions = [
                position
                for position in scope.positions
                if not pd.isna(final_actions[position])
                and final_actions[position] == action
            ]
            count = len(selected_positions)
            event = evidence["event"]
            mature = [
                position
                for position in selected_positions
                if np.isfinite(event[position])
            ]
            event_count = int(sum(event[position] == 1 for position in mature))
            if risk is None:
                event_unavailable = _metric(
                    None,
                    support=0,
                    status="not_applicable",
                    reason="source_not_requested",
                )
                event_count_metric = evaluable_count_metric = event_rate_metric = (
                    event_unavailable
                )
            elif not mature:
                event_unavailable = _metric(
                    None,
                    support=0,
                    status="not_verifiable",
                    reason="label_not_evaluable",
                )
                event_count_metric = evaluable_count_metric = event_rate_metric = (
                    event_unavailable
                )
            else:
                evaluable_count_metric = _metric(
                    len(mature), support=len(mature), unit="rows"
                )
                event_count_metric = _metric(
                    event_count, support=len(mature), unit="rows"
                )
                event_rate_metric = _ratio(event_count, len(mature), len(mature))
            metrics: dict[str, dict[str, object]] = {
                "action_count": _metric(count, support=len(decided_scope), unit="rows"),
                "action_rate": _ratio(count, len(decided_scope), len(decided_scope)),
                "evaluable_event_count": evaluable_count_metric,
                "event_count": event_count_metric,
                "event_rate": event_rate_metric,
            }
            for key, array_name in (
                ("exposure_sum", "exposure"),
                ("expected_loss_sum", "expected"),
                ("assumption_based_observed_event_loss_sum", "observed_assumption"),
                ("assumed_action_value_sum", "assumed_value"),
                ("assumed_action_cost_sum", "assumed_cost"),
                ("assumption_based_payoff_sum", "payoff"),
            ):
                array = evidence[array_name]
                if array_name == "exposure":
                    if not evidence["exposure_declared"]:
                        metrics[key] = _metric(
                            None,
                            support=0,
                            status="not_applicable",
                            reason="source_not_requested",
                        )
                    else:
                        metrics[key] = _metric(
                            float(np.sum(array[selected_positions]))
                            if selected_positions
                            else 0.0,
                            support=count,
                            unit=config.exposure_unit,
                        )
                elif array_name == "observed_assumption":
                    if risk is None or not (
                        evidence["exposure_declared"]
                        and evidence["loss_fraction_declared"]
                    ):
                        metrics[key] = _metric(
                            None,
                            support=0,
                            status="not_applicable",
                            reason="source_not_requested",
                        )
                        continue
                    mature_positions = [
                        position
                        for position in selected_positions
                        if np.isfinite(evidence["event"][position])
                    ]
                    if not mature_positions:
                        metrics[key] = _metric(
                            None,
                            support=0,
                            status="not_verifiable",
                            reason="label_not_evaluable",
                        )
                    else:
                        missing_reason = _incomplete_component_reason(
                            evidence, mature_positions, probability=False
                        )
                        if missing_reason is not None:
                            metrics[key] = _metric(
                                None,
                                support=0,
                                status="not_verifiable",
                                reason=missing_reason,
                            )
                        else:
                            metrics[key] = _metric(
                                float(np.sum(array[mature_positions])),
                                support=len(mature_positions),
                                unit=config.exposure_unit,
                            )
                elif array_name in {"assumed_value", "assumed_cost"}:
                    if not evidence["assumptions_declared"]:
                        metrics[key] = _metric(
                            None,
                            support=0,
                            status="not_applicable",
                            reason="action_assumption_not_declared",
                        )
                    else:
                        metrics[key] = _metric(
                            float(np.sum(array[selected_positions]))
                            if selected_positions
                            else 0.0,
                            support=count,
                            unit=config.exposure_unit,
                        )
                else:
                    if array_name == "payoff" and not evidence["assumptions_declared"]:
                        metrics[key] = _metric(
                            None,
                            support=0,
                            status="not_applicable",
                            reason="action_assumption_not_declared",
                        )
                    elif risk is None or not (
                        evidence["exposure_declared"]
                        and evidence["loss_fraction_declared"]
                    ):
                        metrics[key] = _metric(
                            None,
                            support=0,
                            status="not_applicable",
                            reason="source_not_requested",
                        )
                    elif not evidence["probability_available"]:
                        metrics[key] = _metric(
                            None,
                            support=0,
                            status="not_verifiable",
                            reason="probability_unavailable",
                        )
                    else:
                        missing_reason = _incomplete_component_reason(
                            evidence, selected_positions, probability=True
                        )
                        if missing_reason is not None:
                            metrics[key] = _metric(
                                None,
                                support=0,
                                status="not_verifiable",
                                reason=missing_reason,
                            )
                        else:
                            metrics[key] = _metric(
                                float(np.sum(array[selected_positions]))
                                if selected_positions
                                else 0.0,
                                support=count,
                                unit=config.exposure_unit,
                            )
            if not _strategy_active(config):
                metrics = {
                    key: _metric(
                        None,
                        support=0,
                        status="inactive",
                        reason="strategy_inactive",
                    )
                    for key in _ACTION_METRICS
                }
            for metric_key in _ACTION_METRICS:
                item = metrics[metric_key]
                row = {
                    **_scope_prefix(scope),
                    "action_key": action,
                    "action_role": mapping[action],
                    "metric_key": metric_key,
                    **item,
                    "finding_key": f"action:{action_order[action]}",
                }
                action_rows.append(row)
                if scope.scope_type == "overall":
                    overall_action[(action, metric_key)] = item
                elif metric_key.endswith("rate"):
                    action_rows.extend(
                        _baseline_rows(row, overall_action[(action, metric_key)])
                    )
        business_groups: list[tuple[object, object, tuple[int, ...]]] = [
            (pd.NA, pd.NA, scope.positions)
        ]
        business_groups.extend(
            [
                (
                    pd.NA,
                    "selected",
                    tuple(
                        position
                        for position in scope.positions
                        if roles[position] in {"selected", "limited"}
                    ),
                ),
                (
                    pd.NA,
                    "rejected",
                    tuple(
                        position
                        for position in scope.positions
                        if roles[position] == "rejected"
                    ),
                ),
                (
                    pd.NA,
                    "review",
                    tuple(
                        position
                        for position in scope.positions
                        if roles[position] in {"review", "request_information"}
                    ),
                ),
            ]
        )
        for action_key, action_role, group_positions in business_groups:
            metrics = _business_metrics(
                scope,
                group_positions,
                config,
                mapping,
                row_decisions,
                evidence,
                risk,
                action_role,
            )
            if not _strategy_active(config):
                metrics = {
                    key: _metric(
                        None,
                        support=0,
                        status="inactive",
                        reason="strategy_inactive",
                    )
                    for key in _BUSINESS_METRICS
                }
            for metric_key in _BUSINESS_METRICS:
                item = metrics[metric_key]
                row = {
                    **_scope_prefix(scope),
                    "action_key": action_key,
                    "action_role": action_role,
                    "metric_key": metric_key,
                    **item,
                    "finding_key": f"strategy:{config.strategy_key}",
                }
                business_rows.append(row)
                inventory_key = (
                    None if pd.isna(action_key) else action_key,
                    None if pd.isna(action_role) else action_role,
                    metric_key,
                )
                if scope.scope_type == "overall":
                    overall_business[inventory_key] = item
                elif metric_key in {
                    "decided_rate",
                    "observed_event_rate",
                    "selected_rate",
                    "rejected_rate",
                    "review_capacity_rate",
                    "unknown_action_rate",
                    "expected_loss_rate",
                    "selected_event_rate",
                    "event_probability_mean",
                    "exposure_sum",
                    "assumption_based_payoff_sum",
                }:
                    business_rows.extend(
                        _baseline_rows(row, overall_business[inventory_key])
                    )
    if len(rule_rows) + len(action_rows) + len(business_rows) > 100_000:
        raise _resource_error("scope_summary_rows")
    return (
        _frame(rule_rows, _RULE_SUMMARY_COLUMNS, _SUMMARY_DTYPES),
        _frame(action_rows, _ACTION_SUMMARY_COLUMNS, _ACTION_SUMMARY_DTYPES),
        _frame(business_rows, _BUSINESS_SUMMARY_COLUMNS, _ACTION_SUMMARY_DTYPES),
    )


def _baseline_rows(
    row: dict[str, object], baseline: dict[str, object]
) -> list[dict[str, object]]:
    metric_key = row["metric_key"]
    baseline_row = {**row, "metric_key": f"{metric_key}_overall_baseline", **baseline}
    if row["status"] == "available" and baseline["status"] == "available":
        delta = _metric(
            float(row["metric_value"]) - float(baseline["metric_value"]),
            support=int(row["support_n_rows"]),
        )
    else:
        delta = _metric(
            None,
            support=int(row["support_n_rows"]),
            status=str(row["status"]),
            reason=str(row["reason"]),
        )
    return [
        baseline_row,
        {**row, "metric_key": f"{metric_key}_absolute_delta", **delta},
    ]


def _array_stats(
    array: np.ndarray,
    positions: tuple[int, ...],
    declared: bool,
    unavailable_reason: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    values = array[list(positions)] if positions else np.array([], dtype=float)
    finite = values[np.isfinite(values)]
    if not declared:
        item = _unavailable(False, unavailable_reason)
        return item, item, item
    if len(finite) == 0:
        item = _metric(
            None, support=0, status="not_verifiable", reason=unavailable_reason
        )
        return item, item, item
    return (
        _metric(float(finite.mean()), support=len(finite)),
        _metric(float(finite.min()), support=len(finite)),
        _metric(float(finite.max()), support=len(finite)),
    )


def _business_metrics(
    scope: _Scope,
    positions: tuple[int, ...],
    config: DecisionStrategyConfig,
    mapping: dict[str, str],
    row_decisions: pd.DataFrame,
    evidence: dict[str, object],
    risk: BinaryRiskValidationResult | None,
    action_role: object,
) -> dict[str, dict[str, object]]:
    final = row_decisions["final_action_name"].tolist()
    roles = [mapping.get(action) if not pd.isna(action) else None for action in final]
    n = len(positions)
    decided = sum(
        final[position] is not pd.NA and not pd.isna(final[position])
        for position in positions
    )
    selected = sum(roles[position] in {"selected", "limited"} for position in positions)
    rejected = sum(roles[position] == "rejected" for position in positions)
    review = sum(
        roles[position] in {"review", "request_information"} for position in positions
    )
    unknown = sum(
        not pd.isna(final[position]) and final[position] == config.unknown_action_name
        for position in positions
    )
    ranking_stats = _array_stats(
        evidence["ranking"],
        positions,
        bool(np.isfinite(evidence["ranking"]).any()),
        "score_unavailable",
    )
    probability_stats = _array_stats(
        evidence["probability"],
        positions,
        bool(np.isfinite(evidence["probability"]).any()),
        "probability_unavailable",
    )
    event = evidence["event"]
    mature = [position for position in positions if np.isfinite(event[position])]
    events = int(sum(event[position] == 1 for position in mature))
    result: dict[str, dict[str, object]] = {
        "row_count": _metric(n, support=n, unit="rows"),
        "decided_rate": _ratio(decided, n, n),
        "ranking_score_mean": ranking_stats[0],
        "ranking_score_min": ranking_stats[1],
        "ranking_score_max": ranking_stats[2],
        "event_probability_mean": probability_stats[0],
        "event_probability_min": probability_stats[1],
        "event_probability_max": probability_stats[2],
        "selected_rate": _ratio(selected, decided, decided),
        "rejected_rate": _ratio(rejected, decided, decided),
        "review_capacity_rate": _ratio(review, decided, decided),
        "unknown_action_rate": _ratio(unknown, decided, decided),
    }
    if risk is None:
        result["observed_event_count"] = result["observed_event_rate"] = _metric(
            None,
            support=0,
            status="not_applicable",
            reason="source_not_requested",
        )
    elif not mature:
        result["observed_event_count"] = result["observed_event_rate"] = _metric(
            None,
            support=0,
            status="not_verifiable",
            reason="label_not_evaluable",
        )
    else:
        result["observed_event_count"] = _metric(
            events, support=len(mature), unit="rows"
        )
        result["observed_event_rate"] = _ratio(events, len(mature), len(mature))
    for metric_key, array_name, missing_reason in (
        ("exposure_sum", "exposure", "exposure_unavailable"),
        ("expected_loss_sum", "expected", "probability_unavailable"),
        (
            "assumption_based_observed_event_loss_sum",
            "observed_assumption",
            "label_not_evaluable",
        ),
        ("assumed_action_value_sum", "assumed_value", "action_assumption_not_declared"),
        ("assumed_action_cost_sum", "assumed_cost", "action_assumption_not_declared"),
        ("assumption_based_payoff_sum", "payoff", "action_assumption_not_declared"),
    ):
        array = evidence[array_name]
        if array_name == "exposure":
            requested = bool(evidence["exposure_declared"])
            available = requested
        elif array_name == "expected":
            requested = risk is not None and bool(
                evidence["exposure_declared"] and evidence["loss_fraction_declared"]
            )
            available = requested and bool(evidence["probability_available"])
        elif array_name == "observed_assumption":
            requested = risk is not None and bool(
                evidence["exposure_declared"] and evidence["loss_fraction_declared"]
            )
            available = requested and bool(mature)
        elif array_name in {"assumed_value", "assumed_cost"}:
            requested = bool(evidence["assumptions_declared"])
            available = requested
        else:
            if not evidence["assumptions_declared"]:
                result[metric_key] = _metric(
                    None,
                    support=0,
                    status="not_applicable",
                    reason="action_assumption_not_declared",
                )
                continue
            requested = risk is not None and bool(
                evidence["exposure_declared"] and evidence["loss_fraction_declared"]
            )
            available = requested and bool(evidence["probability_available"])
        if not requested:
            reason = (
                "action_assumption_not_declared"
                if array_name in {"assumed_value", "assumed_cost"}
                else "source_not_requested"
            )
            result[metric_key] = _metric(
                None, support=0, status="not_applicable", reason=reason
            )
        elif not available:
            result[metric_key] = _metric(
                None, support=0, status="not_verifiable", reason=missing_reason
            )
        elif array_name == "observed_assumption":
            mature_positions = tuple(
                position
                for position in positions
                if np.isfinite(evidence["event"][position])
            )
            if not mature_positions:
                result[metric_key] = _metric(
                    None,
                    support=0,
                    status="not_verifiable",
                    reason="label_not_evaluable",
                )
            else:
                component_reason = _incomplete_component_reason(
                    evidence, mature_positions, probability=False
                )
                if component_reason is not None:
                    result[metric_key] = _metric(
                        None,
                        support=0,
                        status="not_verifiable",
                        reason=component_reason,
                    )
                else:
                    result[metric_key] = _metric(
                        float(np.sum(array[list(mature_positions)])),
                        support=len(mature_positions),
                        unit=config.exposure_unit,
                    )
        elif any(not np.isfinite(array[position]) for position in positions):
            component_reason = (
                _incomplete_component_reason(evidence, positions, probability=True)
                if array_name in {"expected", "payoff"}
                else missing_reason
            )
            result[metric_key] = _metric(
                None,
                support=0,
                status="not_verifiable",
                reason=component_reason or missing_reason,
            )
        else:
            result[metric_key] = _metric(
                float(np.sum(array[list(positions)])) if positions else 0.0,
                support=n,
                unit=config.exposure_unit,
            )
    exposure = evidence["exposure"]
    expected = evidence["expected"]
    if risk is None or exposure is None or not evidence["loss_fraction_declared"]:
        result["expected_loss_rate"] = _unavailable(False, "exposure_unavailable")
    elif not evidence["probability_available"]:
        result["expected_loss_rate"] = _metric(
            None, support=0, status="not_verifiable", reason="probability_unavailable"
        )
    elif any(not np.isfinite(expected[position]) for position in positions):
        component_reason = _incomplete_component_reason(
            evidence, positions, probability=True
        )
        result["expected_loss_rate"] = _metric(
            None,
            support=0,
            status="not_verifiable",
            reason=component_reason or "probability_unavailable",
        )
    else:
        result["expected_loss_rate"] = _ratio(
            float(np.sum(expected[list(positions)])),
            float(np.sum(exposure[list(positions)])),
            n,
        )
    result["selected_event_rate"] = (
        _ratio(
            sum(
                event[position] == 1 and roles[position] in {"selected", "limited"}
                for position in mature
            ),
            sum(roles[position] in {"selected", "limited"} for position in mature),
            sum(roles[position] in {"selected", "limited"} for position in mature),
        )
        if risk is not None
        else _unavailable(False, "label_not_evaluable")
    )
    historical_status = row_decisions["historical_mapping_status"].tolist()
    result["historical_mapped_rate"] = (
        _ratio(
            sum(historical_status[position] == "available" for position in positions),
            n,
            n,
        )
        if config.historical_action_column is not None
        else _unavailable(False, "historical_action_unavailable")
    )
    actual_sum, actual_rate = _actual_observed_loss(risk)
    if scope.scope_type != "overall" or not pd.isna(action_role):
        actual_sum = actual_rate = _metric(
            None,
            support=0,
            status="not_verifiable",
            reason="observed_loss_not_resegmentable",
        )
    result["actual_observed_loss_sum"] = actual_sum
    result["actual_observed_loss_rate"] = actual_rate
    return result


def _actual_observed_loss(
    risk: BinaryRiskValidationResult | None,
) -> tuple[dict[str, object], dict[str, object]]:
    if risk is None:
        item = _unavailable(False, "observed_loss_not_mature")
        return item, item
    table = risk.business_metrics
    overall = table.loc[table["segment_kind"] == "all"]
    loss_rows = overall.loc[overall["metric"] == "observed_loss_sum"]
    exposure_rows = overall.loc[overall["metric"] == "exposure_sum"]
    if len(loss_rows) != 1:
        item = _metric(
            None, support=0, status="not_verifiable", reason="observed_loss_not_mature"
        )
        return item, item
    loss_row = loss_rows.iloc[0]
    if type(loss_row["status"]) is not str:
        raise _alignment_error("observed_loss_evidence")
    if loss_row["status"] != "available":
        item = _metric(
            None, support=0, status="not_verifiable", reason="observed_loss_not_mature"
        )
        return item, item
    loss = _task15_optional_number(
        loss_row["value"],
        "observed_loss_evidence",
        nonfinite_unavailable=True,
    )
    support = _task15_exact_support(
        loss_row["n_observed_loss_mature_rows"], "observed_loss_evidence"
    )
    loss_unit = _task15_optional_unit(loss_row["unit"], "observed_loss_evidence")
    if loss is None or loss_unit is None:
        item = _metric(
            None, support=0, status="not_verifiable", reason="observed_loss_not_mature"
        )
        return item, item
    total = _metric(loss, support=support, unit=loss_unit)
    if len(exposure_rows) != 1:
        item = _metric(
            None,
            support=support,
            status="not_verifiable",
            reason="exposure_unavailable",
        )
        return item, item
    exposure_row = exposure_rows.iloc[0]
    if type(exposure_row["status"]) is not str:
        raise _alignment_error("observed_loss_evidence")
    if exposure_row["status"] != "available":
        item = _metric(
            None,
            support=support,
            status="not_verifiable",
            reason="exposure_unavailable",
        )
        return item, item
    exposure = _task15_optional_number(
        exposure_row["value"],
        "observed_loss_evidence",
        minimum=0.0,
        nonfinite_unavailable=True,
    )
    exposure_support = _task15_exact_support(
        exposure_row["n_observed_loss_mature_rows"], "observed_loss_evidence"
    )
    exposure_unit = _task15_optional_unit(
        exposure_row["unit"], "observed_loss_evidence"
    )
    if exposure is None or exposure_support != support or exposure_unit != loss_unit:
        item = _metric(
            None,
            support=support,
            status="not_verifiable",
            reason="exposure_unavailable",
        )
        return item, item
    return total, _ratio(loss, exposure, support)


def _historical(
    data: pd.DataFrame,
    config: DecisionStrategyConfig,
    mapping: dict[str, str],
    action_order: dict[str, int],
    row_decisions: pd.DataFrame,
) -> pd.DataFrame:
    if config.historical_action_column is None:
        return _frame([], _TRANSITION_COLUMNS, _TRANSITION_DTYPES)
    raw_mapping = {
        _scalar_identity_config(raw): action
        for raw, action in config.historical_action_mapping
    }
    counts: dict[tuple[str, str], int] = {}
    unmapped = 0
    for position in range(len(data)):
        raw = data[config.historical_action_column].iat[position]
        identity = _scalar_identity(raw)
        historical = raw_mapping.get(identity)
        simulated = row_decisions.at[position, "final_action_name"]
        if historical is None or pd.isna(simulated):
            unmapped += 1
            row_decisions.at[position, "historical_mapping_status"] = "not_verifiable"
            continue
        row_decisions.at[position, "historical_mapping_status"] = "available"
        counts[(historical, simulated)] = counts.get((historical, simulated), 0) + 1

    def order(action: str) -> tuple[int, int, str]:
        return (_ROLE_ORDER.index(mapping[action]), action_order[action], action)

    rows = [
        {
            "historical_action_name": historical,
            "simulated_action_name": simulated,
            "row_count": count,
            "row_rate": count / len(data) if len(data) else pd.NA,
            "status": "available",
            "reason": "computed",
            "finding_key": "historical:transition",
        }
        for (historical, simulated), count in sorted(
            counts.items(), key=lambda item: (order(item[0][0]), order(item[0][1]))
        )
    ]
    if unmapped:
        rows.append(
            {
                "historical_action_name": pd.NA,
                "simulated_action_name": pd.NA,
                "row_count": unmapped,
                "row_rate": unmapped / len(data) if len(data) else pd.NA,
                "status": "not_verifiable",
                "reason": "historical_action_unmapped",
                "finding_key": "historical:transition",
            }
        )
    return _frame(rows, _TRANSITION_COLUMNS, _TRANSITION_DTYPES)


def _constraint_value(
    constraint: DecisionConstraint,
    config: DecisionStrategyConfig,
    mapping: dict[str, str],
    row_decisions: pd.DataFrame,
    evidence: dict[str, object],
    risk: BinaryRiskValidationResult | None,
) -> dict[str, object]:
    actions = row_decisions["final_action_name"].tolist()
    decided_positions = [i for i, action in enumerate(actions) if not pd.isna(action)]
    roles = {position: mapping[actions[position]] for position in decided_positions}
    d = len(decided_positions)
    metric = constraint.metric
    if not _strategy_active(config):
        return {
            "value": None,
            "support": 0,
            "status": "not_applicable",
            "reason": "strategy_inactive",
        }
    if metric == "action_count":
        return {
            "value": sum(
                actions[p] == constraint.action_name for p in decided_positions
            ),
            "support": d,
            "status": "available",
            "reason": "computed",
        }
    if metric == "action_rate":
        numerator = sum(actions[p] == constraint.action_name for p in decided_positions)
        return _constraint_ratio(numerator, d, d)
    if metric in {
        "selected_rate",
        "rejected_rate",
        "review_count",
        "review_rate",
        "request_information_rate",
    }:
        if metric == "selected_rate":
            numerator = sum(
                roles[p] in {"selected", "limited"} for p in decided_positions
            )
        elif metric == "rejected_rate":
            numerator = sum(roles[p] == "rejected" for p in decided_positions)
        elif metric in {"review_count", "review_rate"}:
            numerator = sum(
                roles[p] in {"review", "request_information"} for p in decided_positions
            )
        else:
            numerator = sum(
                roles[p] == "request_information" for p in decided_positions
            )
        if metric == "review_count":
            return {
                "value": numerator,
                "support": d,
                "status": "available",
                "reason": "computed",
            }
        return _constraint_ratio(numerator, d, d)
    if metric == "selected_exposure_sum":
        selected = [p for p in decided_positions if roles[p] in {"selected", "limited"}]
        if evidence["exposure"] is None:
            return {
                "value": None,
                "support": 0,
                "status": "not_applicable",
                "reason": "source_not_requested",
            }
        if any(not np.isfinite(evidence["exposure"][p]) for p in selected):
            return {
                "value": None,
                "support": 0,
                "status": "not_verifiable",
                "reason": "exposure_unavailable",
            }
        return {
            "value": float(np.sum(evidence["exposure"][selected])) if selected else 0.0,
            "support": len(selected),
            "status": "available",
            "reason": "computed",
        }
    if metric in {"expected_loss_sum", "expected_loss_rate"}:
        if evidence["exposure"] is None or evidence["loss_fraction"] is None:
            return {
                "value": None,
                "support": 0,
                "status": "not_applicable",
                "reason": "source_not_requested",
            }
        if risk is None:
            return {
                "value": None,
                "support": 0,
                "status": "not_applicable",
                "reason": "source_not_requested",
            }
        if not evidence["probability_available"]:
            return {
                "value": None,
                "support": 0,
                "status": "not_verifiable",
                "reason": "probability_unavailable",
            }
        if any(not np.isfinite(evidence["expected"][p]) for p in decided_positions):
            reason = _incomplete_component_reason(
                evidence, decided_positions, probability=True
            )
            return {
                "value": None,
                "support": 0,
                "status": "not_verifiable",
                "reason": reason or "probability_unavailable",
            }
        total = float(np.sum(evidence["expected"][decided_positions]))
        if metric == "expected_loss_sum":
            return {
                "value": total,
                "support": d,
                "status": "available",
                "reason": "computed",
            }
        return _constraint_ratio(
            total, float(np.sum(evidence["exposure"][decided_positions])), d
        )
    if metric in {"observed_loss_sum", "observed_loss_rate"}:
        total, rate = _actual_observed_loss(risk)
        item = total if metric.endswith("sum") else rate
        return {
            "value": None
            if pd.isna(item["metric_value"])
            else float(item["metric_value"]),
            "support": item["support_n_rows"],
            "status": item["status"],
            "reason": item["reason"],
        }
    selected_mature = [
        p
        for p in decided_positions
        if roles[p] in {"selected", "limited"} and np.isfinite(evidence["event"][p])
    ]
    if risk is None:
        return {
            "value": None,
            "support": 0,
            "status": "not_applicable",
            "reason": "source_not_requested",
        }
    return _constraint_ratio(
        sum(evidence["event"][p] == 1 for p in selected_mature),
        len(selected_mature),
        len(selected_mature),
    )


def _constraint_ratio(
    numerator: float, denominator: float, support: int
) -> dict[str, object]:
    if denominator == 0:
        return {
            "value": None,
            "support": support,
            "status": "undefined",
            "reason": "zero_denominator",
        }
    return {
        "value": numerator / denominator,
        "support": support,
        "status": "available",
        "reason": "computed",
    }


def _constraints(
    config: DecisionStrategyConfig,
    mapping: dict[str, str],
    row_decisions: pd.DataFrame,
    evidence: dict[str, object],
    risk: BinaryRiskValidationResult | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for constraint in config.constraints:
        item = _constraint_value(
            constraint, config, mapping, row_decisions, evidence, risk
        )
        value = item["value"]
        status = item["status"]
        reason = item["reason"]
        if status == "available" and item["support"] < constraint.minimum_support:
            value, status, reason = None, "undefined", "insufficient_support"
        if status == "available":
            gap = (
                float(value) - float(constraint.threshold)
                if constraint.operator == "ge"
                else float(constraint.threshold) - float(value)
            )
            violation = max(0.0, -gap)
            reason = "constraint_satisfied" if gap >= 0 else "constraint_failed"
        else:
            gap = violation = None
        rows.append(
            {
                "constraint_key": constraint.constraint_key,
                "metric": constraint.metric,
                "operator": constraint.operator,
                "threshold": float(constraint.threshold),
                "action_name": constraint.action_name
                if constraint.action_name is not None
                else pd.NA,
                "action_role": constraint.action_role
                if constraint.action_role is not None
                else pd.NA,
                "actual_value": pd.NA if value is None else float(value),
                "status": status,
                "reason": reason,
                "support_n": int(item["support"]),
                "gap": pd.NA if gap is None else gap,
                "violation_magnitude": pd.NA if violation is None else violation,
                "finding_key": f"constraint:{constraint.constraint_key}",
            }
        )
    return _frame(rows, _CONSTRAINT_COLUMNS, _CONSTRAINT_DTYPES)


def _risk_fingerprint(result: BinaryRiskValidationResult | None) -> str | None:
    if result is None:
        return None
    payload = {
        "validation_mode": result.validation_mode,
        "prediction_scope": result.prediction_scope,
        "score_source": result.score_source,
        "score_direction": result.score_direction,
        "probability_provenance": result.probability_provenance,
        "positive_label_type_family": _scalar_family(result.positive_label),
        "input_n_rows": result.input_n_rows,
        "eligible_n_rows": result.eligible_n_rows,
        "predicted_n_rows": result.predicted_n_rows,
        "evaluable_n_rows": result.evaluable_n_rows,
        "folds": [
            {
                "fold_id": row["fold_id"],
                "validation": list(row["validation_row_positions"]),
                "evaluable": list(row["evaluable_validation_row_positions"]),
                "validation_n": row["validation_n"],
                "validation_mature_n": row["validation_mature_n"],
                "validation_excluded_n": row["validation_excluded_n"],
                "evaluable_validation_n": row["evaluable_validation_n"],
                "immature_validation_n": row["immature_validation_n"],
            }
            for row in result.folds.to_dict("records")
        ],
        "excluded_positions": result.excluded_rows["row_position"].tolist(),
        "predictions": [
            {
                "position": row["row_position"],
                "fold": row["fold_id"],
                "evaluable": bool(row["is_evaluable"]),
                "ranking": None
                if pd.isna(row["ranking_score"])
                else float(row["ranking_score"]).hex(),
                "probability": None
                if pd.isna(row["event_probability"])
                else float(row["event_probability"]).hex(),
            }
            for row in result.predictions.to_dict("records")
        ],
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _provenance(
    config: DecisionStrategyConfig,
    strategy_fingerprint: str,
    risk: BinaryRiskValidationResult | None,
    audit: DataAuditResult | None,
    ranking_available: bool,
    probability_available: bool,
) -> pd.DataFrame:
    risk_fp = _risk_fingerprint(risk)
    values: list[tuple[str, object, str, str]] = [
        ("strategy_schema_version", _SCHEMA_VERSION, "available", "computed"),
        ("condition_kernel_version", _KERNEL_VERSION, "available", "computed"),
        ("strategy_key", config.strategy_key, "available", "computed"),
        ("strategy_version", config.strategy_version, "available", "computed"),
        (
            "strategy_effective_from",
            config.effective_from.isoformat(),
            "available",
            "computed",
        ),
        (
            "strategy_expires_at",
            config.expires_at.isoformat() if config.expires_at else None,
            "available" if config.expires_at else "not_applicable",
            "computed" if config.expires_at else "source_not_requested",
        ),
        (
            "evaluation_time",
            config.evaluation_time.isoformat(),
            "available",
            "computed",
        ),
        ("strategy_fingerprint", strategy_fingerprint, "available", "computed"),
        ("row_identity", "zero_based_row_position", "available", "computed"),
        (
            "score_source",
            "dataframe"
            if config.ranking_score_column
            else ("task15" if risk is not None and ranking_available else None),
            "available" if ranking_available else "not_applicable",
            "computed" if ranking_available else "source_not_requested",
        ),
        (
            "score_direction",
            config.ranking_score_direction
            or (
                risk.score_direction if risk is not None and ranking_available else None
            ),
            "available" if ranking_available else "not_applicable",
            "computed" if ranking_available else "source_not_requested",
        ),
        (
            "probability_provenance",
            risk.probability_provenance
            if probability_available and risk is not None
            else None,
            "available"
            if probability_available
            else ("not_verifiable" if risk is not None else "not_applicable"),
            "computed"
            if probability_available
            else (
                "probability_unavailable"
                if risk is not None
                else "source_not_requested"
            ),
        ),
        (
            "prediction_scope",
            risk.prediction_scope if risk is not None else None,
            "available" if risk is not None else "not_applicable",
            "computed" if risk is not None else "source_not_requested",
        ),
        (
            "positive_label_type_family",
            _scalar_family(risk.positive_label) if risk is not None else None,
            "available" if risk is not None else "not_applicable",
            "computed" if risk is not None else "source_not_requested",
        ),
        (
            "task15_evidence_status",
            "aligned" if risk is not None else None,
            "available" if risk is not None else "not_applicable",
            "computed" if risk is not None else "source_not_requested",
        ),
        (
            "task15_evidence_version",
            _TASK15_VERSION if risk is not None else None,
            "available" if risk is not None else "not_applicable",
            "computed" if risk is not None else "source_not_requested",
        ),
        (
            "task15_evidence_fingerprint",
            risk_fp,
            "available" if risk_fp else "not_applicable",
            "computed" if risk_fp else "source_not_requested",
        ),
        (
            "task16_evidence_status",
            "aligned" if audit is not None else None,
            "available" if audit is not None else "not_applicable",
            "computed" if audit is not None else "source_not_requested",
        ),
        (
            "task16_config_fingerprint",
            audit.config_fingerprint if audit is not None else None,
            "available" if audit is not None else "not_applicable",
            "computed" if audit is not None else "source_not_requested",
        ),
        (
            "historical_policy_version",
            config.historical_policy_version,
            "available" if config.historical_policy_version else "not_applicable",
            "computed" if config.historical_policy_version else "source_not_requested",
        ),
        (
            "historical_mapping_count",
            len(config.historical_action_mapping),
            "available",
            "computed",
        ),
        (
            "action_mapping_count",
            len(config.action_role_mapping),
            "available",
            "computed",
        ),
        (
            "action_assumption_count",
            len(config.action_assumptions),
            "available",
            "computed",
        ),
        ("rule_count", len(config.rules), "available", "computed"),
        ("constraint_count", len(config.constraints), "available", "computed"),
        ("segment_column_count", len(config.segment_columns), "available", "computed"),
        (
            "time_slice_declared",
            str(config.time_slice_column is not None).lower(),
            "available",
            "computed",
        ),
        (
            "exposure_unit",
            config.exposure_unit,
            "available" if config.exposure_unit else "not_applicable",
            "computed" if config.exposure_unit else "source_not_requested",
        ),
    ]
    rows = [
        {
            "provenance_key": key,
            "provenance_value": pd.NA if value is None else str(value),
            "status": status,
            "reason": reason,
            "finding_key": f"strategy:{config.strategy_key}",
        }
        for key, value, status, reason in values
    ]
    return _frame(rows, _PROVENANCE_COLUMNS, _PROVENANCE_DTYPES)


def simulate_decision_strategy(
    data: pd.DataFrame,
    config: DecisionStrategyConfig,
    *,
    risk_validation: BinaryRiskValidationResult | None = None,
    data_audit: DataAuditResult | None = None,
) -> DecisionStrategyResult:
    """Simulate one caller-frozen decision strategy without executing actions.

    Parameters
    ----------
    data
        Raw or prepared DataFrame. Identity is always zero-based row position.
    config
        Frozen strategy, rule, role, constraint, and evidence declarations.
    risk_validation, data_audit
        Optional frozen Task 15 and Task 16 evidence. They are validated and
        consumed read-only; no upstream metrics or audit evidence are recomputed.

    Returns
    -------
    DecisionStrategyResult
        Eight newly allocated typed evidence tables plus sanitized provenance.

    Raises
    ------
    ValueError
        For invalid config, input schema, condition, source alignment, or resource
        limits, using the five stable Task 17 error prefixes.

    Notes
    -----
    Inputs are never modified. Missing condition operands remain three-valued and
    route to the declared safe unknown action. Results are offline simulations,
    not production approvals, causal effects, or optimized policy recommendations.

    Examples
    --------
    >>> frame = pd.DataFrame({"x": [1, 2]})
    >>> condition = StrategyCondition("atomic", "ge", "column", "x", "literal", 2)
    >>> rule = DecisionRule("r1", "decision", 1, condition, "select")
    >>> config = DecisionStrategyConfig(
    ...     "s", "v1", datetime(2025, 1, 1), None, datetime(2025, 1, 2),
    ...     (rule,), "select", "review", (("select", "selected"), ("review", "review")),
    ... )
    >>> simulate_decision_strategy(frame, config).decided_n_rows
    2
    """
    _validate_top_level(data, config)
    mapping, action_order = _validate_config(config)
    _validate_condition_trees(config)
    _validate_input_columns(data, config)
    risk_meta = _validate_risk(risk_validation, data)
    _validate_data_audit(data_audit, data)
    scopes = _scopes(data, config)
    _validate_summary_row_budget(scopes, config, len(mapping))
    strategy_fingerprint = _fingerprint(config)
    working, ranking, probability, event, ranking_available, probability_available = (
        _prepare_sources(data, config, risk_validation, risk_meta)
    )
    required_ranking = False
    required_probability = False
    for rule in config.rules:
        rule_ranking, rule_probability = _required_condition_sources(rule.condition)
        required_ranking = required_ranking or rule_ranking
        required_probability = required_probability or rule_probability
    if required_ranking and not ranking_available:
        raise _config_error("score_unavailable")
    if required_probability and not probability_available:
        raise _config_error("probability_unavailable")
    strategy_active = _strategy_active(config)
    active_rule_count = (
        sum(_rule_active(rule, config) for rule in config.rules)
        if strategy_active
        else 0
    )
    if len(data) * active_rule_count > 1_000_000:
        raise _resource_error("rule_evaluation_rows")
    ordered, evaluations = _evaluate_rules(working, config, strategy_active)
    row_decisions, rule_evaluations = _assign_rows(
        data, config, ordered, evaluations, mapping, strategy_active
    )
    historical_transitions = _historical(
        data, config, mapping, action_order, row_decisions
    )
    evidence = _evidence_arrays(
        data, config, row_decisions, ranking, probability, event
    )
    rule_summary, action_summary, business_summary = _summary_rows(
        scopes,
        config,
        mapping,
        action_order,
        ordered,
        row_decisions,
        rule_evaluations,
        evidence,
        risk_validation,
    )
    constraint_summary = _constraints(
        config, mapping, row_decisions, evidence, risk_validation
    )
    provenance = _provenance(
        config,
        strategy_fingerprint,
        risk_validation,
        data_audit,
        ranking_available,
        probability_available,
    )
    decided_n = int(row_decisions["final_action_name"].notna().sum())
    unavailable_n = len(data) - decided_n
    limitations = ["simulated_actions_not_executed"]
    if config.historical_action_column is not None:
        limitations.append("historical_comparison_not_causal")
    if probability_available:
        limitations.append("model_expectation_not_observed")
    if (
        risk_validation is not None
        and risk_validation.evaluable_n_rows < risk_validation.predicted_n_rows
    ):
        limitations.append("outcome_support_limited")
    if config.ranking_score_column is not None:
        limitations.append("custom_score_provenance_caller_declared")
    return DecisionStrategyResult(
        strategy_key=config.strategy_key,
        strategy_version=config.strategy_version,
        strategy_fingerprint=strategy_fingerprint,
        input_n_rows=len(data),
        decided_n_rows=decided_n,
        unavailable_n_rows=unavailable_n,
        requested_rule_count=len(config.rules),
        active_rule_count=active_rule_count,
        requested_constraint_count=len(config.constraints),
        row_decisions=row_decisions,
        rule_evaluations=rule_evaluations,
        rule_summary=rule_summary,
        action_summary=action_summary,
        business_summary=business_summary,
        constraint_summary=constraint_summary,
        historical_transitions=historical_transitions,
        provenance=provenance,
        warnings=(),
        limitations=tuple(limitations),
    )
