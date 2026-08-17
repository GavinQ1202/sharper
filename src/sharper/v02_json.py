"""Closed, non-executing JSON carriers for the Sharper v0.2 workflow.

This module is intentionally private to Task 20.  It parses only the two
versioned JSON documents approved by the Task 20 contract and returns the
already-frozen Task 20 request carriers.  It never evaluates a condition,
simulates a policy, monitors a lifecycle, or invokes an owner function.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Final, TypeAlias

from sharper.decision_strategy import (
    DecisionConstraint,
    DecisionRule,
    DecisionStrategyConfig,
    StrategyCondition,
)
from sharper.lifecycle_monitoring import (
    EarlyWarningRule,
    LifecycleMonitoringConfig,
    LifecycleState,
    MonitoringCondition,
    WarningScenario,
)
from sharper.v02_workflow import V02PostLoanRequest, V02PreLoanRequest

_JsonSource: TypeAlias = str | bytes | Path

_POLICY_SCHEMA_VERSION: Final = "task20.policy.v1"
_WARNING_SCHEMA_VERSION: Final = "task20.warning.v1"

_MAX_JSON_BYTES = 5_000_000
_MAX_NESTING_DEPTH = 16
_MAX_OBJECT_MEMBERS = 256
_MAX_ARRAY_ITEMS = 4_096
_MAX_CONDITION_NODES = 128
_MAX_POLICY_RULES = 100
_MAX_POLICY_CONSTRAINTS = 50
_MAX_WARNING_SCENARIOS = 10
_MAX_WARNING_RULES = 50
_MAX_WARNING_STATES = 50

_POLICY_FIELDS = (
    "schema_version",
    "strategy_key",
    "strategy_version",
    "effective_from",
    "expires_at",
    "evaluation_time",
    "rules",
    "default_action_name",
    "unknown_action_name",
    "action_role_mapping",
    "constraints",
    "ranking_score_column",
    "ranking_score_direction",
    "historical_action_column",
    "historical_action_mapping",
    "historical_policy_version",
    "exposure_column",
    "loss_fraction",
    "action_assumptions",
    "exposure_unit",
    "segment_columns",
    "time_slice_column",
)
_POLICY_REQUIRED = frozenset(
    {
        "schema_version",
        "strategy_key",
        "strategy_version",
        "effective_from",
        "expires_at",
        "evaluation_time",
        "rules",
        "default_action_name",
        "unknown_action_name",
        "action_role_mapping",
    }
)
_POLICY_RULE_FIELDS = (
    "rule_key",
    "phase",
    "priority",
    "condition",
    "action_name",
    "stop_on_hit",
    "enabled",
    "effective_from",
    "expires_at",
    "description_key",
)
_POLICY_RULE_REQUIRED = frozenset(_POLICY_RULE_FIELDS)
_CONDITION_FIELDS = (
    "kind",
    "operator",
    "left_kind",
    "left",
    "right_kind",
    "right",
    "children",
)
_CONDITION_REQUIRED = frozenset(_CONDITION_FIELDS)
_CONSTRAINT_FIELDS = (
    "constraint_key",
    "metric",
    "operator",
    "threshold",
    "action_name",
    "action_role",
    "minimum_support",
)
_CONSTRAINT_REQUIRED = frozenset({"constraint_key", "metric", "operator", "threshold"})

_WARNING_FIELDS = (
    "schema_version",
    "monitoring_key",
    "monitoring_version",
    "analysis_as_of",
    "entity_column",
    "observation_time_column",
    "available_time_column",
    "condition_feature_columns",
    "event_time_column",
    "positive_event_key",
    "prediction_horizon",
    "horizon_end_inclusive",
    "recent_window",
    "history_window",
    "history_start_inclusive",
    "expected_observation_interval",
    "period_unit",
    "time_zone",
    "scenarios",
    "reference_scenario_key",
    "alert_level_ranks",
    "states",
    "default_state_key",
    "unknown_state_key",
    "allowed_transitions",
    "adverse_state_keys",
    "cure_state_keys",
    "cohort_time_column",
    "cohort_column",
    "peer_group_columns",
    "peer_reference_start",
    "peer_reference_end",
    "ranking_score_column",
    "ranking_score_direction",
    "exposure_column",
    "loss_fraction",
    "observed_loss_column",
    "observed_loss_available_time_column",
    "observed_loss_is_mature_snapshot",
    "segment_columns",
    "time_frequency",
)
_WARNING_REQUIRED = frozenset(
    {
        "schema_version",
        "monitoring_key",
        "monitoring_version",
        "analysis_as_of",
        "entity_column",
        "observation_time_column",
        "available_time_column",
        "condition_feature_columns",
        "event_time_column",
        "positive_event_key",
        "prediction_horizon",
        "horizon_end_inclusive",
        "recent_window",
        "history_window",
        "history_start_inclusive",
        "expected_observation_interval",
        "period_unit",
        "time_zone",
        "scenarios",
        "reference_scenario_key",
        "alert_level_ranks",
        "states",
        "default_state_key",
        "unknown_state_key",
    }
)
_SCENARIO_FIELDS = ("scenario_key", "scenario_kind", "rules")
_RULE_FIELDS = (
    "rule_key",
    "priority",
    "alert_level",
    "condition",
    "persistence_observations",
    "resolution_observations",
    "cooldown",
    "enabled",
    "effective_from",
    "expires_at",
    "description_key",
)
_STATE_FIELDS = (
    "state_key",
    "state_rank",
    "priority",
    "condition",
    "terminal",
    "enabled",
    "description_key",
)

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
_POLICY_ROLES = frozenset(
    {"selected", "rejected", "review", "request_information", "limited", "other"}
)
_POLICY_PHASES = frozenset({"eligibility", "decision"})
_RANKING_DIRECTIONS = frozenset({"higher_risk", "lower_risk"})
_CONSTRAINT_METRICS = frozenset(
    {
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
    }
)
_PERIOD_UNITS = frozenset({"day", "week", "month", "quarter"})
_SCENARIO_KINDS = frozenset(
    {"no_alert", "single_threshold", "rule_set", "model_score", "model_plus_rules"}
)
_CONDITION_KINDS = frozenset({"atomic", "and", "or", "not"})
_POLICY_LEFT_KINDS = frozenset({"column", "ranking_score", "event_probability"})
_WARNING_LEFT_KINDS = frozenset(
    {
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
    }
)
_RIGHT_KINDS = frozenset({"literal", "column"})
_CONDITION_WINDOWS = frozenset({"recent", "history"})

_DATETIME_RE = re.compile(
    r"^(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})"
    r"T(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})"
    r"\.(?P<microsecond>[0-9]{6})(?P<utc>Z)?$",
    re.ASCII,
)


class _DuplicateKey(Exception):
    """Internal parser signal that never crosses the adapter boundary."""


class _NonFinite:
    """Sentinel retained until the scalar-validation stage."""


_NONFINITE = _NonFinite()


def _error(key: str) -> ValueError:
    return ValueError(f"sharper task20: {key}")


def _fail(key: str) -> None:
    raise _error(key)


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey
        result[key] = value
    return result


def _acquire(source: _JsonSource) -> bytes:
    if type(source) is bytes:
        return source
    if type(source) is str:
        try:
            return source.encode("utf-8")
        except UnicodeEncodeError:
            _fail("json_encoding")
    if isinstance(source, Path):
        return source.read_bytes()
    _fail("json_structure")
    raise AssertionError("unreachable")


def _parse(source: _JsonSource) -> object:
    raw = _acquire(source)
    if len(raw) > _MAX_JSON_BYTES:
        _fail("json_budget")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        _fail("json_encoding")
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_pairs,
            parse_constant=lambda _value: _NONFINITE,
        )
    except _DuplicateKey:
        _fail("json_duplicate_key")
    except JSONDecodeError:
        _fail("json_decode")
    except ValueError:
        _fail("json_scalar")
    except RecursionError:
        _fail("json_budget")
    raise AssertionError("unreachable")


def _check_object_fields(
    value: object,
    allowed: tuple[str, ...],
    required: frozenset[str],
) -> None:
    if type(value) is not dict:
        return
    if any(key not in allowed for key in value):
        _fail("json_unknown_field")
    if any(key not in value for key in required):
        _fail("json_structure")


def _check_policy_fields(document: object) -> None:
    _check_object_fields(document, _POLICY_FIELDS, _POLICY_REQUIRED)
    if type(document) is not dict:
        return
    rules = document.get("rules")
    if type(rules) is list:
        for rule in rules:
            _check_object_fields(rule, _POLICY_RULE_FIELDS, _POLICY_RULE_REQUIRED)
            if type(rule) is dict:
                condition = rule.get("condition")
                _check_policy_condition_fields(condition)
    constraints = document.get("constraints")
    if type(constraints) is list:
        for constraint in constraints:
            _check_object_fields(constraint, _CONSTRAINT_FIELDS, _CONSTRAINT_REQUIRED)


def _check_policy_condition_fields(condition: object) -> None:
    _check_object_fields(condition, _CONDITION_FIELDS, _CONDITION_REQUIRED)
    if type(condition) is not dict:
        return
    children = condition.get("children")
    if type(children) is list:
        for child in children:
            _check_policy_condition_fields(child)


def _check_warning_fields(document: object) -> None:
    _check_object_fields(document, _WARNING_FIELDS, _WARNING_REQUIRED)
    if type(document) is not dict:
        return
    scenarios = document.get("scenarios")
    if type(scenarios) is list:
        for scenario in scenarios:
            _check_object_fields(
                scenario, _SCENARIO_FIELDS, frozenset(_SCENARIO_FIELDS)
            )
            if type(scenario) is dict:
                rules = scenario.get("rules")
                if type(rules) is list:
                    for rule in rules:
                        _check_object_fields(
                            rule,
                            _RULE_FIELDS,
                            frozenset(
                                {
                                    "rule_key",
                                    "priority",
                                    "alert_level",
                                    "condition",
                                    "cooldown",
                                    "effective_from",
                                    "expires_at",
                                }
                            ),
                        )
                        if type(rule) is dict:
                            _check_warning_condition_fields(rule.get("condition"))
    states = document.get("states")
    if type(states) is list:
        for state in states:
            _check_object_fields(
                state,
                _STATE_FIELDS,
                frozenset({"state_key", "state_rank", "priority", "condition"}),
            )
            if type(state) is dict:
                _check_warning_condition_fields(state.get("condition"))


def _check_warning_condition_fields(condition: object) -> None:
    allowed = (*_CONDITION_FIELDS, "window")
    required = frozenset({"kind"})
    _check_object_fields(condition, allowed, required)
    if type(condition) is not dict:
        return
    children = condition.get("children")
    if type(children) is list:
        for child in children:
            _check_warning_condition_fields(child)


def _is_container(value: object) -> bool:
    return type(value) in {dict, list}


def _shape_scalar(value: object, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if _is_container(value):
        _fail("json_structure")


def _shape_list(value: object, *, nullable: bool = False) -> list[object] | None:
    if value is None and nullable:
        return None
    if type(value) is not list:
        _fail("json_structure")
    return value


def _shape_object(value: object) -> dict[str, object]:
    if type(value) is not dict:
        _fail("json_structure")
    return value


def _shape_pair_array(value: object) -> list[list[object]]:
    values = _shape_list(value)
    assert values is not None
    for item in values:
        if type(item) is not list:
            _fail("json_structure")
    return values  # type: ignore[return-value]


def _shape_policy(document: dict[str, object]) -> None:
    for field in (
        "strategy_key",
        "strategy_version",
        "effective_from",
        "evaluation_time",
        "default_action_name",
        "unknown_action_name",
    ):
        _shape_scalar(document[field])
    _shape_scalar(document["expires_at"], nullable=True)
    rules = _shape_list(document["rules"])
    assert rules is not None
    for rule_value in rules:
        rule = _shape_object(rule_value)
        for field in (
            "rule_key",
            "phase",
            "priority",
            "action_name",
            "stop_on_hit",
            "enabled",
        ):
            _shape_scalar(rule[field])
        _shape_scalar(rule["effective_from"], nullable=True)
        _shape_scalar(rule["expires_at"], nullable=True)
        _shape_scalar(rule["description_key"], nullable=True)
        _shape_condition(rule["condition"], warning=False)
    mappings = _shape_pair_array(document["action_role_mapping"])
    for item in mappings:
        if len(item) != 2:
            _fail("json_structure")
        _shape_scalar(item[0])
        _shape_scalar(item[1])
    if "constraints" in document:
        constraints = _shape_list(document["constraints"])
        assert constraints is not None
        for value in constraints:
            constraint = _shape_object(value)
            for field in _CONSTRAINT_FIELDS:
                if field in constraint:
                    _shape_scalar(
                        constraint[field],
                        nullable=field in {"action_name", "action_role"},
                    )
    if "historical_action_mapping" in document:
        for item in _shape_pair_array(document["historical_action_mapping"]):
            if len(item) != 2:
                _fail("json_structure")
            _shape_scalar(item[0])
            _shape_scalar(item[1])
    if "action_assumptions" in document:
        for item in _shape_pair_array(document["action_assumptions"]):
            if len(item) != 3:
                _fail("json_structure")
            for member in item:
                _shape_scalar(member)
    if "segment_columns" in document:
        for value in _shape_list(document["segment_columns"]):  # type: ignore[arg-type]
            _shape_scalar(value)
    for field in (
        "ranking_score_column",
        "ranking_score_direction",
        "historical_action_column",
        "historical_policy_version",
        "exposure_column",
        "loss_fraction",
        "exposure_unit",
        "time_slice_column",
    ):
        if field in document:
            _shape_scalar(document[field], nullable=True)


def _shape_condition(condition_value: object, *, warning: bool) -> None:
    condition = _shape_object(condition_value)
    for field in _CONDITION_FIELDS[:-1]:
        value = condition.get(field) if warning else condition[field]
        if field == "right" and type(value) is list:
            for item in value:
                _shape_scalar(item)
        else:
            _shape_scalar(value, nullable=field != "kind")
    if warning:
        _shape_scalar(condition.get("window"), nullable=True)
    children = _shape_list(
        condition.get("children", []) if warning else condition["children"]
    )
    assert children is not None
    for child in children:
        _shape_condition(child, warning=warning)


def _shape_warning(document: dict[str, object]) -> None:
    for field in (
        "monitoring_key",
        "monitoring_version",
        "analysis_as_of",
        "entity_column",
        "observation_time_column",
        "available_time_column",
        "event_time_column",
        "positive_event_key",
        "prediction_horizon",
        "horizon_end_inclusive",
        "recent_window",
        "history_window",
        "history_start_inclusive",
        "expected_observation_interval",
        "period_unit",
        "time_zone",
        "reference_scenario_key",
        "default_state_key",
        "unknown_state_key",
    ):
        _shape_scalar(
            document[field],
            nullable=field
            in {
                "event_time_column",
                "positive_event_key",
                "prediction_horizon",
                "expected_observation_interval",
                "time_zone",
            },
        )
    for field in (
        "condition_feature_columns",
        "scenarios",
        "alert_level_ranks",
        "states",
    ):
        _shape_list(document[field])
    for field in (
        "allowed_transitions",
        "adverse_state_keys",
        "cure_state_keys",
        "peer_group_columns",
        "segment_columns",
    ):
        if field in document:
            _shape_list(document[field])
    for field in (
        "cohort_time_column",
        "cohort_column",
        "peer_reference_start",
        "peer_reference_end",
        "ranking_score_column",
        "ranking_score_direction",
        "exposure_column",
        "loss_fraction",
        "observed_loss_column",
        "observed_loss_available_time_column",
        "time_frequency",
    ):
        if field in document:
            _shape_scalar(document[field], nullable=True)
    scenarios = _shape_list(document["scenarios"])
    assert scenarios is not None
    for scenario_value in scenarios:
        scenario = _shape_object(scenario_value)
        for field in _SCENARIO_FIELDS[:-1]:
            _shape_scalar(scenario[field])
        rules = _shape_list(scenario["rules"])
        assert rules is not None
        for rule_value in rules:
            rule = _shape_object(rule_value)
            for field in (
                "rule_key",
                "priority",
                "alert_level",
                "persistence_observations",
                "resolution_observations",
                "cooldown",
                "enabled",
            ):
                if field in rule:
                    _shape_scalar(rule[field])
            for field in ("effective_from", "expires_at", "description_key"):
                if field in rule:
                    _shape_scalar(rule[field], nullable=True)
            _shape_condition(rule["condition"], warning=True)
    states = _shape_list(document["states"])
    assert states is not None
    for state_value in states:
        state = _shape_object(state_value)
        for field in ("state_key", "state_rank", "priority", "terminal", "enabled"):
            if field in state:
                _shape_scalar(state[field])
        if "description_key" in state:
            _shape_scalar(state["description_key"], nullable=True)
        _shape_condition(state["condition"], warning=True)
    for item in _shape_pair_array(document["alert_level_ranks"]):
        if len(item) != 2:
            _fail("json_structure")
        _shape_scalar(item[0])
        _shape_scalar(item[1])
    if "allowed_transitions" in document:
        for item in _shape_pair_array(document["allowed_transitions"]):
            if len(item) != 2:
                _fail("json_structure")
            _shape_scalar(item[0])
            _shape_scalar(item[1])


def _json_scalar(value: object, *, nullable: bool = False) -> None:
    if value is None:
        if nullable:
            return
        _fail("json_scalar")
    if type(value) is bool:
        return
    if type(value) is int:
        return
    if type(value) is float:
        if value != value or value in {float("inf"), float("-inf")}:
            _fail("json_scalar")
        return
    if type(value) is str:
        return
    if type(value) is _NonFinite:
        _fail("json_scalar")
    if _is_container(value):
        _fail("json_structure")
    _fail("json_scalar")


def _string(value: object, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if type(value) is not str:
        _fail("json_scalar")


def _boolean(value: object) -> None:
    if type(value) is not bool:
        _fail("json_scalar")


def _integer(value: object, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if type(value) is not int:
        _fail("json_scalar")


def _number_or_string(value: object, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if type(value) is str:
        return
    if type(value) in {int, float} and type(value) is not bool:
        _json_scalar(value)
        return
    _fail("json_scalar")


def _validate_condition_scalars(condition_value: object, *, warning: bool) -> None:
    condition = _shape_object(condition_value)
    get = condition.get if warning else condition.__getitem__
    _string(get("kind"))
    _string(get("operator"), nullable=True)
    _string(get("left_kind"), nullable=True)
    _string(get("left"), nullable=True)
    _string(get("right_kind"), nullable=True)
    right = get("right")
    if type(right) is list:
        for item in right:
            _json_scalar(item)
    else:
        _json_scalar(right, nullable=True)
    if warning:
        _string(condition.get("window"), nullable=True)
    children = condition.get("children", []) if warning else condition["children"]
    for child in children:  # type: ignore[union-attr]
        _validate_condition_scalars(child, warning=warning)


def _validate_policy_scalars(document: dict[str, object]) -> None:
    for field in (
        "strategy_key",
        "strategy_version",
        "default_action_name",
        "unknown_action_name",
    ):
        _string(document[field])
    for field in ("effective_from", "evaluation_time"):
        _string(document[field])
    _string(document["expires_at"], nullable=True)
    for rule_value in document["rules"]:  # type: ignore[union-attr]
        rule = _shape_object(rule_value)
        _string(rule["rule_key"])
        _string(rule["phase"])
        _integer(rule["priority"])
        _string(rule["action_name"])
        _boolean(rule["stop_on_hit"])
        _boolean(rule["enabled"])
        _string(rule["effective_from"], nullable=True)
        _string(rule["expires_at"], nullable=True)
        _string(rule["description_key"], nullable=True)
        _validate_condition_scalars(rule["condition"], warning=False)
    for item in document["action_role_mapping"]:  # type: ignore[union-attr]
        _string(item[0])
        _string(item[1])
    if "constraints" in document:
        for constraint_value in document["constraints"]:  # type: ignore[union-attr]
            constraint = _shape_object(constraint_value)
            _string(constraint["constraint_key"])
            _string(constraint["metric"])
            _string(constraint["operator"])
            _number(constraint["threshold"])
            if "action_name" in constraint:
                _string(constraint["action_name"], nullable=True)
            if "action_role" in constraint:
                _string(constraint["action_role"], nullable=True)
            if "minimum_support" in constraint:
                _integer(constraint["minimum_support"])
    if "historical_action_mapping" in document:
        for item in document["historical_action_mapping"]:  # type: ignore[union-attr]
            _mapping_scalar(item[0])
            _string(item[1])
    if "action_assumptions" in document:
        for item in document["action_assumptions"]:  # type: ignore[union-attr]
            _string(item[0])
            _number(item[1])
            _number(item[2])
    if "segment_columns" in document:
        for value in document["segment_columns"]:  # type: ignore[union-attr]
            _string(value)
    for field in (
        "ranking_score_column",
        "historical_action_column",
        "historical_policy_version",
        "exposure_column",
        "exposure_unit",
        "time_slice_column",
    ):
        if field in document:
            _string(document[field], nullable=True)
    if "ranking_score_direction" in document:
        _string(document["ranking_score_direction"], nullable=True)
    if "loss_fraction" in document:
        _number_or_string(document["loss_fraction"], nullable=True)


def _number(value: object) -> None:
    if type(value) not in {int, float} or type(value) is bool:
        _fail("json_scalar")
    _json_scalar(value)


def _mapping_scalar(value: object) -> None:
    _json_scalar(value)
    if value is None:
        _fail("json_scalar")


def _validate_warning_scalars(document: dict[str, object]) -> None:
    for field in (
        "monitoring_key",
        "monitoring_version",
        "analysis_as_of",
        "entity_column",
        "observation_time_column",
        "available_time_column",
        "period_unit",
        "reference_scenario_key",
        "default_state_key",
        "unknown_state_key",
    ):
        _string(document[field])
    for field in ("event_time_column", "positive_event_key", "time_zone"):
        _string(document[field], nullable=True)
    for field in ("analysis_as_of",):
        _string(document[field])
    for field in (
        "prediction_horizon",
        "recent_window",
        "history_window",
        "expected_observation_interval",
    ):
        _integer(
            document[field],
            nullable=field in {"prediction_horizon", "expected_observation_interval"},
        )
    _boolean(document["horizon_end_inclusive"])
    _boolean(document["history_start_inclusive"])
    for value in document["condition_feature_columns"]:  # type: ignore[union-attr]
        _string(value)
    for value in document["scenarios"]:  # type: ignore[union-attr]
        scenario = _shape_object(value)
        _string(scenario["scenario_key"])
        _string(scenario["scenario_kind"])
        for rule_value in scenario["rules"]:  # type: ignore[union-attr]
            rule = _shape_object(rule_value)
            _string(rule["rule_key"])
            _integer(rule["priority"])
            _string(rule["alert_level"])
            if "persistence_observations" in rule:
                _integer(rule["persistence_observations"])
            if "resolution_observations" in rule:
                _integer(rule["resolution_observations"])
            if "cooldown" in rule:
                _integer(rule["cooldown"])
            if "enabled" in rule:
                _boolean(rule["enabled"])
            for field in ("effective_from", "expires_at"):
                if field in rule:
                    _string(rule[field], nullable=True)
            if "description_key" in rule:
                _string(rule["description_key"], nullable=True)
            _validate_condition_scalars(rule["condition"], warning=True)
    for value in document["states"]:  # type: ignore[union-attr]
        state = _shape_object(value)
        _string(state["state_key"])
        _integer(state["state_rank"])
        _integer(state["priority"])
        if "terminal" in state:
            _boolean(state["terminal"])
        if "enabled" in state:
            _boolean(state["enabled"])
        if "description_key" in state:
            _string(state["description_key"], nullable=True)
        _validate_condition_scalars(state["condition"], warning=True)
    for item in document["alert_level_ranks"]:  # type: ignore[union-attr]
        _string(item[0])
        _integer(item[1])
    if "allowed_transitions" in document:
        for item in document["allowed_transitions"]:  # type: ignore[union-attr]
            _string(item[0])
            _string(item[1])
    for field in (
        "adverse_state_keys",
        "cure_state_keys",
        "peer_group_columns",
        "segment_columns",
    ):
        if field in document:
            for value in document[field]:  # type: ignore[union-attr]
                _string(value)
    for field in (
        "cohort_time_column",
        "cohort_column",
        "ranking_score_column",
        "exposure_column",
        "observed_loss_column",
        "observed_loss_available_time_column",
    ):
        if field in document:
            _string(document[field], nullable=True)
    for field in ("peer_reference_start", "peer_reference_end"):
        if field in document:
            _string(document[field], nullable=True)
    if "ranking_score_direction" in document:
        _string(document["ranking_score_direction"], nullable=True)
    if "loss_fraction" in document:
        _number_or_string(document["loss_fraction"], nullable=True)
    if "observed_loss_is_mature_snapshot" in document:
        _boolean(document["observed_loss_is_mature_snapshot"])
    if "time_frequency" in document:
        _string(document["time_frequency"])


def _check_budget(document: dict[str, object], *, warning: bool) -> None:
    maximums = {"depth": 0, "objects": 0, "arrays": 0, "conditions": 0}

    def walk(value: object, depth: int) -> None:
        if type(value) is dict:
            maximums["depth"] = max(maximums["depth"], depth)
            maximums["objects"] = max(maximums["objects"], len(value))
            if "kind" in value:
                maximums["conditions"] += 1
            for child in value.values():
                if _is_container(child):
                    walk(child, depth + 1)
        elif type(value) is list:
            maximums["depth"] = max(maximums["depth"], depth)
            maximums["arrays"] = max(maximums["arrays"], len(value))
            for child in value:
                if _is_container(child):
                    walk(child, depth + 1)

    walk(document, 1)
    if maximums["depth"] > _MAX_NESTING_DEPTH:
        _fail("json_budget")
    if maximums["objects"] > _MAX_OBJECT_MEMBERS:
        _fail("json_budget")
    if maximums["arrays"] > _MAX_ARRAY_ITEMS:
        _fail("json_budget")
    if maximums["conditions"] > _MAX_CONDITION_NODES:
        _fail("json_budget")
    if warning:
        scenarios = document["scenarios"]
        if len(scenarios) > _MAX_WARNING_SCENARIOS:  # type: ignore[arg-type]
            _fail("json_budget")
        for scenario in scenarios:  # type: ignore[union-attr]
            if len(scenario["rules"]) > _MAX_WARNING_RULES:  # type: ignore[index]
                _fail("json_budget")
        if len(document["states"]) > _MAX_WARNING_STATES:  # type: ignore[arg-type]
            _fail("json_budget")
    else:
        if len(document["rules"]) > _MAX_POLICY_RULES:  # type: ignore[arg-type]
            _fail("json_budget")
        if (
            "constraints" in document
            and len(document["constraints"]) > _MAX_POLICY_CONSTRAINTS
        ):  # type: ignore[arg-type]
            _fail("json_budget")


def _parse_datetime(
    value: object, *, nullable: bool, family: bool | None
) -> tuple[datetime | None, bool | None]:
    if value is None and nullable:
        return None, family
    if type(value) is not str:
        _fail("json_scalar")
    match = _DATETIME_RE.fullmatch(value)
    if match is None:
        _fail("json_scalar")
    date_part = match.group("date")
    try:
        year, month, day = (int(part) for part in date_part.split("-"))
        result = datetime(
            year,
            month,
            day,
            int(match.group("hour")),
            int(match.group("minute")),
            int(match.group("second")),
            int(match.group("microsecond")),
            tzinfo=timezone.utc if match.group("utc") else None,
        )
    except ValueError:
        _fail("json_scalar")
    aware = match.group("utc") is not None
    if family is not None and family is not aware:
        _fail("json_scalar")
    return result, aware if family is None else family


def _duration(
    value: object, *, nullable: bool, positive: bool, name: str
) -> timedelta | None:
    if value is None and nullable:
        return None
    if type(value) is not int:
        _fail("json_scalar")
    if value < 0 or (positive and value == 0):
        _fail("json_scalar")
    try:
        return timedelta(microseconds=value)
    except OverflowError:
        _fail("json_scalar")
    raise AssertionError(name)


def _temporal_policy(document: dict[str, object]) -> dict[str, object]:
    family: bool | None = None
    parsed: dict[str, object] = {}
    for field, nullable in (
        ("effective_from", False),
        ("expires_at", True),
        ("evaluation_time", False),
    ):
        parsed[field], family = _parse_datetime(
            document[field], nullable=nullable, family=family
        )
    parsed_rules: list[dict[str, object]] = []
    for rule_value in document["rules"]:  # type: ignore[union-attr]
        rule = _shape_object(rule_value)
        parsed_rule = dict(rule)
        for field, nullable in (("effective_from", True), ("expires_at", True)):
            parsed_rule[field], family = _parse_datetime(
                rule[field], nullable=nullable, family=family
            )
        parsed_rules.append(parsed_rule)
    parsed["rules"] = parsed_rules
    return parsed


def _temporal_warning(document: dict[str, object]) -> dict[str, object]:
    family: bool | None = None
    parsed = dict(document)
    parsed["analysis_as_of"], family = _parse_datetime(
        document["analysis_as_of"], nullable=False, family=family
    )
    parsed["prediction_horizon"] = _duration(
        document["prediction_horizon"],
        nullable=True,
        positive=True,
        name="prediction_horizon",
    )
    parsed["recent_window"] = _duration(
        document["recent_window"], nullable=False, positive=True, name="recent_window"
    )
    parsed["history_window"] = _duration(
        document["history_window"], nullable=False, positive=True, name="history_window"
    )
    parsed["expected_observation_interval"] = _duration(
        document["expected_observation_interval"],
        nullable=True,
        positive=True,
        name="expected_observation_interval",
    )
    if parsed["history_window"] < parsed["recent_window"]:  # type: ignore[operator]
        _fail("json_scalar")
    for field in ("peer_reference_start", "peer_reference_end"):
        if field in document:
            parsed[field], family = _parse_datetime(
                document[field], nullable=True, family=family
            )
    parsed_scenarios: list[dict[str, object]] = []
    for scenario_value in document["scenarios"]:  # type: ignore[union-attr]
        scenario = _shape_object(scenario_value)
        parsed_scenario = dict(scenario)
        parsed_rules: list[dict[str, object]] = []
        for rule_value in scenario["rules"]:  # type: ignore[union-attr]
            rule = _shape_object(rule_value)
            parsed_rule = dict(rule)
            parsed_rule["cooldown"] = _duration(
                rule.get("cooldown", 0), nullable=False, positive=False, name="cooldown"
            )
            for field in ("effective_from", "expires_at"):
                if field in rule:
                    parsed_rule[field], family = _parse_datetime(
                        rule[field], nullable=True, family=family
                    )
            parsed_rules.append(parsed_rule)
        parsed_scenario["rules"] = parsed_rules
        parsed_scenarios.append(parsed_scenario)
    parsed["scenarios"] = parsed_scenarios
    time_zone = document["time_zone"]
    if family and time_zone != "UTC":
        _fail("json_scalar")
    if not family and time_zone is not None:
        _fail("json_scalar")
    return parsed


def _mapping_condition(
    value: object, *, warning: bool
) -> StrategyCondition | MonitoringCondition:
    condition = _shape_object(value)
    get = condition.get if warning else condition.__getitem__
    operator = get("operator")
    if operator is not None and operator not in _OPERATORS:
        _fail("json_unknown_operator")
    kind = get("kind")
    left_kind = get("left_kind")
    left = get("left")
    right_kind = get("right_kind")
    right = get("right")
    children_value = condition.get("children", []) if warning else condition["children"]
    children = tuple(
        _mapping_condition(child, warning=warning) for child in children_value
    )  # type: ignore[union-attr]
    if kind not in _CONDITION_KINDS:
        _fail("policy_mapping" if not warning else "warning_mapping")
    allowed_left = _WARNING_LEFT_KINDS if warning else _POLICY_LEFT_KINDS
    if kind == "atomic":
        if operator is None or left_kind not in allowed_left:
            _fail("policy_mapping" if not warning else "warning_mapping")
        if left_kind == "column" and type(left) is not str:
            _fail("policy_mapping" if not warning else "warning_mapping")
        if left_kind != "column" and left is not None:
            _fail("policy_mapping" if not warning else "warning_mapping")
        if operator in {"is_missing", "is_not_missing"}:
            if right_kind is not None or right is not None:
                _fail("policy_mapping" if not warning else "warning_mapping")
        else:
            if right_kind not in _RIGHT_KINDS:
                _fail("policy_mapping" if not warning else "warning_mapping")
            if right_kind == "column" and type(right) is not str:
                _fail("policy_mapping" if not warning else "warning_mapping")
            if right_kind == "literal":
                if operator in {"in", "not_in", "between"}:
                    if type(right) is not list:
                        _fail("policy_mapping" if not warning else "warning_mapping")
                    if not right or (operator == "between" and len(right) != 2):
                        _fail("policy_mapping" if not warning else "warning_mapping")
                    right = tuple(right)
                elif right is None or _is_container(right):
                    _fail("policy_mapping" if not warning else "warning_mapping")
    else:
        if (
            operator is not None
            or left_kind is not None
            or left is not None
            or right_kind is not None
            or right is not None
            or (kind == "not" and len(children) != 1)
            or (kind in {"and", "or"} and len(children) < 2)
            or (kind not in {"and", "or", "not"} and children)
        ):
            _fail("policy_mapping" if not warning else "warning_mapping")
    if warning:
        window = condition.get("window")
        if window is not None and window not in _CONDITION_WINDOWS:
            _fail("warning_mapping")
        return MonitoringCondition(
            kind=kind,
            operator=operator,
            left_kind=left_kind,
            left=left,
            right_kind=right_kind,
            right=right,
            window=window,
            children=children,  # type: ignore[arg-type]
        )
    return StrategyCondition(
        kind=kind,
        operator=operator,
        left_kind=left_kind,
        left=left,
        right_kind=right_kind,
        right=right,
        children=children,  # type: ignore[arg-type]
    )


def _mapped_number_or_string(value: object, *, error_key: str) -> float | str | None:
    if value is None or type(value) is str:
        return value
    try:
        return float(value)
    except (OverflowError, ValueError):
        _fail(error_key)
    raise AssertionError("unreachable")


def _owner_float(value: object, *, error_key: str) -> float:
    try:
        return float(value)
    except (OverflowError, ValueError):
        _fail(error_key)
    raise AssertionError("unreachable")


def _map_policy(document: dict[str, object]) -> V02PreLoanRequest:
    parsed_temporal = _temporal_policy(document)
    if (
        document.get("ranking_score_direction") is not None
        and document["ranking_score_direction"] not in _RANKING_DIRECTIONS
    ):
        _fail("policy_mapping")
    for item in document["action_role_mapping"]:  # type: ignore[union-attr]
        if item[1] not in _POLICY_ROLES:
            _fail("policy_mapping")
    rules: list[DecisionRule] = []
    for rule_value, parsed_rule_value in zip(
        document["rules"], parsed_temporal["rules"]
    ):  # type: ignore[union-attr]
        rule = _shape_object(rule_value)
        parsed_rule = _shape_object(parsed_rule_value)
        if rule["phase"] not in _POLICY_PHASES:
            _fail("policy_mapping")
        rules.append(
            DecisionRule(
                rule_key=rule["rule_key"],
                phase=rule["phase"],
                priority=rule["priority"],
                condition=_mapping_condition(rule["condition"], warning=False),  # type: ignore[arg-type]
                action_name=rule["action_name"],
                stop_on_hit=rule["stop_on_hit"],
                enabled=rule["enabled"],
                effective_from=parsed_rule["effective_from"],
                expires_at=parsed_rule["expires_at"],
                description_key=rule["description_key"],
            )
        )
    constraints: list[DecisionConstraint] = []
    for value in document.get("constraints", []):  # type: ignore[union-attr]
        constraint = _shape_object(value)
        if constraint["metric"] not in _CONSTRAINT_METRICS:
            _fail("policy_mapping")
        if constraint["operator"] not in {"le", "ge"}:
            _fail("policy_mapping")
        if (
            constraint.get("action_role") is not None
            and constraint["action_role"] not in _POLICY_ROLES
        ):
            _fail("policy_mapping")
        constraints.append(
            DecisionConstraint(
                constraint_key=constraint["constraint_key"],
                metric=constraint["metric"],
                operator=constraint["operator"],
                threshold=_owner_float(
                    constraint["threshold"], error_key="policy_mapping"
                ),
                action_name=constraint.get("action_name"),
                action_role=constraint.get("action_role"),
                minimum_support=constraint.get("minimum_support", 1),
            )
        )
    try:
        config = DecisionStrategyConfig(
            strategy_key=document["strategy_key"],
            strategy_version=document["strategy_version"],
            effective_from=parsed_temporal["effective_from"],
            expires_at=parsed_temporal["expires_at"],
            evaluation_time=parsed_temporal["evaluation_time"],
            rules=tuple(rules),
            default_action_name=document["default_action_name"],
            unknown_action_name=document["unknown_action_name"],
            action_role_mapping=tuple(
                (item[0], item[1])
                for item in document["action_role_mapping"]  # type: ignore[union-attr]
            ),
            constraints=tuple(constraints),
            ranking_score_column=document.get("ranking_score_column"),
            ranking_score_direction=document.get("ranking_score_direction"),
            historical_action_column=document.get("historical_action_column"),
            historical_action_mapping=tuple(
                (item[0], item[1])
                for item in document.get("historical_action_mapping", [])  # type: ignore[union-attr]
            ),
            historical_policy_version=document.get("historical_policy_version"),
            exposure_column=document.get("exposure_column"),
            loss_fraction=_mapped_number_or_string(
                document.get("loss_fraction"), error_key="policy_mapping"
            ),
            action_assumptions=tuple(
                (
                    item[0],
                    _owner_float(item[1], error_key="policy_mapping"),
                    _owner_float(item[2], error_key="policy_mapping"),
                )
                for item in document.get("action_assumptions", [])  # type: ignore[union-attr]
            ),
            exposure_unit=document.get("exposure_unit"),
            segment_columns=tuple(document.get("segment_columns", [])),  # type: ignore[arg-type]
            time_slice_column=document.get("time_slice_column"),
        )
    except (TypeError, ValueError, OverflowError):
        _fail("policy_mapping")
    return V02PreLoanRequest(config=config)


def _map_warning(document: dict[str, object]) -> V02PostLoanRequest:
    parsed = _temporal_warning(document)
    if document["period_unit"] not in _PERIOD_UNITS:
        _fail("warning_mapping")
    if document.get("time_frequency", "month") not in _PERIOD_UNITS or (
        document.get("ranking_score_direction") is not None
        and document["ranking_score_direction"] not in _RANKING_DIRECTIONS
    ):
        _fail("warning_mapping")
    scenarios: list[WarningScenario] = []
    for scenario_value, parsed_scenario_value in zip(
        document["scenarios"], parsed["scenarios"]
    ):  # type: ignore[union-attr]
        scenario = _shape_object(scenario_value)
        parsed_scenario = _shape_object(parsed_scenario_value)
        if scenario["scenario_kind"] not in _SCENARIO_KINDS:
            _fail("warning_mapping")
        rules: list[EarlyWarningRule] = []
        for rule_value, parsed_rule_value in zip(
            scenario["rules"], parsed_scenario["rules"]
        ):  # type: ignore[union-attr]
            rule = _shape_object(rule_value)
            parsed_rule = _shape_object(parsed_rule_value)
            rules.append(
                EarlyWarningRule(
                    rule_key=rule["rule_key"],
                    priority=rule["priority"],
                    alert_level=rule["alert_level"],
                    condition=_mapping_condition(rule["condition"], warning=True),  # type: ignore[arg-type]
                    persistence_observations=rule.get("persistence_observations", 1),
                    resolution_observations=rule.get("resolution_observations", 1),
                    cooldown=parsed_rule["cooldown"],
                    enabled=rule.get("enabled", True),
                    effective_from=parsed_rule.get("effective_from"),
                    expires_at=parsed_rule.get("expires_at"),
                    description_key=rule.get("description_key"),
                )
            )
        scenarios.append(
            WarningScenario(
                scenario_key=scenario["scenario_key"],
                scenario_kind=scenario["scenario_kind"],
                rules=tuple(rules),
            )
        )
    states: list[LifecycleState] = []
    for value in document["states"]:  # type: ignore[union-attr]
        state = _shape_object(value)
        states.append(
            LifecycleState(
                state_key=state["state_key"],
                state_rank=state["state_rank"],
                priority=state["priority"],
                condition=_mapping_condition(state["condition"], warning=True),  # type: ignore[arg-type]
                terminal=state.get("terminal", False),
                enabled=state.get("enabled", True),
                description_key=state.get("description_key"),
            )
        )
    try:
        config = LifecycleMonitoringConfig(
            monitoring_key=document["monitoring_key"],
            monitoring_version=document["monitoring_version"],
            analysis_as_of=parsed["analysis_as_of"],
            entity_column=document["entity_column"],
            observation_time_column=document["observation_time_column"],
            available_time_column=document["available_time_column"],
            condition_feature_columns=tuple(document["condition_feature_columns"]),  # type: ignore[arg-type]
            event_time_column=document["event_time_column"],
            positive_event_key=document["positive_event_key"],
            prediction_horizon=parsed["prediction_horizon"],
            horizon_end_inclusive=document["horizon_end_inclusive"],
            recent_window=parsed["recent_window"],
            history_window=parsed["history_window"],
            history_start_inclusive=document["history_start_inclusive"],
            expected_observation_interval=parsed["expected_observation_interval"],
            period_unit=document["period_unit"],
            time_zone=document["time_zone"],
            scenarios=tuple(scenarios),
            reference_scenario_key=document["reference_scenario_key"],
            alert_level_ranks=tuple(
                (item[0], item[1])
                for item in document["alert_level_ranks"]  # type: ignore[union-attr]
            ),
            states=tuple(states),
            default_state_key=document["default_state_key"],
            unknown_state_key=document["unknown_state_key"],
            allowed_transitions=tuple(
                (item[0], item[1])
                for item in document.get("allowed_transitions", [])  # type: ignore[union-attr]
            ),
            adverse_state_keys=tuple(document.get("adverse_state_keys", [])),  # type: ignore[arg-type]
            cure_state_keys=tuple(document.get("cure_state_keys", [])),  # type: ignore[arg-type]
            cohort_time_column=document.get("cohort_time_column"),
            cohort_column=document.get("cohort_column"),
            peer_group_columns=tuple(document.get("peer_group_columns", [])),  # type: ignore[arg-type]
            peer_reference_start=parsed.get("peer_reference_start"),
            peer_reference_end=parsed.get("peer_reference_end"),
            ranking_score_column=document.get("ranking_score_column"),
            ranking_score_direction=document.get("ranking_score_direction"),
            exposure_column=document.get("exposure_column"),
            loss_fraction=_mapped_number_or_string(
                document.get("loss_fraction"), error_key="warning_mapping"
            ),
            observed_loss_column=document.get("observed_loss_column"),
            observed_loss_available_time_column=document.get(
                "observed_loss_available_time_column"
            ),
            observed_loss_is_mature_snapshot=document.get(
                "observed_loss_is_mature_snapshot", False
            ),
            segment_columns=tuple(document.get("segment_columns", [])),  # type: ignore[arg-type]
            time_frequency=document.get("time_frequency", "month"),
        )
    except (TypeError, ValueError, OverflowError):
        _fail("warning_mapping")
    return V02PostLoanRequest(config=config)


def _prepare(source: _JsonSource, *, warning: bool) -> dict[str, object]:
    document = _parse(source)
    if type(document) is not dict:
        _fail("json_not_object")
    expected_version = _WARNING_SCHEMA_VERSION if warning else _POLICY_SCHEMA_VERSION
    if document.get("schema_version") != expected_version:
        _fail("json_schema_version")
    if warning:
        _check_warning_fields(document)
        _shape_warning(document)
        _validate_warning_scalars(document)
        _check_budget(document, warning=True)
    else:
        _check_policy_fields(document)
        _shape_policy(document)
        _validate_policy_scalars(document)
        _check_budget(document, warning=False)
    return document


def load_policy_json(source: _JsonSource) -> V02PreLoanRequest:
    """Load and validate one closed policy JSON carrier.

    Parameters
    ----------
    source:
        UTF-8 JSON text, UTF-8 bytes, or an exact :class:`pathlib.Path` whose
        bytes are read literally.  Strings are document text; paths should be
        supplied as ``Path`` objects to keep acquisition unambiguous.

    Returns
    -------
    V02PreLoanRequest
        A typed Task 20 request containing a ``DecisionStrategyConfig``.

    Raises
    ------
    ValueError
        With the stable ``sharper task20: ...`` JSON error prefix for content
        validation.  File read failures remain ``OSError``.
    """

    return _map_policy(_prepare(source, warning=False))


def load_warning_json(source: _JsonSource) -> V02PostLoanRequest:
    """Load and validate one closed warning/lifecycle JSON carrier."""

    return _map_warning(_prepare(source, warning=True))


def load_v02_json(
    policy_source: _JsonSource | None = None,
    warning_source: _JsonSource | None = None,
) -> tuple[V02PreLoanRequest | None, V02PostLoanRequest | None]:
    """Load policy before warning, preserving the frozen adapter precedence."""

    if policy_source is None and warning_source is None:
        _fail("json_structure")
    policy = load_policy_json(policy_source) if policy_source is not None else None
    warning = load_warning_json(warning_source) if warning_source is not None else None
    return policy, warning
