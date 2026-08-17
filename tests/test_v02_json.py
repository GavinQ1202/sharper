"""Direct contract tests for the private Task 20 closed JSON adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sharper.v02_json as adapter


def _condition(*, warning: bool = False, operator: str = "ge") -> dict[str, object]:
    right_kind: str | None = "literal"
    right: object = 1
    if operator in {"is_missing", "is_not_missing"}:
        right_kind = None
        right = None
    value: dict[str, object] = {
        "kind": "atomic",
        "operator": operator,
        "left_kind": "column",
        "left": "score",
        "right_kind": right_kind,
        "right": right,
        "children": [],
    }
    if warning:
        value["window"] = None
    return value


def _policy_document() -> dict[str, object]:
    return {
        "schema_version": "task20.policy.v1",
        "strategy_key": "policy-a",
        "strategy_version": "v1",
        "effective_from": "2026-01-01T00:00:00.000000",
        "expires_at": "2026-12-31T23:59:59.000000",
        "evaluation_time": "2026-06-01T12:30:00.123456",
        "rules": [
            {
                "rule_key": "r1",
                "phase": "decision",
                "priority": 1,
                "condition": _condition(),
                "action_name": "review",
                "stop_on_hit": True,
                "enabled": True,
                "effective_from": None,
                "expires_at": None,
                "description_key": "desc.r1",
            }
        ],
        "default_action_name": "select",
        "unknown_action_name": "review",
        "action_role_mapping": [["select", "selected"], ["review", "review"]],
        "constraints": [
            {
                "constraint_key": "c1",
                "metric": "review_rate",
                "operator": "le",
                "threshold": 0.2,
                "action_name": None,
                "action_role": "review",
                "minimum_support": 2,
            }
        ],
        "ranking_score_column": "score",
        "ranking_score_direction": "higher_risk",
        "historical_action_column": "old_action",
        "historical_action_mapping": [[0, "select"], ["R", "review"]],
        "historical_policy_version": "old-v1",
        "exposure_column": "exposure",
        "loss_fraction": 0.4,
        "action_assumptions": [["select", 1.5, 0.1], ["review", 0.5, 0.2]],
        "exposure_unit": "currency",
        "segment_columns": ["segment", "region"],
        "time_slice_column": "month",
    }


def _warning_rule(*, condition: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "rule_key": "arrears",
        "priority": 1,
        "alert_level": "high",
        "condition": condition or _condition(warning=True),
        "persistence_observations": 2,
        "resolution_observations": 3,
        "cooldown": 1_000_000,
        "enabled": True,
        "effective_from": None,
        "expires_at": None,
        "description_key": "desc.arrears",
    }


def _warning_document() -> dict[str, object]:
    return {
        "schema_version": "task20.warning.v1",
        "monitoring_key": "monitoring-a",
        "monitoring_version": "v1",
        "analysis_as_of": "2026-06-01T12:30:00.123456",
        "entity_column": "entity",
        "observation_time_column": "observed_at",
        "available_time_column": "available_at",
        "condition_feature_columns": ["score", "days_past_due"],
        "event_time_column": "event_at",
        "positive_event_key": "bad",
        "prediction_horizon": 86_400_000_000,
        "horizon_end_inclusive": False,
        "recent_window": 7 * 86_400_000_000,
        "history_window": 30 * 86_400_000_000,
        "history_start_inclusive": True,
        "expected_observation_interval": 86_400_000_000,
        "period_unit": "day",
        "time_zone": None,
        "scenarios": [
            {
                "scenario_key": "reference",
                "scenario_kind": "rule_set",
                "rules": [_warning_rule()],
            }
        ],
        "reference_scenario_key": "reference",
        "alert_level_ranks": [["high", 1], ["low", 2]],
        "states": [
            {
                "state_key": "performing",
                "state_rank": 0,
                "priority": 1,
                "condition": _condition(warning=True, operator="is_not_missing"),
                "terminal": False,
                "enabled": True,
                "description_key": "desc.performing",
            }
        ],
        "default_state_key": "performing",
        "unknown_state_key": "performing",
        "allowed_transitions": [["performing", "performing"]],
        "adverse_state_keys": [],
        "cure_state_keys": ["performing"],
        "cohort_time_column": None,
        "cohort_column": None,
        "peer_group_columns": ["region"],
        "peer_reference_start": None,
        "peer_reference_end": None,
        "ranking_score_column": "score",
        "ranking_score_direction": "higher_risk",
        "exposure_column": "exposure",
        "loss_fraction": "loss_fraction",
        "observed_loss_column": "loss",
        "observed_loss_available_time_column": "loss_available_at",
        "observed_loss_is_mature_snapshot": True,
        "segment_columns": ["region"],
        "time_frequency": "month",
    }


def _text(document: dict[str, object]) -> str:
    return json.dumps(document, separators=(",", ":"), allow_nan=False)


def _error(document: dict[str, object], *, warning: bool = False) -> str:
    function = adapter.load_warning_json if warning else adapter.load_policy_json
    with pytest.raises(ValueError) as caught:
        function(_text(document))
    return str(caught.value)


def test_v02_policy_warning_schema_field_mapping() -> None:
    policy = adapter.load_policy_json(_text(_policy_document()))
    assert policy.config.strategy_key == "policy-a"
    assert policy.config.effective_from.microsecond == 0
    assert policy.config.evaluation_time.microsecond == 123456
    assert policy.config.action_role_mapping == (
        ("select", "selected"),
        ("review", "review"),
    )
    assert policy.config.historical_action_mapping == ((0, "select"), ("R", "review"))
    assert policy.config.action_assumptions == (
        ("select", 1.5, 0.1),
        ("review", 0.5, 0.2),
    )
    assert policy.config.segment_columns == ("segment", "region")
    assert policy.config.rules[0].condition.right == 1

    warning = adapter.load_warning_json(_text(_warning_document()))
    assert warning.config.monitoring_key == "monitoring-a"
    assert warning.config.recent_window.days == 7
    assert warning.config.history_window.days == 30
    assert warning.config.scenarios[0].rules[0].cooldown.total_seconds() == 1
    assert warning.config.alert_level_ranks == (("high", 1), ("low", 2))
    assert warning.config.states[0].condition.window is None


def test_v02_json_duplicate_keys_rejected() -> None:
    duplicate_top = '{"schema_version":"task20.policy.v1","schema_version":"bad"}'
    with pytest.raises(ValueError, match=r"^sharper task20: json_duplicate_key$"):
        adapter.load_policy_json(duplicate_top)

    nested = _text(_policy_document()).replace(
        '"kind":"atomic","operator":"ge"',
        '"kind":"atomic","operator":"ge","operator":"lt"',
        1,
    )
    with pytest.raises(ValueError, match=r"^sharper task20: json_duplicate_key$"):
        adapter.load_policy_json(nested)

    array_duplicate = _policy_document()
    array_duplicate["segment_columns"] = ["region", "region"]
    parsed = adapter.load_policy_json(_text(array_duplicate))
    assert parsed.config.segment_columns == ("region", "region")


def test_v02_json_unknown_version_field_operator() -> None:
    bad_version = _policy_document()
    bad_version["schema_version"] = "task20.policy.v2"
    assert _error(bad_version) == "sharper task20: json_schema_version"

    bad_field = _policy_document()
    bad_field["unexpected"] = 1
    assert _error(bad_field) == "sharper task20: json_unknown_field"

    bad_operator = _policy_document()
    bad_operator["rules"][0]["condition"]["operator"] = "regex"  # type: ignore[index]
    assert _error(bad_operator) == "sharper task20: json_unknown_operator"


def test_v02_json_budget_max_and_max_plus_one() -> None:
    raw = _text(_policy_document()).encode()
    at_max = raw + b" " * (5_000_000 - len(raw))
    assert adapter.load_policy_json(at_max).config.strategy_key == "policy-a"
    with pytest.raises(ValueError, match=r"^sharper task20: json_budget$"):
        adapter.load_policy_json(at_max + b" ")

    policy = _policy_document()
    policy["segment_columns"] = [f"c{i}" for i in range(4_096)]
    assert len(adapter.load_policy_json(_text(policy)).config.segment_columns) == 4_096
    policy["segment_columns"].append("overflow")  # type: ignore[union-attr]
    assert _error(policy) == "sharper task20: json_budget"

    policy = _policy_document()
    policy["rules"] = [policy["rules"][0] for _ in range(100)]  # type: ignore[index]
    assert len(adapter.load_policy_json(_text(policy)).config.rules) == 100
    policy["rules"].append(_policy_document()["rules"][0])  # type: ignore[union-attr]
    assert _error(policy) == "sharper task20: json_budget"

    warning = _warning_document()
    warning["scenarios"] = [warning["scenarios"][0] for _ in range(10)]  # type: ignore[index]
    assert len(adapter.load_warning_json(_text(warning)).config.scenarios) == 10
    warning["scenarios"].append(_warning_document()["scenarios"][0])  # type: ignore[union-attr]
    assert _error(warning, warning=True) == "sharper task20: json_budget"


def test_v02_json_depth_condition_and_warning_collection_budgets() -> None:
    def chain(count: int) -> dict[str, object]:
        node: dict[str, object] = _condition()
        for _ in range(count - 1):
            node = {
                "kind": "not",
                "operator": None,
                "left_kind": None,
                "left": None,
                "right_kind": None,
                "right": None,
                "children": [node],
            }
        return node

    policy = _policy_document()
    policy["rules"][0]["condition"] = chain(6)  # type: ignore[index]
    assert (
        adapter.load_policy_json(_text(policy)).config.rules[0].condition.kind == "not"
    )
    policy["rules"][0]["condition"] = chain(7)  # type: ignore[index]
    assert _error(policy) == "sharper task20: json_budget"

    policy = _policy_document()
    root = {
        "kind": "and",
        "operator": None,
        "left_kind": None,
        "left": None,
        "right_kind": None,
        "right": None,
        "children": [_condition() for _ in range(127)],
    }
    policy["rules"][0]["condition"] = root  # type: ignore[index]
    assert (
        len(adapter.load_policy_json(_text(policy)).config.rules[0].condition.children)
        == 127
    )
    root["children"].append(_condition())  # type: ignore[union-attr]
    assert _error(policy) == "sharper task20: json_budget"

    policy = _policy_document()
    policy["constraints"] = [policy["constraints"][0] for _ in range(50)]  # type: ignore[index]
    assert len(adapter.load_policy_json(_text(policy)).config.constraints) == 50
    policy["constraints"].append(_policy_document()["constraints"][0])  # type: ignore[union-attr]
    assert _error(policy) == "sharper task20: json_budget"

    warning = _warning_document()
    warning["scenarios"][0]["rules"] = [  # type: ignore[index]
        _warning_rule() for _ in range(50)
    ]
    assert (
        len(adapter.load_warning_json(_text(warning)).config.scenarios[0].rules) == 50
    )
    warning["scenarios"][0]["rules"].append(_warning_rule())  # type: ignore[index]
    assert _error(warning, warning=True) == "sharper task20: json_budget"

    warning = _warning_document()
    warning["states"] = [warning["states"][0] for _ in range(50)]  # type: ignore[index]
    assert len(adapter.load_warning_json(_text(warning)).config.states) == 50
    warning["states"].append(_warning_document()["states"][0])  # type: ignore[union-attr]
    assert _error(warning, warning=True) == "sharper task20: json_budget"


def test_v02_json_temporal_grammar_and_duration_rules() -> None:
    for value in (
        "2026-01-01 00:00:00.000000",
        "2026-01-01T00:00:00.00000",
        "2026-01-01T00:00:00.0000000",
        "2026-01-01t00:00:00.000000",
        "2026-01-01T00:00:00.000000z",
        "2026-01-01T00:00:00.000000+00:00",
        " 2026-01-01T00:00:00.000000",
        "2026-02-29T00:00:00.000000",
        "2026-01-01T24:00:00.000000",
    ):
        policy = _policy_document()
        policy["effective_from"] = value
        assert _error(policy) == "sharper task20: json_scalar"

    warning = _warning_document()
    warning["recent_window"] = True
    assert _error(warning, warning=True) == "sharper task20: json_scalar"
    warning = _warning_document()
    warning["recent_window"] = 1.5
    assert _error(warning, warning=True) == "sharper task20: json_scalar"
    warning = _warning_document()
    warning["recent_window"] = "PT1S"
    assert _error(warning, warning=True) == "sharper task20: json_scalar"
    warning = _warning_document()
    warning["history_window"] = warning["recent_window"] - 1
    assert _error(warning, warning=True) == "sharper task20: json_scalar"

    aware = _warning_document()
    aware["analysis_as_of"] = "2026-06-01T12:30:00.123456Z"
    aware["time_zone"] = "UTC"
    loaded = adapter.load_warning_json(_text(aware))
    assert loaded.config.analysis_as_of.tzinfo is not None
    aware["time_zone"] = None
    assert _error(aware, warning=True) == "sharper task20: json_scalar"


def test_v02_json_structure_scalar_and_parser_containment() -> None:
    policy = _policy_document()
    policy["rules"] = {}
    assert _error(policy) == "sharper task20: json_structure"
    policy = _policy_document()
    policy["rules"][0]["priority"] = True  # type: ignore[index]
    assert _error(policy) == "sharper task20: json_scalar"

    with pytest.raises(ValueError, match=r"^sharper task20: json_not_object$"):
        adapter.load_policy_json("[]")
    with pytest.raises(ValueError, match=r"^sharper task20: json_decode$"):
        adapter.load_policy_json('{"schema_version":')
    with pytest.raises(ValueError, match=r"^sharper task20: json_encoding$"):
        adapter.load_policy_json(b"\xff")
    with pytest.raises(ValueError, match=r"^sharper task20: json_scalar$"):
        adapter.load_policy_json(
            _text(_policy_document()).replace(
                '"loss_fraction":0.4', '"loss_fraction":NaN'
            )
        )


def test_v02_json_policy_precedes_warning_and_errors_are_private() -> None:
    policy = _policy_document()
    policy["schema_version"] = "wrong"
    warning = _warning_document()
    warning["schema_version"] = "wrong"
    with pytest.raises(ValueError, match=r"^sharper task20: json_schema_version$"):
        adapter.load_v02_json(_text(policy), _text(warning))

    message = _error({**_policy_document(), "credential": "secret"})
    assert "secret" not in message
    assert "credential" not in message


def test_v02_json_condition_membership_arrays_map_to_tuples() -> None:
    policy = _policy_document()
    condition = policy["rules"][0]["condition"]  # type: ignore[index]
    condition["operator"] = "between"  # type: ignore[index]
    condition["right"] = [0, 1]  # type: ignore[index]
    loaded = adapter.load_policy_json(_text(policy))
    assert loaded.config.rules[0].condition.right == (0, 1)


def test_v02_json_no_owner_execution_and_literal_file_boundary(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(_text(_policy_document()), encoding="utf-8")
    loaded = adapter.load_policy_json(policy_path)
    assert loaded.config.strategy_key == "policy-a"
    with pytest.raises(OSError):
        adapter.load_policy_json(tmp_path / "missing.json")

    assert not hasattr(adapter, "simulate_decision_strategy")
    assert not hasattr(adapter, "monitor_lifecycle")
    assert not hasattr(adapter, "audit_data_quality")
    assert not hasattr(adapter, "evaluate_governance")
    assert not hasattr(adapter, "_condition_kernel")


def test_v02_json_defaulted_warning_nested_fields_map_without_execution() -> None:
    warning = _warning_document()
    rule = warning["scenarios"][0]["rules"][0]  # type: ignore[index]
    rule["condition"].pop("children")  # type: ignore[index]
    for field in (
        "persistence_observations",
        "resolution_observations",
        "enabled",
        "description_key",
    ):
        rule.pop(field)  # type: ignore[union-attr]
    state = warning["states"][0]  # type: ignore[index]
    for field in ("terminal", "enabled", "description_key"):
        state.pop(field)  # type: ignore[union-attr]
    loaded = adapter.load_warning_json(_text(warning))
    assert loaded.config.scenarios[0].rules[0].persistence_observations == 1
    assert loaded.config.scenarios[0].rules[0].cooldown.total_seconds() == 1
    assert loaded.config.states[0].terminal is False
