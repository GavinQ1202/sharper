"""Direct Task 18A contract tests."""

import inspect
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from sharper import lifecycle_monitoring
from sharper.data_audit import audit_data_quality
from sharper.lifecycle_monitoring import (
    EarlyWarningRule,
    LifecycleMonitoringConfig,
    LifecycleState,
    MonitoringCondition,
    WarningScenario,
    _event_resource_gates,
    _fingerprint,
    _state_resource_gates,
    _warning_resource_gates,
    monitor_lifecycle,
)
from sharper.risk_validation import (
    BinaryRiskValidationConfig,
    ExternalRiskPredictions,
    validate_binary_risk,
)


def _condition() -> MonitoringCondition:
    return MonitoringCondition("atomic", "gt", "column", "feature", "literal", 0)


def _config(condition: MonitoringCondition | None = None) -> LifecycleMonitoringConfig:
    checked = condition or _condition()
    return LifecycleMonitoringConfig(
        "monitor",
        "v1",
        datetime(2025, 1, 5),
        "entity",
        "observed",
        "available",
        ("feature", "other"),
        None,
        None,
        None,
        False,
        timedelta(days=1),
        timedelta(days=2),
        False,
        None,
        "day",
        None,
        (
            WarningScenario(
                "reference", "rule_set", (EarlyWarningRule("rule", 0, "high", checked),)
            ),
        ),
        "reference",
        (("high", 1),),
        (
            LifecycleState("current", 0, 0, checked),
            LifecycleState("unknown", 0, 1, checked),
        ),
        "current",
        "unknown",
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity": ["secret-b", "secret-a"],
            "observed": [datetime(2025, 1, 2), datetime(2025, 1, 1)],
            "available": [datetime(2025, 1, 2), datetime(2025, 1, 1)],
            "feature": [2, 1],
            "other": [0, 0],
        },
        index=[9, 9],
    )


def _state_config(
    *candidates: LifecycleState,
    **changes: object,
) -> LifecycleMonitoringConfig:
    """Build a state-focused config with the required inventory states."""
    states = (
        *candidates,
        LifecycleState("default", 0, 98, _condition()),
        LifecycleState("unknown", 0, 99, _condition()),
    )
    return replace(
        _config(),
        states=states,
        default_state_key="default",
        unknown_state_key="unknown",
        **changes,
    )


def _state_frame(
    values: list[object], *, days: list[int] | None = None
) -> pd.DataFrame:
    dates = days or list(range(1, len(values) + 1))
    observed = [datetime(2025, 1, day) for day in dates]
    return pd.DataFrame(
        {
            "entity": ["e"] * len(values),
            "observed": observed,
            "available": observed,
            "feature": values,
            "other": [0] * len(values),
        }
    )


def _event_config(
    condition: MonitoringCondition | None = None,
    **changes: object,
) -> LifecycleMonitoringConfig:
    defaults: dict[str, object] = {
        "analysis_as_of": datetime(2025, 1, 5),
        "event_time_column": "event_time",
        "positive_event_key": "event",
        "prediction_horizon": timedelta(days=2),
    }
    defaults.update(changes)
    return replace(_config(condition), **defaults)


def _event_frame(event_time: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity": ["e", "e", "e"],
            "observed": [datetime(2025, 1, day) for day in (1, 2, 3)],
            "available": [datetime(2025, 1, day) for day in (1, 2, 3)],
            "feature": [1, 0, 0],
            "other": [0, 0, 0],
            "event_time": [pd.NaT, event_time, pd.NaT],
        }
    )


def _event_metric_frame() -> pd.DataFrame:
    days = list(range(1, 10))
    return pd.DataFrame(
        {
            "entity": ["e"] * len(days) + ["late"],
            "observed": [datetime(2025, 1, day) for day in days]
            + [datetime(2025, 1, 9)],
            "available": [datetime(2025, 1, day) for day in days]
            + [datetime(2025, 1, 9)],
            "feature": [1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            "other": [0] * 10,
            "event_time": [
                pd.NaT,
                datetime(2025, 1, 2),
                datetime(2025, 1, 3),
                datetime(2025, 1, 4),
                pd.NaT,
                pd.NaT,
                pd.NaT,
                datetime(2025, 1, 8),
                pd.NaT,
                datetime(2025, 1, 9),
            ],
        }
    )


def _task15_result(frame: pd.DataFrame):
    labels = frame["target"].tolist()
    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    fold_ids: dict[int, int] = {}
    fit_rows: list[tuple[int, tuple[int, ...]]] = []
    for fold_id, (train, validation) in enumerate(
        splitter.split(range(len(frame)), labels)
    ):
        fit_rows.append((fold_id, tuple(sorted(int(position) for position in train))))
        for position in validation:
            fold_ids[int(position)] = fold_id
    positions = tuple(range(len(frame)))
    return validate_binary_risk(
        frame,
        "target",
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_kfold", n_splits=2
        ),
        external_predictions=ExternalRiskPredictions(
            row_positions=positions,
            fold_ids=tuple(fold_ids[position] for position in positions),
            fold_fit_row_positions=tuple(fit_rows),
            ranking_scores=tuple(0.1 + position / 10 for position in positions),
            ranking_direction="higher_risk",
            event_probabilities=tuple(0.2 + position / 10 for position in positions),
            probability_positive_label=1,
            probability_provenance="external_declared",
        ),
    )


def test_foundation_returns_all_typed_tables_and_physical_identity() -> None:
    result = monitor_lifecycle(_frame(), _config())
    tables = (
        result.observation_history,
        result.rule_evaluations,
        result.notifications,
        result.alert_episodes,
        result.event_matches,
        result.state_history,
        result.state_transitions,
        result.monitoring_summary,
        result.scenario_comparison,
        result.lifecycle_summary,
        result.provenance,
    )
    assert len(tables) == 11
    assert result.observation_history["row_position"].tolist() == [0, 1]
    assert result.observation_history["entity_position"].tolist() == [0, 1]
    assert str(result.observation_history["observation_time"].dtype) == "datetime64[ns]"
    assert all(
        table is not other for table in tables for other in tables if table is not other
    )
    assert result.rule_evaluations.shape[0] == 2
    assert result.notifications.shape[0] == 2
    assert result.alert_episodes.shape[0] == 2
    assert "secret-a" not in result.provenance.to_string()
    assert "secret-b" not in result.provenance.to_string()


def test_empty_table_instances_are_not_shared_between_runs() -> None:
    first = monitor_lifecycle(_frame(), _config())
    second = monitor_lifecycle(_frame(), _config())
    assert first.notifications is not second.notifications
    assert first.notifications.dtypes.equals(second.notifications.dtypes)
    assert first.monitoring_fingerprint == second.monitoring_fingerprint


def test_ir03_date_literal_fingerprint_is_canonical_and_stable() -> None:
    condition = MonitoringCondition(
        "atomic", "eq", "column", "feature", "literal", date(2025, 1, 1)
    )
    config = _config(condition)
    assert _fingerprint(config) == _fingerprint(config)
    changed = replace(config, analysis_as_of=datetime(2025, 1, 6))
    assert _fingerprint(config) != _fingerprint(changed)


def test_ir03_canonical_encoder_rejects_unsupported_object_without_callbacks() -> None:
    class Bad:
        callbacks = 0

        def _bad(self, *_: object) -> None:
            type(self).callbacks += 1
            raise AssertionError("callback")

        __repr__ = _bad
        __str__ = _bad
        __hash__ = _bad
        __iter__ = _bad
        __float__ = _bad
        __int__ = _bad

    bad = Bad()
    with pytest.raises(TypeError, match="unsupported canonical value"):
        lifecycle_monitoring._canonical_json(bad)
    assert bad.callbacks == 0


def test_ir03_provenance_has_exact_contract_inventory_and_is_private() -> None:
    frame = _frame().copy()
    frame["segment_secret"] = ["SECRET_SEGMENT_A", "SECRET_SEGMENT_B"]
    frame["cohort_secret"] = ["SECRET_COHORT_A", "SECRET_COHORT_B"]
    result = monitor_lifecycle(
        frame,
        replace(
            _config(),
            segment_columns=("segment_secret",),
            cohort_column="cohort_secret",
        ),
    )
    expected = {
        "contract_version",
        "monitoring_fingerprint",
        "row_identity",
        "entity_identity",
        "time_model",
        "analysis_as_of",
        "history_windows",
        "horizon_policy",
        "scenario_inventory",
        "rule_inventory",
        "alert_level_inventory",
        "state_inventory",
        "transition_policy",
        "event_source",
        "score_source",
        "probability_source",
        "task15_evidence_status",
        "task15_evidence_fingerprint",
        "task16_evidence_status",
        "task16_config_fingerprint",
        "task16_snapshot_identity",
        "exposure_source",
        "observed_loss_source",
        "scope_inventory",
        "resource_usage",
    }
    assert set(result.provenance["provenance_key"]) == expected
    text = result.provenance.to_string()
    assert "SECRET_" not in text
    assert result.provenance.loc[
        result.provenance["provenance_key"] == "task16_snapshot_identity",
        "provenance_value",
    ].iat[0] == "unverified"


def test_ir06_all_emitted_finding_keys_use_closed_families() -> None:
    result = monitor_lifecycle(_event_frame(datetime(2025, 1, 4)), _event_config())
    families = ("monitoring:", "scenario:", "rule:", "entity:", "state:", "transition:")
    keys = []
    for table in (
        result.observation_history, result.rule_evaluations, result.notifications,
        result.alert_episodes, result.event_matches, result.state_history,
        result.state_transitions, result.monitoring_summary, result.scenario_comparison,
        result.lifecycle_summary, result.provenance,
    ):
        if "finding_key" in table:
            keys.extend(str(value) for value in table["finding_key"].dropna())
    assert keys
    assert all(key.startswith(families) for key in keys)
    assert not any(key.startswith("event:") for key in keys)
    assert all("secret" not in key.lower() for key in keys)


@pytest.mark.parametrize(
    ("column", "key"),
    [
        ("observed", "observation_after_as_of"),
        ("available", "available_after_as_of"),
    ],
)
def test_as_of_validation(column: str, key: str) -> None:
    frame = _frame()
    frame.iat[0, frame.columns.get_loc(column)] = datetime(2025, 1, 6)
    with pytest.raises(ValueError, match=key):
        monitor_lifecycle(frame, _config())


def test_available_must_not_follow_observation() -> None:
    frame = _frame()
    frame.iat[0, frame.columns.get_loc("available")] = datetime(2025, 1, 3)
    with pytest.raises(ValueError, match="available_after_observation"):
        monitor_lifecycle(frame, _config())


def test_aware_times_are_normalized_to_utc() -> None:
    zone = timezone(timedelta(hours=8))
    frame = _frame()
    frame["observed"] = [
        datetime(2025, 1, 2, tzinfo=zone),
        datetime(2025, 1, 1, tzinfo=zone),
    ]
    frame["available"] = frame["observed"]
    config = replace(_config(), analysis_as_of=datetime(2025, 1, 5, tzinfo=zone))
    result = monitor_lifecycle(frame, config)
    assert (
        str(result.observation_history["observation_time"].dtype)
        == "datetime64[ns, UTC]"
    )


@pytest.mark.parametrize("right", ["entity", "observed", "available"])
def test_left_and_right_roles_are_isolated_before_kernel(right: str) -> None:
    condition = MonitoringCondition(
        "atomic", "gt", "column", "feature", "column", right
    )
    with pytest.raises(
        ValueError, match="lifecycle condition: forbidden_condition_role"
    ):
        monitor_lifecycle(_frame(), _config(condition))


def test_reverse_forbidden_role_and_nested_short_circuit_are_rejected() -> None:
    illegal = MonitoringCondition(
        "atomic", "eq", "column", "observed", "column", "feature"
    )
    nested = MonitoringCondition("or", children=(_condition(), illegal))
    with pytest.raises(ValueError, match="forbidden_condition_role"):
        monitor_lifecycle(_frame(), _config(nested))


def test_unary_rhs_is_not_silently_ignored() -> None:
    condition = MonitoringCondition(
        "atomic", "is_missing", "column", "feature", "column", "observed"
    )
    with pytest.raises(ValueError, match="forbidden_condition_role"):
        monitor_lifecycle(_frame(), _config(condition))


def test_kernel_delegation_preserves_boolean_unknown_truth() -> None:
    condition = MonitoringCondition(
        "and",
        children=(
            MonitoringCondition("atomic", "gt", "column", "feature", "literal", 0),
            MonitoringCondition("atomic", "is_not_missing", "column", "other"),
        ),
    )
    assert monitor_lifecycle(_frame(), _config(condition)).input_n_rows == 2


def test_duplicate_entity_observation_time_is_rejected_despite_index() -> None:
    frame = _frame()
    frame.iat[1, frame.columns.get_loc("entity")] = frame.iat[
        0, frame.columns.get_loc("entity")
    ]
    frame.iat[1, frame.columns.get_loc("observed")] = frame.iat[
        0, frame.columns.get_loc("observed")
    ]
    with pytest.raises(ValueError, match="duplicate_entity_observation_time"):
        monitor_lifecycle(frame, _config())


class _Malicious:
    callbacks = 0

    def _called(self, *_: object) -> None:
        type(self).callbacks += 1
        raise AssertionError("callback")

    __array__ = _called
    __float__ = _called
    __eq__ = _called
    __lt__ = _called
    __iter__ = _called
    __str__ = _called
    __repr__ = _called
    __hash__ = _called


def test_malicious_entity_is_rejected_without_protocol_callback() -> None:
    _Malicious.callbacks = 0
    frame = _frame()
    frame["entity"] = pd.Series(
        [_Malicious(), "secret-a"], index=frame.index, dtype="object"
    )
    with pytest.raises(ValueError, match="missing_entity"):
        monitor_lifecycle(frame, _config())
    assert _Malicious.callbacks == 0


def test_resource_gate_is_pre_materialization() -> None:
    scenarios = tuple(WarningScenario(f"s{i}", "no_alert", ()) for i in range(11))
    config = replace(_config(), scenarios=scenarios)
    with pytest.raises(
        ValueError, match="lifecycle resource limit exceeded: scenarios"
    ):
        monitor_lifecycle(_frame(), config)


@pytest.mark.parametrize(
    ("signal", "operator", "threshold", "expected"),
    [
        ("prior_value", "eq", 1.0, ["unknown", "true", "false"]),
        ("change", "eq", 2.0, ["unknown", "true", "true"]),
        ("history_mean", "eq", 1.0, ["unknown", "true", "false"]),
        ("trend", "gt", 0.0, ["unknown", "unknown", "true"]),
    ],
)
def test_prior_only_derived_signals(signal, operator, threshold, expected) -> None:
    frame = pd.DataFrame(
        {
            "entity": ["e", "e", "e"],
            "observed": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 2),
                datetime(2025, 1, 3),
            ],
            "available": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 2),
                datetime(2025, 1, 3),
            ],
            "feature": [1.0, 3.0, 5.0],
            "other": [0, 0, 0],
        }
    )
    condition = MonitoringCondition(
        "atomic", operator, signal, "feature", "literal", threshold, "history"
    )
    config = replace(_config(condition), history_window=timedelta(days=3))
    result = monitor_lifecycle(frame, config)
    assert result.rule_evaluations["truth"].tolist() == expected


def _history_signal_oracle(
    frame: pd.DataFrame,
    config: LifecycleMonitoringConfig,
    source: str,
    signal: str,
    entities: list[int],
) -> list[object]:
    """Independent contract oracle; intentionally scans eligible prior rows."""
    output: list[object] = [pd.NA] * len(frame)
    order = sorted(
        range(len(frame)),
        key=lambda position: (
            entities[position],
            frame[config.observation_time_column].iat[position],
            position,
        ),
    )
    history: dict[int, list[int]] = {}
    window = config.history_window
    for position in order:
        current_time = frame[config.observation_time_column].iat[position]
        start = current_time - window
        candidates: list[tuple[datetime, float]] = []
        for prior in history.get(entities[position], []):
            prior_time = frame[config.observation_time_column].iat[prior]
            boundary = (
                prior_time >= start
                if config.history_start_inclusive
                else prior_time > start
            )
            available = frame[config.available_time_column].iat[prior]
            value = frame[source].iat[prior]
            if boundary and prior_time < current_time and available <= current_time:
                if lifecycle_monitoring._finite_real(value):
                    candidates.append((prior_time, float(value)))
        current = frame[source].iat[position]
        if lifecycle_monitoring._finite_real(current) and candidates:
            if signal == "prior_value":
                output[position] = candidates[-1][1]
            elif signal == "change":
                output[position] = float(current) - candidates[-1][1]
            elif signal == "history_mean":
                output[position] = sum(
                    value for _, value in candidates
                ) / len(candidates)
            elif signal == "trend" and len(candidates) >= 2:
                x0 = candidates[0][0]
                xs = [(time - x0).total_seconds() for time, _ in candidates]
                ys = [value for _, value in candidates]
                x_mean = sum(xs) / len(xs)
                y_mean = sum(ys) / len(ys)
                denominator = sum((x - x_mean) ** 2 for x in xs)
                if denominator:
                    output[position] = sum(
                        (x - x_mean) * (y - y_mean)
                        for x, y in zip(xs, ys, strict=True)
                    ) / denominator
        history.setdefault(entities[position], []).append(position)
    return output


def _peer_signal_oracle(
    frame: pd.DataFrame,
    config: LifecycleMonitoringConfig,
    source: str,
) -> list[object]:
    """Independent peer baseline oracle with direct candidate scanning."""
    output: list[object] = [pd.NA] * len(frame)
    end = config.peer_reference_end
    start = config.peer_reference_start
    groups = config.peer_group_columns
    for position in range(len(frame)):
        current_time = frame[config.observation_time_column].iat[position]
        current = frame[source].iat[position]
        group = tuple(frame[column].iat[position] for column in groups)
        if (
            end is None
            or end >= current_time
            or not lifecycle_monitoring._finite_real(current)
        ):
            continue
        if any(lifecycle_monitoring._scope_missing(item) for item in group):
            continue
        values: list[float] = []
        for candidate in range(len(frame)):
            observed = frame[config.observation_time_column].iat[candidate]
            available = frame[config.available_time_column].iat[candidate]
            candidate_group = tuple(frame[column].iat[candidate] for column in groups)
            if observed > end or available > end:
                continue
            if start is not None and observed < start:
                continue
            if any(
                lifecycle_monitoring._scope_missing(item)
                for item in candidate_group
            ):
                continue
            if not all(
                type(left) is type(right) and left == right
                for left, right in zip(group, candidate_group, strict=True)
            ):
                continue
            value = frame[source].iat[candidate]
            if lifecycle_monitoring._finite_real(value):
                values.append(float(value))
        if len(values) >= 2:
            output[position] = float(current) - sum(values) / len(values)
    return output


def _assert_nullable_values(actual: pd.Series, expected: list[object]) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual.tolist(), expected, strict=True):
        if pd.isna(want):
            assert pd.isna(got)
        else:
            assert got == pytest.approx(want)


@pytest.mark.parametrize("signal", ["prior_value", "change", "history_mean", "trend"])
def test_ir08_history_signals_match_independent_oracle(signal: str) -> None:
    frame = pd.DataFrame(
        {
            "entity": ["a", "b", "a", "a", "b", "a"],
            "observed": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 1),
                datetime(2025, 1, 3),
                datetime(2025, 1, 5),
                datetime(2025, 1, 6),
                datetime(2025, 1, 8),
            ],
            "available": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 1),
                datetime(2025, 1, 3),
                datetime(2025, 1, 5),
                datetime(2025, 1, 5),
                datetime(2025, 1, 8),
            ],
            "feature": [1.0, 10.0, 3.0, float("nan"), 14.0, 9.0],
            "other": [0] * 6,
        }
    )
    entities = [0, 1, 0, 0, 1, 0]
    condition = MonitoringCondition(
        "atomic", "eq", signal, "feature", "literal", 0.0, "history"
    )
    optimized = lifecycle_monitoring._derived_signal(
        frame, _config(condition), condition, entities
    )
    expected = _history_signal_oracle(
        frame, _config(condition), "feature", signal, entities
    )
    _assert_nullable_values(optimized, expected)


def test_ir08_history_window_and_future_invariance() -> None:
    frame = _state_frame([1.0, 2.0, 4.0], days=[1, 3, 5])
    condition = MonitoringCondition(
        "atomic", "eq", "history_mean", "feature", "literal", 0.0, "history"
    )
    config = replace(_config(condition), history_window=timedelta(days=2))
    entities = [0, 0, 0]
    before = lifecycle_monitoring._derived_signal(frame, config, condition, entities)
    future = pd.concat(
        [frame, _state_frame([100000.0], days=[20])], ignore_index=True
    )
    after = lifecycle_monitoring._derived_signal(
        future, config, condition, entities + [0]
    )
    _assert_nullable_values(before, after.iloc[:3].tolist())
    assert pd.isna(before.iloc[0])
    assert pd.isna(before.iloc[1])
    assert pd.isna(before.iloc[2])


def test_ir08_peer_deviation_matches_independent_oracle_and_future_invariance() -> None:
    frame = pd.DataFrame(
        {
            "entity": ["a", "b", "c", "d", "e"],
            "observed": [datetime(2025, 1, day) for day in [1, 2, 3, 4, 5]],
            "available": [datetime(2025, 1, day) for day in [1, 2, 3, 4, 5]],
            "feature": [2.0, 4.0, 6.0, 8.0, 999.0],
            "other": [0] * 5,
            "group": ["g", "g", "g", pd.NA, "g"],
        }
    )
    condition = MonitoringCondition(
        "atomic", "eq", "peer_deviation", "feature", "literal", 0.0, "history"
    )
    config = replace(
        _config(condition),
        peer_group_columns=("group",),
        peer_reference_start=datetime(2025, 1, 1),
        peer_reference_end=datetime(2025, 1, 3),
    )
    optimized = lifecycle_monitoring._peer_deviation(frame, config, "feature")
    expected = _peer_signal_oracle(frame, config, "feature")
    _assert_nullable_values(optimized, expected)
    assert pd.isna(optimized.iloc[3])
    assert optimized.iloc[4] == pytest.approx(995.0)
    repeated = lifecycle_monitoring._peer_deviation(frame, config, "feature")
    _assert_nullable_values(repeated, optimized.tolist())


def test_ir08_structure_has_no_nested_full_history_or_frame_scan() -> None:
    for helper in (
        lifecycle_monitoring._derived_signal,
        lifecycle_monitoring._peer_deviation,
    ):
        source = inspect.getsource(helper)
        assert "for prior in" not in source
        assert "for candidate in" not in source
        assert source.count("range(len(data))") <= 2


def test_ir08_history_contract_scale_smoke_is_deterministic() -> None:
    size = 10_000
    observed = [datetime(2020, 1, 1) + timedelta(days=index) for index in range(size)]
    frame = pd.DataFrame(
        {
            "entity": ["e"] * size,
            "observed": observed,
            "available": observed,
            "feature": [float(index % 17) for index in range(size)],
            "other": [0] * size,
        }
    )
    condition = MonitoringCondition(
        "atomic", "eq", "history_mean", "feature", "literal", 0.0, "history"
    )
    config = replace(_config(condition), history_window=timedelta(days=size))
    entities = [0] * size
    first = lifecycle_monitoring._derived_signal(frame, config, condition, entities)
    second = lifecycle_monitoring._derived_signal(frame, config, condition, entities)
    pd.testing.assert_series_equal(first, second)
    assert len(first) == size
    assert pd.isna(first.iloc[0])


def test_peer_deviation_requires_strict_prior_reference_end() -> None:
    frame = pd.DataFrame(
        {
            "entity": ["a", "b", "c"],
            "observed": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 2),
                datetime(2025, 2, 1),
            ],
            "available": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 2),
                datetime(2025, 2, 1),
            ],
            "feature": [2.0, 4.0, 8.0],
            "other": [0, 0, 0],
            "group": ["g", "g", "g"],
        }
    )
    condition = MonitoringCondition(
        "atomic", "eq", "peer_deviation", "feature", "literal", 5.0, "history"
    )
    config = replace(
        _config(condition),
        analysis_as_of=datetime(2025, 2, 5),
        peer_group_columns=("group",),
        peer_reference_end=datetime(2025, 1, 2),
    )
    result = monitor_lifecycle(frame, config)
    assert result.rule_evaluations["truth"].tolist() == ["unknown", "unknown", "true"]
    equal = replace(config, peer_reference_end=datetime(2025, 2, 1))
    assert (
        monitor_lifecycle(frame, equal).rule_evaluations["truth"].tolist()[-1]
        == "unknown"
    )


@pytest.mark.parametrize("missing", [pd.NA, None, float("nan")])
def test_ir04_missing_current_peer_key_is_unknown_without_pandas_exception(
    missing: object,
) -> None:
    frame = pd.DataFrame(
        {
            "entity": ["a", "b", "c"],
            "observed": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 2),
                datetime(2025, 1, 3),
            ],
            "available": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 2),
                datetime(2025, 1, 3),
            ],
            "feature": [2.0, 4.0, 8.0],
            "other": [0, 0, 0],
            "group": ["g", "g", missing],
        }
    )
    condition = MonitoringCondition(
        "atomic", "eq", "peer_deviation", "feature", "literal", 1.0, "history"
    )
    config = replace(
        _config(condition),
        peer_group_columns=("group",),
        peer_reference_end=datetime(2025, 1, 2),
    )
    result = monitor_lifecycle(frame, config)
    assert result.rule_evaluations["truth"].tolist()[-1] == "unknown"


def test_ir04_missing_candidate_peer_key_does_not_pollute_baseline() -> None:
    frame = pd.DataFrame(
        {
            "entity": ["a", "b", "c", "d"],
            "observed": [datetime(2025, 1, day) for day in range(1, 5)],
            "available": [datetime(2025, 1, day) for day in range(1, 5)],
            "feature": [2.0, 4.0, 4.0, 8.0],
            "other": [0] * 4,
            "group": ["g", pd.NA, "g", "g"],
        }
    )
    condition = MonitoringCondition(
        "atomic", "eq", "peer_deviation", "feature", "literal", 5.0, "history"
    )
    config = replace(
        _config(condition),
        peer_group_columns=("group",),
        peer_reference_end=datetime(2025, 1, 3),
    )
    result = monitor_lifecycle(frame, config)
    assert result.rule_evaluations["truth"].tolist()[-1] == "true"


def test_ir04_both_missing_peer_keys_have_unknown_signal() -> None:
    frame = pd.DataFrame(
        {
            "entity": ["a", "b", "c"],
            "observed": [datetime(2025, 1, day) for day in (1, 2, 3)],
            "available": [datetime(2025, 1, day) for day in (1, 2, 3)],
            "feature": [2.0, 4.0, 8.0],
            "other": [0, 0, 0],
            "group": [pd.NA, pd.NA, "g"],
        }
    )
    condition = MonitoringCondition(
        "atomic", "eq", "peer_deviation", "feature", "literal", 1.0, "history"
    )
    config = replace(
        _config(condition),
        peer_group_columns=("group",),
        peer_reference_end=datetime(2025, 1, 2),
    )
    result = monitor_lifecycle(frame, config)
    assert result.rule_evaluations["truth"].tolist()[:2] == ["unknown", "unknown"]


def test_ir04_malicious_peer_group_is_rejected_without_protocol_calls() -> None:
    _Malicious.callbacks = 0
    frame = _frame().assign(
        group=pd.Series([_Malicious(), "g"], index=_frame().index, dtype="object")
    )
    condition = MonitoringCondition(
        "atomic", "eq", "peer_deviation", "feature", "literal", 1.0, "history"
    )
    config = replace(
        _config(condition),
        peer_group_columns=("group",),
        peer_reference_end=datetime(2025, 1, 1),
    )
    with pytest.raises(ValueError, match="unsupported_dtype"):
        monitor_lifecycle(frame, config)
    assert _Malicious.callbacks == 0


def test_all_rules_are_retained_and_primary_is_rank_then_rule_order() -> None:
    low = EarlyWarningRule("low", 0, "low", _condition())
    high = EarlyWarningRule("high", 0, "high", _condition())
    config = replace(
        _config(),
        scenarios=(WarningScenario("reference", "rule_set", (low, high)),),
        alert_level_ranks=(("low", 1), ("high", 2)),
    )
    result = monitor_lifecycle(_frame(), config)
    assert result.rule_evaluations.shape[0] == 4
    assert result.observation_history["primary_rule_key"].tolist() == ["high", "high"]


def test_persistence_false_tail_resolution_and_reopen() -> None:
    frame = pd.DataFrame(
        {
            "entity": ["e"] * 6,
            "observed": [datetime(2025, 1, day) for day in range(1, 7)],
            "available": [datetime(2025, 1, day) for day in range(1, 7)],
            "feature": [1, 1, 0, 0, 1, 1],
            "other": [0] * 6,
        }
    )
    rule = EarlyWarningRule(
        "rule",
        0,
        "high",
        _condition(),
        persistence_observations=2,
        resolution_observations=2,
        cooldown=timedelta(days=2),
    )
    config = replace(
        _config(),
        analysis_as_of=datetime(2025, 1, 7),
        scenarios=(WarningScenario("reference", "rule_set", (rule,)),),
    )
    result = monitor_lifecycle(frame, config)
    assert result.notifications["notification_kind"].tolist() == [
        "episode_open",
        "episode_reopen",
    ]
    assert result.rule_evaluations["episode_status"].tolist()[2:4] == [
        "pending",
        "resolved",
    ]
    assert pd.isna(result.observation_history["primary_rule_key"].iat[2])
    assert result.alert_episodes["episode_ordinal"].tolist() == [0, 1]
    assert result.alert_episodes["episode_end_time"].iloc[0] == datetime(2025, 1, 4)


def _persistence_config(
    rule: EarlyWarningRule, *, analysis_as_of: datetime
) -> LifecycleMonitoringConfig:
    return replace(
        _config(),
        analysis_as_of=analysis_as_of,
        scenarios=(WarningScenario("reference", "rule_set", (rule,)),),
    )


def test_persistence_cooldown_and_open_episode_diagnostics() -> None:
    frame = pd.DataFrame(
        {
            "entity": ["e"] * 5,
            "observed": [datetime(2025, 1, day) for day in range(1, 6)],
            "available": [datetime(2025, 1, day) for day in range(1, 6)],
            "feature": [1] * 5,
            "other": [0] * 5,
        }
    )
    rule = EarlyWarningRule("rule", 0, "high", _condition(), 2, 2, timedelta(days=2))
    result = monitor_lifecycle(
        frame, _persistence_config(rule, analysis_as_of=datetime(2025, 1, 7))
    )
    evaluations = result.rule_evaluations
    assert evaluations["true_streak"].tolist() == [1, 2, 3, 4, 5]
    assert evaluations["notification_status"].tolist() == [
        "not_emitted",
        "emitted",
        "suppressed",
        "emitted",
        "suppressed",
    ]
    assert evaluations["episode_status"].tolist()[1:] == [
        "active",
        "active",
        "active",
        "active",
    ]
    assert result.observation_history["primary_rule_key"].tolist()[1:] == [
        "rule",
        "rule",
        "rule",
        "rule",
    ]
    assert result.notifications["notification_time"].tolist() == [
        datetime(2025, 1, 2),
        datetime(2025, 1, 4),
    ]
    assert result.notifications["notification_kind"].tolist() == [
        "episode_open",
        "repeated",
    ]
    episode = result.alert_episodes.iloc[0]
    assert pd.isna(episode["episode_end_time"])
    assert episode["duration_seconds"] == 5 * 24 * 60 * 60
    assert episode["raw_hit_count"] == 4
    assert episode["notification_count"] == 2
    assert episode["suppressed_notification_count"] == 2
    summary = result.monitoring_summary.loc[
        result.monitoring_summary["scope_key"] == "overall"
    ].set_index("metric")
    assert summary.loc["notification_count", "numerator"] == 2
    assert summary.loc["notification_count", "support_n"] == 2


def test_unknown_breaks_resolution_without_closing_false_tail_episode() -> None:
    frame = pd.DataFrame(
        {
            "entity": ["e"] * 6,
            "observed": [datetime(2025, 1, day) for day in range(1, 7)],
            "available": [datetime(2025, 1, day) for day in range(1, 7)],
            "feature": [1, 1, 0, None, 0, 0],
            "other": [0] * 6,
        }
    )
    rule = EarlyWarningRule("rule", 0, "high", _condition(), 2, 2)
    result = monitor_lifecycle(
        frame, _persistence_config(rule, analysis_as_of=datetime(2025, 1, 7))
    )
    evaluations = result.rule_evaluations
    assert evaluations["truth"].tolist() == [
        "true",
        "true",
        "false",
        "unknown",
        "false",
        "false",
    ]
    assert evaluations["episode_status"].tolist()[2:] == [
        "pending",
        "active",
        "pending",
        "resolved",
    ]
    assert evaluations["false_streak"].tolist()[2:] == [1, 0, 1, 2]
    assert pd.isna(result.observation_history["primary_rule_key"].iat[2])
    assert result.observation_history["emitted_notification_count"].iat[2] == 0
    episode = result.alert_episodes.iloc[0]
    assert episode["episode_end_time"] == datetime(2025, 1, 6)
    assert episode["duration_seconds"] == 4 * 24 * 60 * 60


def test_cadence_gap_breaks_persistence_without_resolving_open_episode() -> None:
    frame = pd.DataFrame(
        {
            "entity": ["e"] * 4,
            "observed": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 3),
                datetime(2025, 1, 4),
                datetime(2025, 1, 5),
            ],
            "available": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 3),
                datetime(2025, 1, 4),
                datetime(2025, 1, 5),
            ],
            "feature": [1] * 4,
            "other": [0] * 4,
        }
    )
    rule = EarlyWarningRule("rule", 0, "high", _condition(), 2, 2)
    config = replace(
        _persistence_config(rule, analysis_as_of=datetime(2025, 1, 6)),
        expected_observation_interval=timedelta(days=1),
    )
    result = monitor_lifecycle(frame, config)
    assert result.observation_history["is_consecutive"].tolist() == [
        pd.NA,
        False,
        True,
        True,
    ]
    assert result.rule_evaluations["true_streak"].tolist() == [1, 0, 1, 2]
    assert result.notifications["notification_time"].tolist() == [datetime(2025, 1, 5)]


def test_warning_state_is_isolated_by_entity_and_scenario() -> None:
    frame = pd.DataFrame(
        {
            "entity": ["a", "a", "b"],
            "observed": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 2),
                datetime(2025, 1, 1),
            ],
            "available": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 2),
                datetime(2025, 1, 1),
            ],
            "feature": [1, 1, 1],
            "other": [0, 0, 0],
        }
    )
    first = EarlyWarningRule("first", 0, "high", _condition(), 2, 2)
    second = EarlyWarningRule("second", 0, "high", _condition(), 2, 2)
    config = replace(
        _config(),
        analysis_as_of=datetime(2025, 1, 3),
        scenarios=(
            WarningScenario("reference", "rule_set", (first,)),
            WarningScenario("challenger", "rule_set", (second,)),
        ),
    )
    result = monitor_lifecycle(frame, config)
    assert result.notifications["entity_position"].tolist() == [0, 0]
    assert set(result.notifications["scenario_key"].tolist()) == {
        "reference",
        "challenger",
    }
    assert result.alert_episodes["entity_position"].tolist() == [0, 0]


def test_warning_resource_projections_fail_before_rule_evaluation() -> None:
    rows = 10_001
    frame = pd.DataFrame(
        {
            "entity": ["a"] * 10_000 + ["b"],
            "observed": [datetime(2025, 1, 1) + timedelta(days=i) for i in range(rows)],
            "available": [
                datetime(2025, 1, 1) + timedelta(days=i) for i in range(rows)
            ],
            "feature": [1] * rows,
            "other": [0] * rows,
        }
    )
    rules = tuple(EarlyWarningRule(f"r{i}", i, "high", _condition()) for i in range(50))
    config = replace(
        _config(),
        analysis_as_of=datetime(2053, 1, 1),
        time_frequency="quarter",
        scenarios=(
            WarningScenario("reference", "rule_set", rules),
            WarningScenario("challenger", "rule_set", rules),
        ),
    )
    with pytest.raises(
        ValueError, match="lifecycle resource limit exceeded: rule_evaluations"
    ):
        monitor_lifecycle(frame, config)


def test_episode_resource_projection_counts_declared_disabled_rules() -> None:
    rows = 10_001
    frame = pd.DataFrame(
        {
            "entity": ["a"] * 10_000 + ["b"],
            "observed": [datetime(2025, 1, 1) + timedelta(days=i) for i in range(rows)],
            "available": [
                datetime(2025, 1, 1) + timedelta(days=i) for i in range(rows)
            ],
            "feature": [1] * rows,
            "other": [0] * rows,
        }
    )
    rules = tuple(
        EarlyWarningRule(f"r{i}", i, "high", _condition(), enabled=False)
        for i in range(50)
    )
    config = replace(
        _config(),
        analysis_as_of=datetime(2053, 1, 1),
        time_frequency="quarter",
        scenarios=(
            WarningScenario("reference", "rule_set", rules),
            WarningScenario("challenger", "rule_set", rules),
        ),
    )
    with pytest.raises(ValueError, match="lifecycle resource limit exceeded: episodes"):
        monitor_lifecycle(frame, config)


def test_warning_resource_gate_exact_maximum_and_maximum_plus_one() -> None:
    rules = tuple(EarlyWarningRule(f"r{i}", i, "high", _condition()) for i in range(50))
    config = replace(
        _config(),
        scenarios=(
            WarningScenario("reference", "rule_set", rules),
            WarningScenario("challenger", "rule_set", rules),
        ),
    )
    _warning_resource_gates(config, [0] * 10_000)
    with pytest.raises(
        ValueError, match="lifecycle resource limit exceeded: rule_evaluations"
    ):
        _warning_resource_gates(config, [0] * 10_001)

    inactive_rules = tuple(replace(rule, enabled=False) for rule in rules)
    inactive = replace(
        config,
        scenarios=(
            WarningScenario("reference", "rule_set", inactive_rules),
            WarningScenario("challenger", "rule_set", inactive_rules),
        ),
    )
    _warning_resource_gates(inactive, [0] * 10_000)
    with pytest.raises(ValueError, match="lifecycle resource limit exceeded: episodes"):
        _warning_resource_gates(inactive, [0] * 10_001)


def test_warning_episode_processing_is_deterministic_and_keeps_input_unchanged() -> (
    None
):
    frame = pd.DataFrame(
        {
            "entity": ["secret"] * 4,
            "observed": [datetime(2025, 1, day) for day in range(1, 5)],
            "available": [datetime(2025, 1, day) for day in range(1, 5)],
            "feature": [1, 1, 1, 0],
            "other": [0] * 4,
        },
        index=[7, 7, 8, 8],
    )
    original = frame.copy(deep=True)
    rule = EarlyWarningRule("rule", 0, "high", _condition(), 2, 2)
    config = _persistence_config(rule, analysis_as_of=datetime(2025, 1, 5))
    first = monitor_lifecycle(frame, config)
    second = monitor_lifecycle(frame, config)
    pd.testing.assert_frame_equal(first.rule_evaluations, second.rule_evaluations)
    pd.testing.assert_frame_equal(first.notifications, second.notifications)
    pd.testing.assert_frame_equal(first.alert_episodes, second.alert_episodes)
    pd.testing.assert_frame_equal(first.monitoring_summary, second.monitoring_summary)
    pd.testing.assert_frame_equal(frame, original)
    assert frame.index.equals(original.index)
    assert "secret" not in first.alert_episodes.to_string()
    assert "secret" not in first.monitoring_summary.to_string()


def test_state_candidate_precedence_and_fallbacks() -> None:
    high = LifecycleState("high", 2, 1, _condition())
    alpha = LifecycleState("alpha", 1, 0, _condition())
    bravo = LifecycleState("bravo", 3, 0, _condition())
    result = monitor_lifecycle(
        _state_frame([1, 0, None]),
        _state_config(high, bravo, alpha, analysis_as_of=datetime(2025, 1, 4)),
    )
    history = result.state_history
    assert history["candidate_state_key"].tolist()[:2] == ["alpha", "default"]
    assert pd.isna(history["candidate_state_key"].iat[2])
    assert pd.isna(history["candidate_state_rank"].iat[2])
    assert pd.isna(history["candidate_state_priority"].iat[2])
    assert history["effective_state_key"].tolist() == ["alpha", "default", "unknown"]
    assert history["matching_state_count"].tolist() == [3, 0, 0]
    assert history["reason"].tolist() == [
        "computed",
        "default_state_applied",
        "unknown_condition",
    ]
    assert (
        result.observation_history["state_reason"].tolist()
        == history["reason"].tolist()
    )
    assert result.observation_history["effective_state_key"].tolist() == [
        "alpha",
        "default",
        "unknown",
    ]
    assert result.observation_history["effective_state_rank"].tolist() == [1, 0, 0]


def test_state_transitions_keep_invalid_candidate_and_last_valid_effective_state() -> (
    None
):
    states = (
        LifecycleState(
            "a",
            1,
            0,
            MonitoringCondition("atomic", "eq", "column", "feature", "literal", 1),
        ),
        LifecycleState(
            "b",
            2,
            0,
            MonitoringCondition("atomic", "eq", "column", "feature", "literal", 2),
        ),
        LifecycleState(
            "c",
            3,
            0,
            MonitoringCondition("atomic", "eq", "column", "feature", "literal", 3),
        ),
    )
    config = _state_config(
        *states,
        analysis_as_of=datetime(2025, 1, 6),
        allowed_transitions=(("a", "b"), ("b", "a")),
        adverse_state_keys=("b",),
        cure_state_keys=("a",),
    )
    transitions = monitor_lifecycle(
        _state_frame([1, 1, 2, 3, 1]), config
    ).state_transitions
    assert transitions["transition_kind"].tolist() == [
        "entry",
        "stay",
        "change",
        "invalid",
        "change",
    ]
    assert transitions["transition_direction"].tolist() == [
        "not_applicable",
        "flat",
        "roll_forward",
        "not_applicable",
        "roll_back",
    ]
    assert transitions["candidate_to_state_key"].tolist()[3] == "c"
    assert transitions["effective_to_state_key"].tolist()[3] == "b"
    assert transitions["effective_to_state_key"].tolist()[4] == "a"
    assert transitions["reason"].tolist()[3] == "transition_not_allowed"
    assert transitions["is_cure"].tolist() == [False, False, False, False, True]
    assert pd.isna(transitions["from_row_position"].iat[0])
    assert pd.isna(transitions["is_consecutive"].iat[0])


def test_lifecycle_summary_hand_calculates_effective_state_and_transition_metrics() -> (
    None
):
    states = (
        LifecycleState(
            "a",
            1,
            0,
            MonitoringCondition("atomic", "eq", "column", "feature", "literal", 1),
        ),
        LifecycleState(
            "b",
            2,
            0,
            MonitoringCondition("atomic", "eq", "column", "feature", "literal", 2),
        ),
        LifecycleState(
            "c",
            3,
            0,
            MonitoringCondition("atomic", "eq", "column", "feature", "literal", 3),
        ),
    )
    config = _state_config(
        *states,
        analysis_as_of=datetime(2025, 1, 6),
        allowed_transitions=(("a", "b"), ("b", "a")),
        adverse_state_keys=("b",),
        cure_state_keys=("a",),
    )
    result = monitor_lifecycle(_state_frame([1, 1, 2, 3, 1]), config)
    lifecycle = result.lifecycle_summary
    overall = lifecycle.loc[lifecycle["scope_key"] == "overall"].set_index("metric")
    expected = {
        "state_observation_count": (5.0, pd.NA, 5, 5.0),
        "state_observation_rate": (5.0, 5.0, 5, 1.0),
        "entity_state_count": (1.0, pd.NA, 1, 1.0),
        "entity_state_rate": (1.0, 1.0, 1, 1.0),
        "transition_count": (3.0, pd.NA, 3, 3.0),
        "transition_rate": (3.0, 3.0, 3, 1.0),
        "roll_forward_count": (1.0, pd.NA, 3, 1.0),
        "roll_forward_rate": (1.0, 3.0, 3, 1 / 3),
        "roll_back_count": (1.0, pd.NA, 3, 1.0),
        "roll_back_rate": (1.0, 3.0, 3, 1 / 3),
        "cure_count": (1.0, pd.NA, 3, 1.0),
        "cure_rate": (1.0, 1.0, 1, 1.0),
        "entry_count": (1.0, pd.NA, 1, 1.0),
        "reentry_count": (0.0, pd.NA, 0, 0.0),
        "time_in_state_mean": (432000.0, 5.0, 5, 86400.0),
    }
    for metric, (numerator, denominator, support, value) in expected.items():
        assert overall.loc[metric, "numerator"] == numerator
        if denominator is pd.NA:
            assert pd.isna(overall.loc[metric, "denominator"])
        else:
            assert overall.loc[metric, "denominator"] == denominator
        assert overall.loc[metric, "support_n"] == support
        assert overall.loc[metric, "metric_value"] == value
        assert overall.loc[metric, "status"] == "available"
        assert overall.loc[metric, "reason"] == "computed"
    state_a = lifecycle.loc[
        (lifecycle["scope_key"] == "state") & (lifecycle["to_state_key"] == "a")
    ].set_index("metric")
    assert state_a.loc["state_observation_count", "numerator"] == 3
    assert state_a.loc["state_observation_rate", "metric_value"] == 0.6
    assert result.state_transitions["transition_kind"].tolist().count("invalid") == 1
    assert overall.loc["transition_count", "numerator"] == 3


def test_terminal_state_allows_only_stay_and_rejects_exit() -> None:
    performing = LifecycleState(
        "performing",
        0,
        0,
        MonitoringCondition("atomic", "eq", "column", "feature", "literal", 1),
    )
    closed = LifecycleState(
        "closed",
        1,
        0,
        MonitoringCondition("atomic", "eq", "column", "feature", "literal", 2),
        terminal=True,
    )
    config = _state_config(
        performing,
        closed,
        analysis_as_of=datetime(2025, 1, 5),
        allowed_transitions=(("performing", "closed"),),
    )
    transitions = monitor_lifecycle(
        _state_frame([1, 2, 2, 1]), config
    ).state_transitions
    assert transitions["transition_kind"].tolist() == [
        "entry",
        "change",
        "stay",
        "invalid",
    ]
    assert transitions["effective_to_state_key"].tolist()[-1] == "closed"
    assert transitions["reason"].tolist()[-1] == "terminal_state_exit"


def test_state_gap_is_reentry_without_direction_or_transition_allowlist_check() -> None:
    lower = LifecycleState(
        "lower",
        0,
        0,
        MonitoringCondition("atomic", "eq", "column", "feature", "literal", 1),
    )
    higher = LifecycleState(
        "higher",
        1,
        0,
        MonitoringCondition("atomic", "eq", "column", "feature", "literal", 2),
    )
    config = _state_config(
        lower,
        higher,
        analysis_as_of=datetime(2025, 1, 4),
        expected_observation_interval=timedelta(days=1),
        allowed_transitions=(("lower", "higher"),),
    )
    transition = monitor_lifecycle(
        _state_frame([1, 2], days=[1, 3]), config
    ).state_transitions.iloc[1]
    assert transition["transition_kind"] == "reentry"
    assert transition["transition_direction"] == "not_applicable"
    assert bool(transition["is_allowed"])
    assert not bool(transition["is_consecutive"])


def test_state_assignment_does_not_change_warning_outputs_or_mutate_prior_rows() -> (
    None
):
    candidate = LifecycleState("high", 1, 0, _condition())
    frame = _state_frame([1, 0, 1, 0])
    config = _state_config(candidate, analysis_as_of=datetime(2025, 1, 5))
    baseline = monitor_lifecycle(
        frame,
        replace(
            config,
            states=_config().states,
            default_state_key="current",
            unknown_state_key="unknown",
        ),
    )
    result = monitor_lifecycle(frame, config)
    pd.testing.assert_frame_equal(result.rule_evaluations, baseline.rule_evaluations)
    pd.testing.assert_frame_equal(result.notifications, baseline.notifications)
    pd.testing.assert_frame_equal(result.alert_episodes, baseline.alert_episodes)
    mutated = frame.copy(deep=True)
    mutated.loc[3, "feature"] = 1
    changed = monitor_lifecycle(mutated, config)
    pd.testing.assert_frame_equal(
        result.state_history.iloc[:3], changed.state_history.iloc[:3]
    )


def test_state_resource_gates_accept_exact_maximum_and_reject_plus_one() -> None:
    states = tuple(
        LifecycleState(f"s{index}", index, index, _condition()) for index in range(50)
    )
    config = replace(_config(), states=states)
    _state_resource_gates(config, [0] * 100_000)
    with pytest.raises(
        ValueError, match="lifecycle resource limit exceeded: state_evaluations"
    ):
        _state_resource_gates(config, [0] * 100_001)
    no_states = replace(_config(), states=())
    _state_resource_gates(no_states, [0] * 100_000)
    with pytest.raises(
        ValueError, match="lifecycle resource limit exceeded: state_history_rows"
    ):
        _state_resource_gates(no_states, [0] * 100_001)


def test_state_transition_sources_and_terminal_declarations_are_rejected() -> None:
    invalid = LifecycleState(
        "invalid",
        1,
        0,
        MonitoringCondition("atomic", "eq", "prior_state", None, "literal", 1),
    )
    with pytest.raises(ValueError, match="lifecycle config is invalid: invalid_state"):
        monitor_lifecycle(_frame(), _state_config(invalid))

    terminal = LifecycleState("terminal", 1, 0, _condition(), terminal=True)
    with pytest.raises(
        ValueError, match="lifecycle config is invalid: invalid_transition"
    ):
        monitor_lifecycle(
            _frame(),
            _state_config(
                terminal,
                analysis_as_of=datetime(2025, 1, 6),
                allowed_transitions=(("terminal", "default"),),
            ),
        )


@pytest.mark.parametrize(
    ("event_time", "closed", "captured"),
    [
        (datetime(2025, 1, 1), False, False),
        (datetime(2025, 1, 2), False, True),
        (datetime(2025, 1, 3), False, False),
        (datetime(2025, 1, 3), True, True),
    ],
)
def test_event_horizon_boundaries_and_capture_ownership(
    event_time: datetime, closed: bool, captured: bool
) -> None:
    result = monitor_lifecycle(
        _event_frame(event_time), _event_config(horizon_end_inclusive=closed)
    )
    match = result.event_matches.iloc[0]
    assert bool(match["captured"]) is captured
    assert match["event_status"] == (
        "not_eligible" if event_time == datetime(2025, 1, 1) else "mature"
    )
    assert result.notifications["matched_event_count"].sum() == int(captured)


def test_event_censoring_and_uncaptured_rows_are_not_negative_observations() -> None:
    config = _event_config(analysis_as_of=datetime(2025, 1, 2))
    censored = monitor_lifecycle(_event_frame(pd.NaT).iloc[:2], config)
    assert censored.event_matches.empty
    assert censored.observation_history["maturity_status"].tolist() == [
        "immature",
        "immature",
    ]
    assert censored.observation_history["event_within_horizon"].isna().sum() == 0
    uncaptured = monitor_lifecycle(
        _event_frame(datetime(2025, 1, 2)),
        _event_config(
            condition=MonitoringCondition(
                "atomic", "gt", "column", "feature", "literal", 9
            )
        ),
    )
    match = uncaptured.event_matches.iloc[0]
    assert not bool(match["captured"])
    assert pd.isna(match["capturing_notification_row_position"])
    assert match["match_status"] == "not_captured"


def test_event_summary_hand_calculates_all_backtest_metrics() -> None:
    frame = _event_metric_frame()
    config = _event_config(
        analysis_as_of=datetime(2025, 1, 10), horizon_end_inclusive=True
    )
    first = monitor_lifecycle(frame, config)
    second = monitor_lifecycle(frame, config)
    pd.testing.assert_frame_equal(first.monitoring_summary, second.monitoring_summary)
    pd.testing.assert_frame_equal(first.event_matches, second.event_matches)
    matches_before = first.event_matches.copy(deep=True)
    notifications_before = first.notifications.copy(deep=True)
    observations_before = first.observation_history.copy(deep=True)
    lifecycle_monitoring._monitoring_summaries(
        config,
        first.rule_evaluations.to_dict("records"),
        first.notifications.to_dict("records"),
        first.alert_episodes.to_dict("records"),
        first.event_matches.to_dict("records"),
        first.observation_history.to_dict("records"),
    )
    pd.testing.assert_frame_equal(first.event_matches, matches_before)
    pd.testing.assert_frame_equal(first.notifications, notifications_before)
    pd.testing.assert_frame_equal(first.observation_history, observations_before)
    summary = first.monitoring_summary.loc[
        first.monitoring_summary["scope_key"] == "overall"
    ].set_index("metric")
    expected = {
        "captured_event_count": (3.0, pd.NA, 4, 3.0),
        "event_recall": (3.0, 4.0, 4, 0.75),
        "notification_precision": (2.0, 3.0, 3, 2 / 3),
        "false_alert_share": (1.0, 3.0, 3, 1 / 3),
        "false_positive_rate": (1.0, 3.0, 3, 1 / 3),
        "lead_time_mean": (432000.0, 3.0, 3, 144000.0),
        "lead_time_median": (172800.0, 3.0, 3, 172800.0),
        "warning_to_event_rate": (2.0, 3.0, 3, 2 / 3),
    }
    for metric, (numerator, denominator, support, value) in expected.items():
        assert summary.loc[metric, "numerator"] == numerator
        if denominator is pd.NA:
            assert pd.isna(summary.loc[metric, "denominator"])
        else:
            assert summary.loc[metric, "denominator"] == denominator
        assert summary.loc[metric, "support_n"] == support
        assert summary.loc[metric, "metric_value"] == value
        assert summary.loc[metric, "status"] == "available"
        assert summary.loc[metric, "reason"] == "computed"
    assert summary.loc["event_recall", "mature_n"] == 4
    assert summary.loc["event_recall", "censored_n"] == 1
    assert summary.loc["notification_precision", "mature_n"] == 3
    assert summary.loc["notification_precision", "censored_n"] == 1
    matches = first.event_matches
    assert matches["captured"].sum() == 3
    assert matches.loc[matches["event_status"] == "mature", "captured"].sum() == 3
    assert matches.loc[matches["event_status"] == "mature", "captured"].tolist() == [
        True,
        True,
        True,
        False,
    ]
    assert first.notifications["matched_event_count"].tolist() == [2, 1, 0, 0]
    assert first.notifications["maturity_status"].tolist() == [
        "mature",
        "mature",
        "mature",
        "immature",
    ]
    assert "2025-01-02" not in first.monitoring_summary.to_string()


def test_event_summary_empty_evidence_keeps_fixed_metric_rows() -> None:
    frame = _event_frame(pd.NaT)
    result = monitor_lifecycle(
        frame,
        _event_config(
            condition=MonitoringCondition(
                "atomic", "gt", "column", "feature", "literal", 9
            )
        ),
    )
    summary = result.monitoring_summary.loc[
        result.monitoring_summary["scope_key"] == "overall"
    ].set_index("metric")
    assert summary.loc["captured_event_count", "numerator"] == 0
    assert summary.loc["captured_event_count", "status"] == "available"
    for metric in (
        "event_recall",
        "notification_precision",
        "false_alert_share",
        "lead_time_mean",
        "lead_time_median",
        "warning_to_event_rate",
    ):
        numerator = summary.loc[metric, "numerator"]
        assert pd.isna(numerator) or numerator == 0
        assert summary.loc[metric, "denominator"] == 0
        assert pd.isna(summary.loc[metric, "metric_value"])
        assert summary.loc[metric, "support_n"] == 0
        assert summary.loc[metric, "status"] == "undefined"
        assert summary.loc[metric, "reason"] == "zero_denominator"


def test_event_summary_without_event_source_is_not_applicable() -> None:
    result = monitor_lifecycle(_frame(), _config())
    summary = result.monitoring_summary.loc[
        result.monitoring_summary["scope_key"] == "overall"
    ].set_index("metric")
    for metric in (
        "captured_event_count",
        "event_recall",
        "notification_precision",
        "false_alert_share",
        "false_positive_rate",
        "lead_time_mean",
        "lead_time_median",
        "warning_to_event_rate",
    ):
        assert pd.isna(summary.loc[metric, "numerator"])
        assert pd.isna(summary.loc[metric, "denominator"])
        assert pd.isna(summary.loc[metric, "metric_value"])
        assert summary.loc[metric, "support_n"] == 0
        assert summary.loc[metric, "status"] == "not_applicable"
        assert summary.loc[metric, "reason"] == "source_not_requested"


def test_dataframe_ranking_is_not_probability_and_task16_is_diagnostic_only() -> None:
    ranking_condition = MonitoringCondition(
        "atomic", "gt", "ranking_score", None, "literal", 0.5
    )
    frame = _frame().assign(score=[0.8, 0.2])
    config = replace(
        _config(ranking_condition),
        ranking_score_column="score",
        ranking_score_direction="higher_risk",
        scenarios=(
            WarningScenario(
                "reference",
                "model_score",
                (EarlyWarningRule("rule", 0, "high", ranking_condition),),
            ),
        ),
    )
    baseline = monitor_lifecycle(frame, config)
    assert baseline.rule_evaluations["truth"].tolist() == ["true", "false"]
    probability = replace(
        config,
        scenarios=(
            WarningScenario(
                "reference",
                "model_score",
                (
                    EarlyWarningRule(
                        "rule",
                        0,
                        "high",
                        MonitoringCondition(
                            "atomic", "gt", "event_probability", None, "literal", 0.5
                        ),
                    ),
                ),
            ),
        ),
    )
    assert monitor_lifecycle(frame, probability).rule_evaluations["truth"].tolist() == [
        "unknown",
        "unknown",
    ]
    audit = audit_data_quality(frame)
    diagnosed = monitor_lifecycle(frame, config, data_audit=audit)
    pd.testing.assert_frame_equal(baseline.rule_evaluations, diagnosed.rule_evaluations)
    assert (
        diagnosed.provenance.loc[
            diagnosed.provenance["provenance_key"] == "task16_snapshot_identity",
            "provenance_value",
        ].iat[0]
        == "unverified"
    )


def test_task15_ranking_and_probability_are_aligned_by_physical_position() -> None:
    frame = pd.DataFrame(
        {
            "entity": ["e"] * 6,
            "observed": [datetime(2025, 1, day) for day in range(1, 7)],
            "available": [datetime(2025, 1, day) for day in range(1, 7)],
            "feature": [0] * 6,
            "other": [0] * 6,
            "target": [0, 1, 0, 1, 0, 1],
        },
        index=[9] * 6,
    )
    ranking = MonitoringCondition(
        "atomic", "gt", "ranking_score", None, "literal", 0.35
    )
    probability = MonitoringCondition(
        "atomic", "gt", "event_probability", None, "literal", 0.45
    )
    config = replace(
        _config(ranking),
        analysis_as_of=datetime(2025, 1, 7),
        scenarios=(
            WarningScenario(
                "reference",
                "model_score",
                (
                    EarlyWarningRule("ranking", 0, "high", ranking),
                    EarlyWarningRule("probability", 1, "high", probability),
                ),
            ),
        ),
    )
    result = monitor_lifecycle(frame, config, risk_validation=_task15_result(frame))
    evaluations = result.rule_evaluations
    assert evaluations.loc[evaluations["rule_key"] == "ranking", "truth"].tolist() == [
        "false",
        "false",
        "false",
        "true",
        "true",
        "true",
    ]
    assert evaluations.loc[
        evaluations["rule_key"] == "probability", "truth"
    ].tolist() == ["false", "false", "false", "true", "true", "true"]
    assert (
        result.provenance.loc[
            result.provenance["provenance_key"] == "task15_evidence_status",
            "provenance_value",
        ].iat[0]
        == "provided"
    )


def test_event_resource_gates_accept_exact_maximums_and_reject_plus_one() -> None:
    config = _event_config()
    _event_resource_gates(config, range(0), 500_000)
    with pytest.raises(
        ValueError, match="lifecycle resource limit exceeded: event_match_rows"
    ):
        _event_resource_gates(config, range(0), 500_001)
    _event_resource_gates(config, range(11_000_000), 0)
    with pytest.raises(
        ValueError,
        match="lifecycle resource limit exceeded: event_match_scan_operations",
    ):
        _event_resource_gates(config, range(11_000_001), 0)


def test_monitoring_summary_counts_multilabel_facts() -> None:
    low = EarlyWarningRule("low", 1, "low", _condition())
    high = EarlyWarningRule("high", 0, "high", _condition())
    config = replace(
        _config(),
        scenarios=(WarningScenario("reference", "rule_set", (low, high)),),
        alert_level_ranks=(("low", 1), ("high", 2)),
    )
    summary = monitor_lifecycle(_frame(), config).monitoring_summary
    overall = summary.loc[summary["scope_key"] == "overall"].set_index("metric")
    assert overall.loc["warning_hit_count", "numerator"] == 4
    assert pd.isna(overall.loc["warning_hit_count", "denominator"])
    assert overall.loc["warning_hit_count", "support_n"] == 4
    assert overall.loc["warning_hit_count", "metric_value"] == 4.0
    assert overall.loc["warning_observation_rate", "numerator"] == 2
    assert overall.loc["warning_observation_rate", "denominator"] == 2
    assert overall.loc["warning_observation_rate", "support_n"] == 2
    assert overall.loc["warning_observation_rate", "metric_value"] == 1.0
    assert overall.loc["warned_entity_count", "numerator"] == 2
    assert overall.loc["warned_entity_count", "support_n"] == 2
    assert overall.loc["warned_entity_rate", "denominator"] == 2
    assert overall.loc["warned_entity_rate", "metric_value"] == 1.0
    assert overall.loc["persistent_warning_count", "numerator"] == 2
    assert overall.loc["persistent_warning_count", "support_n"] == 2
    assert overall.loc["persistent_warning_rate", "denominator"] == 2
    assert overall.loc["persistent_warning_rate", "metric_value"] == 1.0
    assert overall.loc["notification_count", "numerator"] == 4
    assert overall.loc["notifications_per_entity", "denominator"] == 2
    assert overall.loc["notifications_per_entity", "metric_value"] == 2.0
    assert overall.loc["episode_count", "numerator"] == 4
    scenario = summary.loc[summary["scope_key"] == "scenario"].set_index("metric")
    assert scenario.loc["overlap_count", "numerator"] == 2
    assert scenario.loc["conflict_count", "numerator"] == 0


def test_monitoring_summary_empty_episode_duration_is_undefined_not_zero() -> None:
    never = MonitoringCondition("atomic", "gt", "column", "feature", "literal", 9)
    summary = monitor_lifecycle(_frame(), _config(never)).monitoring_summary
    overall = summary.loc[summary["scope_key"] == "overall"].set_index("metric")
    assert overall.loc["notification_count", "numerator"] == 0
    assert overall.loc["notifications_per_entity", "denominator"] == 2
    assert overall.loc["notifications_per_entity", "metric_value"] == 0.0
    assert overall.loc["episode_count", "numerator"] == 0
    assert pd.isna(overall.loc["episode_duration_mean", "metric_value"])
    assert overall.loc["episode_duration_mean", "status"] == "undefined"
    assert overall.loc["episode_duration_mean", "reason"] == "zero_denominator"


def test_monitoring_summary_aggregates_resolved_and_open_episode_durations() -> None:
    frame = pd.DataFrame(
        {
            "entity": ["e"] * 6,
            "observed": [datetime(2025, 1, day) for day in range(1, 7)],
            "available": [datetime(2025, 1, day) for day in range(1, 7)],
            "feature": [1, 1, 0, 0, 1, 1],
            "other": [0] * 6,
        }
    )
    rule = EarlyWarningRule("rule", 0, "high", _condition(), 2, 2, timedelta(days=2))
    result = monitor_lifecycle(
        frame, _persistence_config(rule, analysis_as_of=datetime(2025, 1, 7))
    )
    overall = result.monitoring_summary.loc[
        result.monitoring_summary["scope_key"] == "overall"
    ].set_index("metric")
    assert overall.loc["notification_count", "numerator"] == 2
    assert overall.loc["notifications_per_entity", "denominator"] == 1
    assert overall.loc["notifications_per_entity", "metric_value"] == 2.0
    assert overall.loc["episode_count", "numerator"] == 2
    assert overall.loc["open_episode_count", "numerator"] == 1
    assert overall.loc["resolved_episode_count", "numerator"] == 1
    assert overall.loc["episode_duration_mean", "metric_value"] == 129600.0
    assert overall.loc["episode_duration_median", "metric_value"] == 129600.0
    assert result.alert_episodes["episode_ordinal"].tolist() == [0, 1]


def test_module_contains_no_production_notification_side_effect_imports() -> None:
    source = Path(lifecycle_monitoring.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "import requests",
        "import socket",
        "import smtplib",
        "import subprocess",
        "import threading",
        "import urllib",
    ):
        assert forbidden not in source


def test_loss_evidence_is_private_position_aligned_and_populates_approved_metrics() -> (
    None
):
    frame = _frame().assign(exposure=[0, 25], observed_loss_value=[0, 4])
    config = replace(
        _config(),
        exposure_column="exposure",
        loss_fraction=0.4,
        observed_loss_column="observed_loss_value",
        observed_loss_is_mature_snapshot=True,
    )
    probabilities = pd.Series([0.2, pd.NA], dtype="Float64")
    evidence, metadata = lifecycle_monitoring._loss_evidence(
        frame, config, probabilities
    )
    assert evidence["row_position"].tolist() == [0, 1]
    assert evidence["exposure"].tolist() == [0.0, 25.0]
    assert evidence["loss_fraction"].tolist() == [0.4, 0.4]
    assert evidence["observed_loss"].tolist() == [0.0, 4.0]
    assert evidence["probability"].tolist()[0] == 0.2
    assert pd.isna(evidence["probability"].tolist()[1])
    assert metadata == {
        "exposure_source": "dataframe",
        "observed_loss_source": "dataframe",
        "loss_fraction_source": "scalar",
    }
    classified = lifecycle_monitoring._classify_loss_evidence(
        evidence, config, "task15"
    )
    assert classified["expected_status"].tolist() == ["available", "unavailable"]
    assert classified["expected_reason"].tolist() == [
        "computed",
        "probability_unavailable",
    ]
    assert classified["observed_status"].tolist() == ["available", "available"]
    result = monitor_lifecycle(frame, config)
    monitoring = _base_monitoring_by_metric(result)
    assert monitoring.loc["exposure_sum", "metric_value"] == 25.0
    assert monitoring.loc["observed_loss_sum", "metric_value"] == 4.0
    assert monitoring.loc["expected_loss_sum", "status"] == "not_applicable"
    lifecycle = result.lifecycle_summary.set_index("metric")
    assert lifecycle.loc["observed_loss_sum", "metric_value"] == 4.0


def test_loss_evidence_classification_preserves_source_and_component_precedence() -> (
    None
):
    frame = _frame().assign(exposure=[None, 0], observed_loss_value=[3, None])
    config = replace(
        _config(),
        exposure_column="exposure",
        loss_fraction=0,
        observed_loss_column="observed_loss_value",
        observed_loss_is_mature_snapshot=True,
    )
    evidence, _ = lifecycle_monitoring._loss_evidence(
        frame, config, pd.Series([pd.NA, 0.4], dtype="Float64")
    )
    source_missing = lifecycle_monitoring._classify_loss_evidence(
        evidence, config, "not_requested"
    )
    assert source_missing["expected_reason"].tolist() == [
        "source_not_requested",
        "source_not_requested",
    ]
    classified = lifecycle_monitoring._classify_loss_evidence(
        evidence, config, "task15"
    )
    assert classified["expected_reason"].tolist() == [
        "probability_unavailable",
        "computed",
    ]
    assert classified["observed_reason"].tolist() == [
        "computed",
        "observed_loss_unavailable",
    ]


def test_observed_loss_maturity_uses_declared_availability() -> None:
    frame = _frame().assign(observed_loss_value=[2, 3])
    config = replace(
        _config(),
        analysis_as_of=datetime(2025, 1, 2),
        observed_loss_column="observed_loss_value",
        observed_loss_available_time_column="available",
    )
    evidence, _ = lifecycle_monitoring._loss_evidence(
        frame, config, pd.Series([pd.NA, pd.NA], dtype="Float64")
    )
    classified = lifecycle_monitoring._classify_loss_evidence(
        evidence, config, "not_requested"
    )
    assert classified["observed_reason"].tolist() == ["computed", "computed"]


def test_loss_metric_rows_are_not_applicable_when_sources_are_not_requested() -> None:
    result = monitor_lifecycle(_frame(), _config())
    monitoring = _base_monitoring_by_metric(result)
    for metric in ("expected_loss_sum", "expected_loss_rate"):
        assert monitoring.loc[metric, "status"] == "not_applicable"
        assert monitoring.loc[metric, "reason"] == "source_not_requested"
    for table in (result.monitoring_summary, result.lifecycle_summary):
        rows = table.loc[
            table["metric"].isin(["observed_loss_sum", "observed_loss_rate"])
        ]
        assert set(rows["status"]) == {"not_applicable"}
        assert set(rows["reason"]) == {"source_not_requested"}
    assert result.scenario_comparison.empty


def test_observed_loss_metrics_are_hand_calculated_and_zero_is_valid() -> None:
    frame = _frame().assign(exposure=[100, 200], observed_loss_value=[0, 30])
    config = replace(
        _config(),
        exposure_column="exposure",
        observed_loss_column="observed_loss_value",
        observed_loss_is_mature_snapshot=True,
    )
    result = monitor_lifecycle(frame, config)
    monitoring = _base_monitoring_by_metric(result)
    lifecycle = result.lifecycle_summary.set_index("metric")
    for summary in (monitoring, lifecycle):
        assert summary.loc["observed_loss_sum", "numerator"] == 30.0
        assert summary.loc["observed_loss_sum", "metric_value"] == 30.0
        assert summary.loc["observed_loss_sum", "support_n"] == 2
        assert summary.loc["observed_loss_sum", "status"] == "available"
        assert summary.loc["observed_loss_sum", "reason"] == "computed"
        assert summary.loc["observed_loss_sum", "unit"] == "exposure_unit"
        assert summary.loc["observed_loss_rate", "denominator"] == 300.0
        assert summary.loc["observed_loss_rate", "metric_value"] == 0.1


def _loss_metric_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity": ["e"] * 6,
            "observed": [datetime(2025, 1, day) for day in range(1, 7)],
            "available": [datetime(2025, 1, day) for day in range(1, 7)],
            "feature": [1] * 6,
            "other": [0] * 6,
            "target": [0, 1, 0, 1, 0, 1],
            "exposure": [100, 200, 100, 0, 0, 0],
            "observed_loss_value": [10, 20, 30, 0, 0, 0],
        }
    )


def _loss_metric_result(frame: pd.DataFrame, **changes: object):
    defaults = {
        "analysis_as_of": datetime(2025, 1, 10),
        "exposure_column": "exposure",
        "loss_fraction": 0.5,
        "observed_loss_column": "observed_loss_value",
        "observed_loss_is_mature_snapshot": True,
    }
    defaults.update(changes)
    config = replace(_config(), **defaults)
    return monitor_lifecycle(frame, config, risk_validation=_task15_result(frame))


def test_lifecycle_loss_metrics_exposure_sum_hand_calculated() -> None:
    row = _base_monitoring_by_metric(_loss_metric_result(_loss_metric_frame())).loc[
        "exposure_sum"
    ]
    assert (
        row["scope_key"] == "overall"
        and row["numerator"] == 400.0
        and pd.isna(row["denominator"])
    )
    assert (
        row["support_n"] == 6
        and row["metric_value"] == 400.0
        and row["status"] == "available"
        and row["reason"] == "computed"
        and row["unit"] == "exposure_unit"
    )


def test_lifecycle_loss_metrics_expected_loss_hand_calculated() -> None:
    summary = _base_monitoring_by_metric(_loss_metric_result(_loss_metric_frame()))
    assert summary.loc["expected_loss_sum", "numerator"] == 60.0
    assert pd.isna(summary.loc["expected_loss_sum", "denominator"])
    assert summary.loc["expected_loss_sum", "support_n"] == 6
    assert summary.loc["expected_loss_sum", "metric_value"] == 60.0
    assert summary.loc["expected_loss_rate", "denominator"] == 400.0
    assert summary.loc["expected_loss_rate", "metric_value"] == 0.15


def test_lifecycle_loss_metrics_expected_and_observed_are_independent() -> None:
    frame = _loss_metric_frame()
    frame["observed_loss_value"] = [10.0, 20.0, 40.0, 0.0, 0.0, 0.0]
    summary = _base_monitoring_by_metric(_loss_metric_result(frame))
    assert summary.loc["expected_loss_sum", "metric_value"] == 60.0
    assert summary.loc["observed_loss_sum", "metric_value"] == 70.0
    assert (
        summary.loc["expected_loss_sum", "metric_value"]
        != summary.loc["observed_loss_sum", "metric_value"]
    )
    assert (
        summary.loc["expected_loss_sum", "status"]
        == summary.loc["observed_loss_sum", "status"]
        == "available"
    )


def test_lifecycle_loss_metrics_expected_partial_evidence_not_aggregated() -> None:
    frame = _loss_metric_frame()
    frame.loc[frame.index[2], "exposure"] = None
    summary = _base_monitoring_by_metric(_loss_metric_result(frame))
    for metric in ("expected_loss_sum", "expected_loss_rate"):
        assert pd.isna(summary.loc[metric, "metric_value"])
        assert (
            summary.loc[metric, "status"] == "unavailable"
            and summary.loc[metric, "reason"] == "exposure_unavailable"
        )


def test_lifecycle_loss_metrics_observed_immature_evidence_affects_aggregate() -> None:
    frame = _loss_metric_frame().assign(
        observed_available=[datetime(2025, 1, 1)] * 5 + [datetime(2025, 1, 20)]
    )
    result = _loss_metric_result(
        frame,
        observed_loss_is_mature_snapshot=False,
        observed_loss_available_time_column="observed_available",
    )
    row = _base_monitoring_by_metric(result).loc["observed_loss_sum"]
    assert (
        pd.isna(row["metric_value"])
        and row["status"] == "not_verifiable"
        and row["reason"] == "observed_loss_not_mature"
    )


def test_lifecycle_loss_metrics_observed_mature_missing_not_zero() -> None:
    frame = _loss_metric_frame()
    frame.loc[frame.index[1], "observed_loss_value"] = None
    row = _base_monitoring_by_metric(_loss_metric_result(frame)).loc[
        "observed_loss_sum"
    ]
    assert (
        pd.isna(row["metric_value"])
        and row["status"] == "unavailable"
        and row["reason"] == "observed_loss_unavailable"
    )


def test_ir05_observed_loss_requires_common_exposure_support() -> None:
    frame = _loss_metric_frame()
    frame.loc[frame.index[1], "exposure"] = None
    result = _loss_metric_result(frame)
    for table in (result.monitoring_summary, result.lifecycle_summary):
        rows = table.loc[
            table["metric"].isin(["observed_loss_sum", "observed_loss_rate"])
        ]
        rows = rows.loc[rows["scope_key"] == "overall"]
        assert set(rows["status"]) == {"not_verifiable"}
        assert set(rows["reason"]) == {"exposure_unavailable"}
        assert rows["numerator"].isna().all()
        assert rows["denominator"].isna().all()


def test_ir05_missing_exposure_on_zero_loss_row_still_invalidates_support() -> None:
    frame = _loss_metric_frame()
    frame.loc[frame.index[1], "observed_loss_value"] = 0
    frame.loc[frame.index[1], "exposure"] = None
    rows = _base_monitoring_by_metric(_loss_metric_result(frame))
    assert rows.loc["observed_loss_sum", "status"] == "not_verifiable"
    assert rows.loc["observed_loss_rate", "status"] == "not_verifiable"


def test_ir05_expected_loss_remains_independent_of_observed_support() -> None:
    frame = _loss_metric_frame()
    frame.loc[frame.index[1], "observed_loss_value"] = None
    rows = _base_monitoring_by_metric(_loss_metric_result(frame))
    assert rows.loc["expected_loss_sum", "metric_value"] == 60.0
    assert rows.loc["observed_loss_sum", "status"] == "unavailable"


def test_ir05_grouped_loss_support_is_isolated() -> None:
    frame = _loss_metric_frame().assign(segment=["A", "A", "B", "B", "B", "B"])
    frame.loc[frame.index[2], "exposure"] = None
    config = replace(
        _loss_metric_result(frame).provenance.attrs.get("config", _config())
        if False
        else _config(),
        exposure_column="exposure",
        observed_loss_column="observed_loss_value",
        observed_loss_is_mature_snapshot=True,
        segment_columns=("segment",),
        analysis_as_of=datetime(2025, 1, 10),
    )
    result = monitor_lifecycle(frame, config)
    rows = result.monitoring_summary.loc[
        result.monitoring_summary["metric"] == "observed_loss_rate"
    ]
    assert rows.loc[rows["scope_key"] == "overall", "status"].iat[0] == "not_verifiable"
    assert rows["scope_key"].notna().all()


def test_ir05_lifecycle_grouped_loss_support_is_not_partial() -> None:
    frame = _loss_metric_frame().assign(segment=["A", "A", "B", "B", "B", "B"])
    frame.loc[frame.index[2], "exposure"] = None
    config = replace(
        _config(),
        analysis_as_of=datetime(2025, 1, 10),
        exposure_column="exposure",
        observed_loss_column="observed_loss_value",
        observed_loss_is_mature_snapshot=True,
        segment_columns=("segment",),
    )
    result = monitor_lifecycle(frame, config)
    rows = result.lifecycle_summary.loc[
        result.lifecycle_summary["metric"] == "observed_loss_rate"
    ]
    assert rows["scope_key"].isin(["segment_time"]).any()
    assert (
        rows.loc[rows["scope_key"] == "segment_time", "status"]
        == "not_verifiable"
    ).any()


def test_ir05_zero_exposure_keeps_zero_denominator_semantics() -> None:
    frame = _loss_metric_frame().assign(exposure=[0, 0, 0, 0, 0, 0])
    rows = _base_monitoring_by_metric(_loss_metric_result(frame))
    assert rows.loc["observed_loss_rate", "status"] == "undefined"
    assert rows.loc["observed_loss_rate", "reason"] == "zero_denominator"


def test_lifecycle_loss_metrics_ranking_does_not_enable_expected_loss() -> None:
    frame = _loss_metric_frame().assign(score=[0.1] * 6)
    config = replace(
        _config(),
        analysis_as_of=datetime(2025, 1, 10),
        ranking_score_column="score",
        ranking_score_direction="higher_risk",
        exposure_column="exposure",
        loss_fraction=0.5,
    )
    summary = _base_monitoring_by_metric(monitor_lifecycle(frame, config))
    for metric in ("expected_loss_sum", "expected_loss_rate"):
        assert pd.isna(summary.loc[metric, "metric_value"])
        assert (
            summary.loc[metric, "status"] == "not_applicable"
            and summary.loc[metric, "reason"] == "source_not_requested"
        )


def _scoped_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity": ["e0", "e0", "e1", "e1", "e2", "e2"],
            "observed": [
                datetime(2025, 1, 2),
                datetime(2025, 1, 3),
                datetime(2025, 1, 2),
                datetime(2025, 1, 3),
                datetime(2025, 1, 2),
                datetime(2025, 1, 3),
            ],
            "available": [
                datetime(2025, 1, 2),
                datetime(2025, 1, 3),
                datetime(2025, 1, 2),
                datetime(2025, 1, 3),
                datetime(2025, 1, 2),
                datetime(2025, 1, 3),
            ],
            "feature": [1, 0, 1, 1, 0, 0],
            "other": [0] * 6,
            "segment": [
                "Z",
                "A",
                "Z",
                "M",
                "A",
                "M",
            ],
            "cohort": [
                "SECRET_COHORT_BETA",
                "A",
                "SECRET_COHORT_BETA",
                "C",
                "A",
                "C",
            ],
            "vintage_origin": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 1),
                datetime(2025, 1, 1),
                datetime(2025, 1, 1),
                datetime(2025, 1, 2),
                datetime(2025, 1, 2),
            ],
        }
    )


def _scoped_summary_result(frame: pd.DataFrame | None = None):
    return monitor_lifecycle(
        _scoped_summary_frame() if frame is None else frame,
        replace(
            _config(),
            segment_columns=("segment",),
            cohort_column="cohort",
            cohort_time_column="vintage_origin",
        ),
    )


def _scope_metric(result, scope_key: str) -> pd.DataFrame:
    return (
        result.monitoring_summary.loc[
            (result.monitoring_summary["scope_key"] == scope_key)
            & (result.monitoring_summary["metric"] == "warning_hit_count")
        ]
        .sort_values("scope_position", kind="stable")
        .reset_index(drop=True)
    )


def _base_monitoring_by_metric(result) -> pd.DataFrame:
    """Retain the one-scenario base projection used by existing loss checks."""
    summary = result.monitoring_summary
    return summary.loc[
        (summary["scope_key"] == "overall")
        | (
            (summary["scope_key"] == "scenario")
            & summary["metric"].isin({"expected_loss_sum", "expected_loss_rate"})
        )
    ].set_index("metric")


def test_lifecycle_summary_segment_scope_hand_calculated() -> None:
    rows = _scope_metric(_scoped_summary_result(), "scenario_segment")
    assert rows["scope_position"].tolist() == [0, 1, 2]
    assert rows["numerator"].tolist() == [2.0, 0.0, 1.0]
    assert rows["support_n"].tolist() == [2, 2, 2]
    assert rows["metric_value"].tolist() == [2.0, 0.0, 1.0]
    assert rows["status"].tolist() == ["available"] * 3
    assert rows["reason"].tolist() == ["computed"] * 3


def test_lifecycle_summary_cohort_scope_hand_calculated() -> None:
    result = _scoped_summary_result()
    rows = _scope_metric(result, "scenario_cohort")
    assert rows["scope_position"].tolist() == [0, 1, 2]
    assert rows["numerator"].tolist() == [2.0, 0.0, 1.0]
    assert rows["support_n"].tolist() == [2, 2, 2]
    assert rows["status"].tolist() == ["available"] * 3
    assert result.observation_history["cohort_position"].tolist() == [0, 1, 0, 2, 1, 2]


def test_lifecycle_summary_vintage_scope_hand_calculated() -> None:
    result = _scoped_summary_result()
    rows = _scope_metric(result, "scenario_vintage")
    assert rows["scope_position"].tolist() == [0, 1, 2]
    assert rows["numerator"].tolist() == [0.0, 2.0, 1.0]
    assert rows["support_n"].tolist() == [1, 3, 2]
    assert rows["status"].tolist() == ["available"] * 3
    assert result.observation_history["period_index"].tolist() == [1, 2, 1, 2, 0, 1]
    assert set(result.lifecycle_summary["scope_key"]) >= {
        "segment_time",
        "cohort_time",
        "vintage_state",
    }


def test_lifecycle_summary_cohort_and_vintage_scopes_are_independent() -> None:
    result = _scoped_summary_result()
    cohort = _scope_metric(result, "scenario_cohort")
    vintage = _scope_metric(result, "scenario_vintage")
    assert cohort[["support_n", "numerator"]].values.tolist() == [
        [2, 2.0],
        [2, 0.0],
        [2, 1.0],
    ]
    assert vintage[["support_n", "numerator"]].values.tolist() == [
        [1, 0.0],
        [3, 2.0],
        [2, 1.0],
    ]


def test_lifecycle_summary_scope_rows_preserve_raw_label_privacy() -> None:
    frame = _scoped_summary_frame()
    frame.loc[frame.index[0], "segment"] = "SECRET_SEGMENT_ALPHA"
    result = _scoped_summary_result(frame)
    rendered = "\n".join(
        str(value)
        for value in (
            result.monitoring_summary,
            result.lifecycle_summary,
            result.scenario_comparison,
            result.provenance,
            repr(result),
        )
    )
    for secret in (
        "SECRET_SEGMENT_ALPHA",
        "SECRET_COHORT_BETA",
        "SECRET_VINTAGE_GAMMA",
    ):
        assert secret not in rendered


def test_lifecycle_summary_segment_ordinals_follow_first_appearance() -> None:
    rows = _scope_metric(_scoped_summary_result(), "scenario_segment")
    assert rows["scope_position"].tolist() == [0, 1, 2]
    assert rows["numerator"].tolist() == [2.0, 0.0, 1.0]


def test_lifecycle_summary_group_expansion_preserves_base_rows() -> None:
    frame = _scoped_summary_frame()
    base = monitor_lifecycle(
        frame.drop(columns=["segment", "cohort", "vintage_origin"]), _config()
    )
    scoped = _scoped_summary_result(frame)
    pd.testing.assert_frame_equal(
        base.monitoring_summary,
        scoped.monitoring_summary.loc[
            scoped.monitoring_summary["scope_key"].isin(
                set(base.monitoring_summary["scope_key"])
            )
        ].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        base.lifecycle_summary,
        scoped.lifecycle_summary.loc[
            scoped.lifecycle_summary["scope_key"].isin(
                {"overall", "state", "transition"}
            )
        ].reset_index(drop=True),
    )


def test_lifecycle_summary_cohort_missing_bucket_is_anonymous_and_last() -> None:
    frame = _scoped_summary_frame()
    frame.loc[frame.index[5], "cohort"] = None
    result = _scoped_summary_result(frame)
    assert result.observation_history["cohort_position"].tolist() == [0, 1, 0, 2, 1, 3]
    rows = _scope_metric(result, "scenario_cohort")
    assert rows["scope_position"].tolist() == [0, 1, 2, 3]
    assert rows["support_n"].tolist() == [2, 2, 1, 1]


def test_lifecycle_summary_rejects_unsafe_group_scalars_without_rendering_them(
) -> None:
    class _UnsafeGroup:
        def __repr__(self) -> str:
            raise AssertionError("repr must not be called")

    frame = _scoped_summary_frame()
    frame["segment"] = pd.Series(
        [_UnsafeGroup(), "A", "Z", "M", "A", "M"], dtype=object
    )
    with pytest.raises(
        ValueError, match="lifecycle input schema is invalid: unsupported_dtype"
    ) as raised:
        _scoped_summary_result(frame)
    assert "UnsafeGroup" not in str(raised.value)


def test_lifecycle_summary_does_not_generate_disallowed_scope_combinations() -> None:
    result = _scoped_summary_result()
    monitoring_scopes = set(result.monitoring_summary["scope_key"])
    lifecycle_scopes = set(result.lifecycle_summary["scope_key"])
    assert "scenario_segment_time" not in monitoring_scopes
    assert "scenario_cohort_time" not in monitoring_scopes
    assert "state_segment" not in lifecycle_scopes
    assert "transition_vintage" not in lifecycle_scopes
    assert result.scenario_comparison.empty


class _MaliciousLossValue:
    def __init__(self) -> None:
        self.calls = 0

    def _called(self, *_: object, **__: object) -> object:
        self.calls += 1
        return 0

    __float__ = _called
    __array__ = _called
    __eq__ = _called
    __ne__ = _called
    __lt__ = _called
    __le__ = _called
    __gt__ = _called
    __ge__ = _called
    __iter__ = _called
    __str__ = _called
    __repr__ = _called
    __hash__ = _called


@pytest.mark.parametrize(
    "source",
    ["exposure", "loss_fraction_scalar", "loss_fraction_column", "observed_loss"],
)
def test_loss_evidence_rejects_malicious_values_without_protocol_calls(
    source: str,
) -> None:
    sentinel = _MaliciousLossValue()
    frame = _frame().assign(
        exposure=[1, 2], loss_fraction_column=[0.2, 0.3], observed_loss_value=[1, 2]
    )
    changes: dict[str, object] = {"observed_loss_is_mature_snapshot": True}
    if source == "exposure":
        frame["exposure"] = pd.Series([sentinel, 2], index=frame.index, dtype="object")
        changes["exposure_column"] = "exposure"
    elif source == "loss_fraction_scalar":
        changes["loss_fraction"] = sentinel
    elif source == "loss_fraction_column":
        frame["loss_fraction_column"] = pd.Series(
            [sentinel, 0.3], index=frame.index, dtype="object"
        )
        changes["loss_fraction"] = "loss_fraction_column"
    else:
        frame["observed_loss_value"] = pd.Series(
            [sentinel, 2], index=frame.index, dtype="object"
        )
        changes["observed_loss_column"] = "observed_loss_value"
    with pytest.raises(
        ValueError, match="lifecycle input schema is invalid: loss_evidence_invalid"
    ) as error:
        monitor_lifecycle(frame, replace(_config(), **changes))
    assert sentinel.calls == 0
    assert "Malicious" not in str(error.value)


# Task 18D-1b-2: frozen scenario-comparison contract evidence.


def _d1b2_comparison_config() -> LifecycleMonitoringConfig:
    """Build two scenarios with deliberate positive, zero, and negative deltas."""
    positive = MonitoringCondition("atomic", "gt", "column", "feature", "literal", 0)
    strict = MonitoringCondition("atomic", "gt", "column", "feature", "literal", 1)
    reference = WarningScenario(
        "reference",
        "rule_set",
        (
            EarlyWarningRule("negative", 0, "high", positive),
            EarlyWarningRule("zero", 1, "medium", positive),
            EarlyWarningRule("positive", 2, "low", strict),
            EarlyWarningRule("reference_only", 3, "low", positive),
        ),
    )
    challenger = WarningScenario(
        "challenger",
        "rule_set",
        (
            EarlyWarningRule("negative", 0, "high", strict),
            EarlyWarningRule("zero", 1, "medium", positive),
            EarlyWarningRule("positive", 2, "low", positive),
            EarlyWarningRule("challenger_only", 3, "low", positive),
        ),
    )
    return replace(
        _config(),
        scenarios=(reference, challenger),
        alert_level_ranks=(("high", 3), ("medium", 2), ("low", 1)),
    )


def _d1b2_comparison_result():
    return monitor_lifecycle(_frame(), _d1b2_comparison_config())


def _d1b2_summary_row(
    scenario_key: str,
    *,
    scope_key: str = "scenario",
    scope_position: object = 0,
    rule_key: object = pd.NA,
    metric: str = "warning_hit_count",
    metric_value: object = 2.0,
    numerator: object = 2.0,
    denominator: object = 4.0,
    support_n: int = 4,
    support_unit: object = "observation",
    mature_n: int = 4,
    censored_n: int = 0,
    unit: object = "count",
    status: str = "available",
    reason: str = "computed",
    finding_key: str = "monitoring:synthetic",
) -> dict[str, object]:
    """Create one frozen monitoring-summary source row for alignment tests."""
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
        "finding_key": finding_key,
    }


def _d1b2_synthetic_comparison_rows(
    *rows: dict[str, object],
) -> list[dict[str, object]]:
    return lifecycle_monitoring._scenario_comparison_rows(
        _d1b2_comparison_config(), pd.DataFrame(rows)
    )


def test_d1b2_scenario_comparison_typed_empty_schema_is_exact() -> None:
    comparison = monitor_lifecycle(_frame(), _config()).scenario_comparison
    assert comparison.empty
    assert comparison.columns.tolist() == [
        "reference_scenario_key",
        "comparator_scenario_key",
        "metric",
        "scope_key",
        "scope_position",
        "rule_key",
        "reference_value",
        "comparator_value",
        "delta",
        "numerator",
        "denominator",
        "support_n",
        "support_unit",
        "status",
        "reason",
        "finding_key",
    ]
    assert {column: str(dtype) for column, dtype in comparison.dtypes.items()} == {
        "reference_scenario_key": "string",
        "comparator_scenario_key": "string",
        "metric": "string",
        "scope_key": "string",
        "scope_position": "Int64",
        "rule_key": "string",
        "reference_value": "Float64",
        "comparator_value": "Float64",
        "delta": "Float64",
        "numerator": "Float64",
        "denominator": "Float64",
        "support_n": "int64",
        "support_unit": "string",
        "status": "string",
        "reason": "string",
        "finding_key": "string",
    }


def test_d1b2_scenario_and_rule_normalization_ignores_source_scenario_ordinal() -> None:
    result = _d1b2_comparison_result()
    source = result.monitoring_summary
    comparison = result.scenario_comparison

    assert set(
        source.loc[
            (source["scope_key"] == "scenario")
            & (source["scenario_key"].isin(["reference", "challenger"])),
            "scope_position",
        ]
    ) == {0, 1}
    scenario_rows = comparison.loc[comparison["scope_key"] == "scenario"]
    assert not scenario_rows.empty
    assert scenario_rows["scope_position"].isna().all()
    assert scenario_rows["rule_key"].isna().all()

    rules = comparison.loc[comparison["scope_key"] == "scenario_rule"]
    assert rules["scope_position"].isna().all()
    assert set(rules["rule_key"]) == {"negative", "zero", "positive"}
    assert "reference_only" not in set(rules["rule_key"])
    assert "challenger_only" not in set(rules["rule_key"])


def test_d1b2_scenario_rule_deltas_keep_positive_zero_and_negative_values() -> None:
    rows = _d1b2_comparison_result().scenario_comparison
    rules = rows.loc[
        (rows["scope_key"] == "scenario_rule") & (rows["metric"] == "warning_hit_count")
    ].set_index("rule_key")
    assert rules.loc["negative", "delta"] < 0
    assert rules.loc["zero", "delta"] == 0.0
    assert rules.loc["positive", "delta"] > 0
    assert rules.loc["negative", "delta"] == (
        rules.loc["negative", "comparator_value"]
        - rules.loc["negative", "reference_value"]
    )


@pytest.mark.parametrize(
    ("reference_value", "comparator_value", "expected_delta"),
    ((0.20, 0.35, 0.15), (0.40, 0.10, -0.30), (0.20, 0.20, 0.0)),
)
def test_d1b2_scenario_delta_is_comparator_minus_reference_by_hand(
    reference_value: float,
    comparator_value: float,
    expected_delta: float,
) -> None:
    rows = _d1b2_synthetic_comparison_rows(
        _d1b2_summary_row(
            "reference",
            metric_value=reference_value,
            numerator=reference_value,
        ),
        _d1b2_summary_row(
            "challenger",
            metric_value=comparator_value,
            numerator=comparator_value,
        ),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["reference_scenario_key"] == "reference"
    assert row["comparator_scenario_key"] == "challenger"
    assert row["scope_key"] == "scenario"
    assert pd.isna(row["scope_position"])
    assert pd.isna(row["rule_key"])
    assert row["reference_value"] == reference_value
    assert row["comparator_value"] == comparator_value
    assert row["delta"] == pytest.approx(expected_delta)
    assert row["support_n"] == 4
    assert row["support_unit"] == "observation"
    assert (row["status"], row["reason"]) == ("available", "computed")


def test_d1b2_comparison_contains_only_eligible_source_backed_metric_scope_rows() -> (
    None
):
    eligible = (
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
    source_rows: list[dict[str, object]] = []
    for position, scope_key in enumerate(eligible):
        for scenario_key, source_position in (("reference", 0), ("challenger", 1)):
            source_rows.append(
                _d1b2_summary_row(
                    scenario_key,
                    scope_key=scope_key,
                    scope_position=(
                        source_position
                        if scope_key in {"scenario", "scenario_rule"}
                        else position
                    ),
                    rule_key="negative" if scope_key == "scenario_rule" else pd.NA,
                )
            )
    for scope_key in ("overall", "segment_time", "cohort_time", "vintage_state"):
        source_rows.extend(
            (
                _d1b2_summary_row("reference", scope_key=scope_key),
                _d1b2_summary_row("challenger", scope_key=scope_key),
            )
        )
    for scope_key, metric in (
        ("scenario_rule", "overlap_count"),
        ("scenario_state", "captured_event_count"),
    ):
        source_rows.extend(
            (
                _d1b2_summary_row("reference", scope_key=scope_key, metric=metric),
                _d1b2_summary_row("challenger", scope_key=scope_key, metric=metric),
            )
        )
    comparison = pd.DataFrame(_d1b2_synthetic_comparison_rows(*source_rows))
    assert set(comparison["scope_key"]) == set(eligible)
    assert not set(comparison["scope_key"]) & {
        "overall",
        "segment_time",
        "cohort_time",
        "vintage_state",
    }
    assert not (
        comparison["metric"].isin({"overlap_count", "conflict_count"})
        & ~comparison["scope_key"].isin(
            {"scenario", "scenario_segment", "scenario_time"}
        )
    ).any()
    assert not (
        comparison["metric"].isin(
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
        & comparison["scope_key"].isin(
            {"scenario_alert_level", "scenario_state", "scenario_transition"}
        )
    ).any()


@pytest.mark.parametrize(
    "scope_key",
    [
        "scenario_alert_level",
        "scenario_segment",
        "scenario_time",
        "scenario_cohort",
        "scenario_vintage",
        "scenario_state",
        "scenario_transition",
    ],
)
def test_d1b2_subordinate_scope_requires_and_keeps_matching_normalized_position(
    scope_key: str,
) -> None:
    rows = _d1b2_synthetic_comparison_rows(
        _d1b2_summary_row("reference", scope_key=scope_key, scope_position=3),
        _d1b2_summary_row("challenger", scope_key=scope_key, scope_position=3),
    )
    assert len(rows) == 1
    assert rows[0]["scope_key"] == scope_key
    assert rows[0]["scope_position"] == 3
    assert pd.isna(rows[0]["rule_key"])


@pytest.mark.parametrize(
    "scope_key",
    [
        "scenario_alert_level",
        "scenario_segment",
        "scenario_time",
        "scenario_cohort",
        "scenario_vintage",
        "scenario_state",
        "scenario_transition",
    ],
)
def test_d1b2_subordinate_scope_mismatched_position_does_not_align(
    scope_key: str,
) -> None:
    rows = _d1b2_synthetic_comparison_rows(
        _d1b2_summary_row("reference", scope_key=scope_key, scope_position=0),
        _d1b2_summary_row("challenger", scope_key=scope_key, scope_position=1),
    )
    assert rows == []


def test_d1b2_support_mismatch_is_not_verifiable_without_intersection() -> None:
    rows = _d1b2_synthetic_comparison_rows(
        _d1b2_summary_row("reference", support_n=4),
        _d1b2_summary_row("challenger", support_n=3),
    )
    assert len(rows) == 1
    row = rows[0]
    assert (row["status"], row["reason"]) == (
        "not_verifiable",
        "support_not_comparable",
    )
    for column in (
        "reference_value",
        "comparator_value",
        "delta",
        "numerator",
        "denominator",
        "support_unit",
    ):
        assert pd.isna(row[column])
    assert row["support_n"] == 0


@pytest.mark.parametrize(
    (
        "reference_status",
        "reference_reason",
        "comparator_status",
        "comparator_reason",
        "expected_status",
        "expected_reason",
    ),
    [
        (
            "unavailable",
            "probability_unavailable",
            "available",
            "computed",
            "unavailable",
            "probability_unavailable",
        ),
        (
            "available",
            "computed",
            "unavailable",
            "exposure_unavailable",
            "unavailable",
            "exposure_unavailable",
        ),
        (
            "unavailable",
            "probability_unavailable",
            "unavailable",
            "exposure_unavailable",
            "unavailable",
            "probability_unavailable",
        ),
        (
            "undefined",
            "zero_denominator",
            "not_applicable",
            "source_not_requested",
            "not_applicable",
            "source_not_requested",
        ),
    ],
)
def test_d1b2_unavailable_comparison_precedence(
    reference_status: str,
    reference_reason: str,
    comparator_status: str,
    comparator_reason: str,
    expected_status: str,
    expected_reason: str,
) -> None:
    rows = _d1b2_synthetic_comparison_rows(
        _d1b2_summary_row(
            "reference",
            status=reference_status,
            reason=reference_reason,
        ),
        _d1b2_summary_row(
            "challenger",
            status=comparator_status,
            reason=comparator_reason,
        ),
    )
    assert len(rows) == 1
    row = rows[0]
    assert (row["status"], row["reason"]) == (expected_status, expected_reason)
    for column in (
        "reference_value",
        "comparator_value",
        "delta",
        "numerator",
        "denominator",
        "support_unit",
    ):
        assert pd.isna(row[column])
    assert row["support_n"] == 0


def test_d1b2_grouped_comparison_preserves_privacy_and_never_selects_a_winner() -> None:
    rows = _d1b2_synthetic_comparison_rows(
        _d1b2_summary_row(
            "reference",
            scope_key="scenario_segment",
            scope_position=0,
            finding_key="monitoring:SECRET_SEGMENT_D1B2",
        ),
        _d1b2_summary_row(
            "challenger",
            scope_key="scenario_segment",
            scope_position=0,
            finding_key="monitoring:SECRET_COHORT_D1B2",
        ),
    )
    assert len(rows) == 1
    rendered = "\n".join(
        str(value) for value in (rows, pd.DataFrame(rows), repr(rows))
    ).lower()
    for secret in ("secret_segment_d1b2", "secret_cohort_d1b2"):
        assert secret not in rendered
    for forbidden in (
        "winner",
        "best",
        "recommended",
        "champion",
        "deployed",
    ):
        assert forbidden not in rendered


def test_d1b2_actual_grouped_comparison_exposes_only_anonymous_positions() -> None:
    frame = _scoped_summary_frame()
    frame.loc[frame.index[0], "segment"] = "SECRET_SEGMENT_D1B2_ACTUAL"
    config = replace(
        _d1b2_comparison_config(),
        segment_columns=("segment",),
        cohort_column="cohort",
        cohort_time_column="vintage_origin",
    )
    result = monitor_lifecycle(frame, config)
    comparison = result.scenario_comparison
    grouped = comparison.loc[comparison["scope_key"] == "scenario_segment"]
    assert not grouped.empty
    assert grouped["scope_position"].notna().all()
    rendered = "\n".join(
        str(value)
        for value in (
            comparison,
            result.monitoring_summary,
            result.lifecycle_summary,
            result.provenance,
            repr(result),
        )
    )
    assert "SECRET_SEGMENT_D1B2_ACTUAL" not in rendered


def test_d1b2_comparison_projection_never_mutates_source_summary() -> None:
    result = _d1b2_comparison_result()
    before = result.monitoring_summary.copy(deep=True)
    lifecycle_monitoring._scenario_comparison_rows(
        _d1b2_comparison_config(), result.monitoring_summary
    )
    pd.testing.assert_frame_equal(result.monitoring_summary, before)


def test_d1b2_comparison_is_deterministic_and_uses_frozen_pair_scope_rule_order() -> (
    None
):
    config = _d1b2_comparison_config()
    later = WarningScenario("later", "rule_set", config.scenarios[1].rules)
    three_scenarios = replace(
        config,
        scenarios=(config.scenarios[0], config.scenarios[1], later),
    )
    source_rows: list[dict[str, object]] = []
    for scenario_position, scenario in enumerate(three_scenarios.scenarios):
        for position, scope_key in enumerate(
            (
                "scenario",
                "scenario_alert_level",
                "scenario_segment",
                "scenario_time",
                "scenario_cohort",
                "scenario_vintage",
                "scenario_state",
                "scenario_transition",
            )
        ):
            source_rows.append(
                _d1b2_summary_row(
                    scenario.scenario_key,
                    scope_key=scope_key,
                    scope_position=(
                        scenario_position if scope_key == "scenario" else position
                    ),
                )
            )
        for rule in ("negative", "zero", "positive"):
            source_rows.append(
                _d1b2_summary_row(
                    scenario.scenario_key,
                    scope_key="scenario_rule",
                    scope_position=scenario_position,
                    rule_key=rule,
                )
            )
    summary = pd.DataFrame(source_rows)
    first = lifecycle_monitoring._scenario_comparison_rows(three_scenarios, summary)
    second = lifecycle_monitoring._scenario_comparison_rows(three_scenarios, summary)
    pd.testing.assert_frame_equal(pd.DataFrame(first), pd.DataFrame(second))
    comparison = pd.DataFrame(first)
    assert comparison["comparator_scenario_key"].drop_duplicates().tolist() == [
        "challenger",
        "later",
    ]
    warning_rows = comparison.loc[
        (comparison["comparator_scenario_key"] == "challenger")
        & (comparison["metric"] == "warning_hit_count")
    ]
    assert warning_rows["scope_key"].drop_duplicates().tolist() == [
        "scenario",
        "scenario_rule",
        "scenario_alert_level",
        "scenario_segment",
        "scenario_time",
        "scenario_cohort",
        "scenario_vintage",
        "scenario_state",
        "scenario_transition",
    ]
    assert warning_rows.loc[
        warning_rows["scope_key"] == "scenario_rule", "rule_key"
    ].tolist() == ["negative", "zero", "positive"]


def _d1b2_summary_projections(**overrides: int) -> dict[str, int]:
    projections = {
        "monitoring_summary_rows": 200_000,
        "lifecycle_summary_rows": 200_000,
        "scenario_comparison_rows": 200_000,
    }
    projections.update(overrides)
    return projections


def test_d1b2_summary_projection_gates_accept_exact_maximums() -> None:
    lifecycle_monitoring._summary_projection_gates(_d1b2_summary_projections())


@pytest.mark.parametrize(
    "key",
    (
        "monitoring_summary_rows",
        "lifecycle_summary_rows",
        "scenario_comparison_rows",
    ),
)
def test_d1b2_summary_projection_gates_reject_each_maximum_plus_one(
    key: str,
) -> None:
    with pytest.raises(ValueError, match=rf"lifecycle resource limit exceeded: {key}"):
        lifecycle_monitoring._summary_projection_gates(
            _d1b2_summary_projections(**{key: 200_001})
        )


def test_d1b2_summary_projection_gate_precedence_is_frozen() -> None:
    with pytest.raises(
        ValueError,
        match="lifecycle resource limit exceeded: monitoring_summary_rows",
    ):
        lifecycle_monitoring._summary_projection_gates(
            _d1b2_summary_projections(
                monitoring_summary_rows=200_001,
                lifecycle_summary_rows=200_001,
                scenario_comparison_rows=200_001,
            )
        )
    with pytest.raises(
        ValueError,
        match="lifecycle resource limit exceeded: lifecycle_summary_rows",
    ):
        lifecycle_monitoring._summary_projection_gates(
            _d1b2_summary_projections(
                lifecycle_summary_rows=200_001,
                scenario_comparison_rows=200_001,
            )
        )
    with pytest.raises(
        ValueError,
        match="lifecycle resource limit exceeded: scenario_comparison_rows",
    ):
        lifecycle_monitoring._summary_projection_gates(
            _d1b2_summary_projections(scenario_comparison_rows=200_001)
        )


def test_d1b2_summary_preflight_precedes_source_alignment_without_raw_leakage() -> (
    None
):
    """A real over-limit projection stops before source alignment/materialization."""
    row_count = 400
    observed = [
        datetime(2000 + (position % 240) // 12, (position % 12) + 1, 1)
        for position in range(row_count)
    ]
    frame = pd.DataFrame(
        {
            "entity": [f"entity-{position}" for position in range(row_count)],
            "observed": observed,
            "available": observed,
            "feature": [1] * row_count,
            "other": [0] * row_count,
            "origin": [
                value - timedelta(days=position % 240)
                for position, value in enumerate(observed)
            ],
        }
    )
    for column in range(4):
        frame[f"segment_{column}"] = [
            f"SECRET_SEGMENT_{column}_{position}"
            if position < 99
            else f"SECRET_SEGMENT_{column}_99"
            for position in range(row_count)
        ]
    rules = tuple(
        EarlyWarningRule(f"r{position}", position, "high", _condition())
        for position in range(10)
    )
    scenarios = tuple(
        WarningScenario(
            "reference" if position == 0 else f"scenario-{position}",
            "rule_set",
            rules,
        )
        for position in range(10)
    )
    config = replace(
        _config(),
        analysis_as_of=datetime(2025, 1, 1),
        scenarios=scenarios,
        segment_columns=tuple(f"segment_{column}" for column in range(4)),
        cohort_time_column="origin",
        time_frequency="month",
    )
    with pytest.raises(
        ValueError,
        match="lifecycle resource limit exceeded: monitoring_summary_rows",
    ) as raised:
        monitor_lifecycle(frame, config, risk_validation=object())
    assert "SECRET_SEGMENT" not in str(raised.value)


@pytest.mark.parametrize(
    ("bucket", "expected_key"),
    (("time", "time_buckets"), ("cohort", "cohort_buckets")),
)
def test_d1b2_scope_bucket_gates_accept_maximum_and_reject_plus_one(
    bucket: str,
    expected_key: str,
) -> None:
    def facts_for(row_count: int):
        dates = [
            datetime(2024, 1, 1) + timedelta(days=position)
            for position in range(row_count)
        ]
        frame = pd.DataFrame(
            {
                "entity": [f"entity-{position}" for position in range(row_count)],
                "observed": dates,
                "available": dates,
                "feature": [1] * row_count,
                "other": [0] * row_count,
            }
        )
        changes: dict[str, object] = {
            "analysis_as_of": datetime(2026, 1, 1),
            "time_frequency": "day",
        }
        if bucket == "cohort":
            frame["cohort"] = [f"cohort-{position}" for position in range(row_count)]
            changes["cohort_column"] = "cohort"
        config = replace(_config(), **changes)
        entities, aware = lifecycle_monitoring._validate_data(frame, config)
        return lifecycle_monitoring._scope_facts(frame, config, entities, aware)

    maximum = facts_for(240)
    if bucket == "time":
        assert len(maximum.time_groups) == 240
    else:
        assert len(maximum.cohort_groups) == 240
    with pytest.raises(
        ValueError,
        match=rf"lifecycle resource limit exceeded: {expected_key}",
    ):
        facts_for(241)


def _scenario_shape_config(
    scenario_kind: str, rules: tuple[EarlyWarningRule, ...]
) -> LifecycleMonitoringConfig:
    return replace(
        _config(),
        scenarios=(WarningScenario("reference", scenario_kind, rules),),
    )


def _score_rule(key: str = "score") -> EarlyWarningRule:
    return EarlyWarningRule(
        key,
        0,
        "high",
        MonitoringCondition("atomic", "gt", "ranking_score", None, "literal", 0.5),
    )


def test_ir01_valid_no_alert_shape() -> None:
    assert monitor_lifecycle(
        _frame(), _scenario_shape_config("no_alert", ())
    ).requested_scenario_count == 1


@pytest.mark.parametrize(
    "rules",
    [(_score_rule(),), (_score_rule("a"), _score_rule("b"))],
)
def test_ir01_no_alert_rejects_rules(rules: tuple[EarlyWarningRule, ...]) -> None:
    with pytest.raises(ValueError, match="invalid_scenario_shape"):
        monitor_lifecycle(_frame(), _scenario_shape_config("no_alert", rules))


def test_ir01_single_threshold_requires_one_score_rule() -> None:
    valid = _scenario_shape_config("single_threshold", (_score_rule(),))
    assert monitor_lifecycle(_frame().assign(score=[0.8, 0.2]), replace(
        valid,
        ranking_score_column="score",
        ranking_score_direction="higher_risk",
    )).requested_scenario_count == 1
    ordinary = EarlyWarningRule("ordinary", 0, "high", _condition())
    with pytest.raises(ValueError, match="invalid_scenario_shape"):
        monitor_lifecycle(
            _frame(), _scenario_shape_config("single_threshold", (ordinary,))
        )


@pytest.mark.parametrize("rules", [(), (_score_rule("a"), _score_rule("b"))])
def test_ir01_single_threshold_requires_exactly_one_rule(
    rules: tuple[EarlyWarningRule, ...],
) -> None:
    with pytest.raises(ValueError, match="invalid_scenario_shape"):
        monitor_lifecycle(_frame(), _scenario_shape_config("single_threshold", rules))


def test_ir01_rule_set_and_model_score_shapes() -> None:
    assert monitor_lifecycle(
        _frame(),
        _scenario_shape_config(
            "rule_set", (EarlyWarningRule("r", 0, "high", _condition()),)
        ),
    ).requested_scenario_count == 1
    with pytest.raises(ValueError, match="invalid_scenario_shape"):
        monitor_lifecycle(_frame(), _scenario_shape_config("rule_set", ()))
    assert monitor_lifecycle(
        _frame().assign(score=[0.8, 0.2]),
        replace(
            _scenario_shape_config("model_score", (_score_rule(),)),
            ranking_score_column="score",
            ranking_score_direction="higher_risk",
        ),
    ).requested_scenario_count == 1
    with pytest.raises(ValueError, match="invalid_scenario_shape"):
        monitor_lifecycle(_frame(), _scenario_shape_config("model_score", ()))


def test_ir01_model_plus_rules_requires_both_families() -> None:
    score = _score_rule()
    ordinary = EarlyWarningRule("ordinary", 1, "high", _condition())
    config = replace(
        _scenario_shape_config("model_plus_rules", (score, ordinary)),
        ranking_score_column="score",
        ranking_score_direction="higher_risk",
    )
    assert (
        monitor_lifecycle(_frame().assign(score=[0.8, 0.2]), config)
        .requested_scenario_count
        == 1
    )
    with pytest.raises(ValueError, match="invalid_scenario_shape"):
        monitor_lifecycle(
            _frame(), _scenario_shape_config("model_plus_rules", (score,))
        )


def test_ir01_unknown_kind_is_stable_and_precedes_data_processing() -> None:
    with pytest.raises(
        ValueError, match="lifecycle config is invalid: invalid_scenario_shape"
    ):
        monitor_lifecycle("not-a-dataframe", _scenario_shape_config("future_kind", ()))


@pytest.mark.parametrize(
    "mapping",
    [(("low", 1),), (("high", -1),), (("high", 1.0),), (("bad level", 1),)],
)
def test_ir02_alert_mapping_is_closed_and_exact(mapping: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="invalid_alert_level_mapping"):
        monitor_lifecycle(_frame(), replace(_config(), alert_level_ranks=mapping))


def test_ir02_missing_alert_mapping_is_rejected_before_row_evaluation() -> None:
    rule = EarlyWarningRule("r", 0, "high", _condition())
    config = replace(
        _config(),
        alert_level_ranks=(("low", 1),),
        scenarios=(WarningScenario("reference", "rule_set", (rule,)),),
    )
    with pytest.raises(ValueError, match="invalid_alert_level_mapping") as raised:
        monitor_lifecycle("not-a-dataframe", config)
    assert not isinstance(raised.value, KeyError)


def test_ir02_rule_boundaries_validate_order_and_exact_types() -> None:
    config = replace(
        _config(),
        scenarios=(
            WarningScenario(
                "reference",
                "rule_set",
                (
                    replace(
                        EarlyWarningRule("r", 0, "high", _condition()),
                        effective_from=datetime(2025, 1, 3),
                        expires_at=datetime(2025, 1, 3),
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="invalid_rule"):
        monitor_lifecycle(_frame(), config)
    bad_type = replace(
        _config(),
        scenarios=(
            WarningScenario(
                "reference",
                "rule_set",
                (
                    replace(
                        EarlyWarningRule("r", 0, "high", _condition()),
                        effective_from=date(2025, 1, 1),
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="invalid_rule"):
        monitor_lifecycle(_frame(), bad_type)


def test_ir02_datetime_awareness_mismatch_is_stable() -> None:
    aware = timezone(timedelta(hours=8))
    rule = replace(
        EarlyWarningRule("r", 0, "high", _condition()),
        effective_from=datetime(2025, 1, 1, tzinfo=aware),
    )
    config = replace(
        _config(),
        analysis_as_of=datetime(2025, 1, 5, tzinfo=aware),
        scenarios=(WarningScenario("reference", "rule_set", (rule,)),),
    )
    with pytest.raises(ValueError, match="datetime_awareness_mismatch"):
        monitor_lifecycle(_frame(), config)


def test_ir02_time_zone_declaration_requires_aware_model() -> None:
    with pytest.raises(ValueError, match="invalid_time_window"):
        monitor_lifecycle(_frame(), replace(_config(), time_zone="UTC"))


def test_ir02_malicious_scalar_is_rejected_without_protocol_callbacks() -> None:
    _Malicious.callbacks = 0
    with pytest.raises(ValueError, match="invalid_alert_level_mapping"):
        monitor_lifecycle(
            _frame(), replace(_config(), alert_level_ranks=(("high", _Malicious()),))
        )
    assert _Malicious.callbacks == 0
