"""Task 17 decision-strategy contract tests."""

from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime
from inspect import signature

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from sharper import (
    BinaryRiskValidationConfig,
    DecisionConstraint,
    DecisionRule,
    DecisionStrategyConfig,
    DecisionStrategyResult,
    ExternalRiskPredictions,
    StrategyCondition,
    simulate_decision_strategy,
    validate_binary_risk,
)
from sharper._condition_kernel import (
    _ConditionNode,
    _ConditionOperand,
    _evaluate_atomic_condition,
    _evaluate_condition,
)

NOW = datetime(2025, 1, 2)
START = datetime(2025, 1, 1)


def _condition(column: str = "x", value: object = 2) -> StrategyCondition:
    return StrategyCondition("atomic", "ge", "column", column, "literal", value)


def _config(
    *rules: DecisionRule,
    constraints: tuple[DecisionConstraint, ...] = (),
    **kwargs: object,
) -> DecisionStrategyConfig:
    action_roles = {
        "select": "selected",
        "review": "review",
        "reject": "rejected",
    }
    action_names = ["select", "review"]
    action_names.extend(
        rule.action_name for rule in rules if rule.action_name not in action_names
    )
    values = {
        "strategy_key": "s1",
        "strategy_version": "v1",
        "effective_from": START,
        "expires_at": None,
        "evaluation_time": NOW,
        "rules": rules,
        "default_action_name": "select",
        "unknown_action_name": "review",
        "action_role_mapping": tuple(
            (action, action_roles[action]) for action in action_names
        ),
        "constraints": constraints,
    }
    values.update(kwargs)
    return DecisionStrategyConfig(**values)  # type: ignore[arg-type]


def _resource_config(
    n_rules: int,
    n_actions: int,
    *,
    segment_columns: tuple[str, ...] = (),
) -> DecisionStrategyConfig:
    """Build a public config for independently calculated resource-boundary tests."""
    actions = ("select", "review") + tuple(
        f"action{index}" for index in range(n_actions - 2)
    )
    decision_count = min(n_rules, 50)
    rules = tuple(
        DecisionRule(
            f"d{index}",
            "decision",
            index,
            _condition(segment_columns[0] if segment_columns else "x", 0),
            actions[index % n_actions],
        )
        for index in range(decision_count)
    ) + tuple(
        DecisionRule(
            f"e{index}",
            "eligibility",
            index,
            _condition(segment_columns[0] if segment_columns else "x", 0),
            "review",
        )
        for index in range(n_rules - decision_count)
    )
    return DecisionStrategyConfig(
        "resource",
        "v1",
        START,
        None,
        NOW,
        rules,
        "select",
        "review",
        tuple(
            (action, "selected" if action == "select" else "review")
            for action in actions
        ),
        segment_columns=segment_columns,
    )


def _risk_result(frame: pd.DataFrame, *, business: bool = False):
    labels = np.asarray(frame["target"])
    positions = np.arange(len(frame))
    fold_by_position: dict[int, int] = {}
    fit_rows = []
    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    for fold_id, (train, validation) in enumerate(splitter.split(positions, labels)):
        fit_rows.append((fold_id, tuple(sorted(int(value) for value in train))))
        for value in validation:
            fold_by_position[int(value)] = fold_id
    probabilities = tuple(np.linspace(0.1, 0.9, len(frame)))
    external = ExternalRiskPredictions(
        tuple(range(len(frame))),
        tuple(fold_by_position[position] for position in range(len(frame))),
        tuple(fit_rows),
        probabilities,
        "higher_risk",
        probabilities,
        1,
        "external_declared",
    )
    risk_config = BinaryRiskValidationConfig(
        validation_mode="stratified_kfold", n_splits=2
    )
    if business:
        risk_config = replace(
            risk_config,
            exposure_column="exposure",
            observed_loss_column="observed_loss",
            analysis_as_of=NOW,
            observed_loss_is_mature_snapshot=True,
            loss_fraction=0.5,
            exposure_unit="units",
        )
    return validate_binary_risk(
        frame,
        "target",
        config=risk_config,
        external_predictions=external,
    )


def _as_time_forward(risk):
    folds = risk.folds.copy(deep=True)
    for row in range(len(folds)):
        validation_n = int(folds.at[row, "validation_n"])
        evaluable_n = int(folds.at[row, "evaluable_validation_n"])
        folds.at[row, "validation_mature_n"] = evaluable_n
        folds.at[row, "validation_excluded_n"] = validation_n - evaluable_n
        folds.at[row, "immature_validation_n"] = validation_n - evaluable_n
    return replace(risk, validation_mode="time_forward", folds=folds)


def _with_immature_positions(risk, positions: tuple[int, ...]):
    predictions = risk.predictions.copy(deep=True)
    for position in positions:
        predictions.at[position, "is_evaluable"] = False
        predictions.at[position, "target_value"] = pd.NA
    folds = risk.folds.copy(deep=True)
    for row in range(len(folds)):
        validation = folds.at[row, "validation_row_positions"]
        evaluable = tuple(value for value in validation if value not in positions)
        validation_n = len(validation)
        evaluable_n = len(evaluable)
        folds.at[row, "evaluable_validation_row_positions"] = evaluable
        folds.at[row, "evaluable_validation_n"] = evaluable_n
        folds.at[row, "validation_mature_n"] = evaluable_n
        folds.at[row, "validation_excluded_n"] = validation_n - evaluable_n
        folds.at[row, "immature_validation_n"] = validation_n - evaluable_n
    return replace(
        risk,
        validation_mode="time_forward",
        folds=folds,
        predictions=predictions,
        evaluable_n_rows=len(predictions) - len(positions),
    )


class _ProtocolBomb:
    calls: list[str] = []

    @classmethod
    def reset(cls) -> None:
        cls.calls = []

    def _called(self, name: str):
        type(self).calls.append(name)
        raise AssertionError(f"unexpected protocol dispatch: {name}")

    def __array__(self, *args, **kwargs):
        return self._called("__array__")

    def __float__(self):
        return self._called("__float__")

    def __eq__(self, other):
        return self._called("__eq__")

    def __lt__(self, other):
        return self._called("__lt__")

    def __iter__(self):
        return self._called("__iter__")

    def __str__(self):
        return self._called("__str__")

    def __repr__(self):
        return self._called("__repr__")

    def __hash__(self):
        return self._called("__hash__")


def test_public_fields_signature_frozen_and_eight_typed_tables() -> None:
    expected = {
        StrategyCondition: [
            "kind",
            "operator",
            "left_kind",
            "left",
            "right_kind",
            "right",
            "children",
        ],
        DecisionRule: [
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
        ],
        DecisionConstraint: [
            "constraint_key",
            "metric",
            "operator",
            "threshold",
            "action_name",
            "action_role",
            "minimum_support",
        ],
        DecisionStrategyConfig: [
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
        ],
        DecisionStrategyResult: [
            "strategy_key",
            "strategy_version",
            "strategy_fingerprint",
            "input_n_rows",
            "decided_n_rows",
            "unavailable_n_rows",
            "requested_rule_count",
            "active_rule_count",
            "requested_constraint_count",
            "row_decisions",
            "rule_evaluations",
            "rule_summary",
            "action_summary",
            "business_summary",
            "constraint_summary",
            "historical_transitions",
            "provenance",
            "warnings",
            "limitations",
        ],
    }
    for data_type, names in expected.items():
        assert data_type.__dataclass_params__.frozen
        assert [field.name for field in fields(data_type)] == names
    with pytest.raises(FrozenInstanceError):
        _condition().kind = "and"  # type: ignore[misc]
    assert str(signature(simulate_decision_strategy)) == (
        "(data: 'pd.DataFrame', config: 'DecisionStrategyConfig', *, "
        "risk_validation: 'BinaryRiskValidationResult | None' = None, "
        "data_audit: 'DataAuditResult | None' = None) -> 'DecisionStrategyResult'"
    )
    result = simulate_decision_strategy(pd.DataFrame({"x": []}), _config())
    assert all(
        isinstance(getattr(result, name), pd.DataFrame)
        for name in (
            "row_decisions",
            "rule_evaluations",
            "rule_summary",
            "action_summary",
            "business_summary",
            "constraint_summary",
            "historical_transitions",
            "provenance",
        )
    )
    assert str(result.row_decisions["row_position"].dtype) == "int64"


def test_decision_default_unknown_and_input_immutability() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, np.nan]}, index=[7, 7, 8])
    before = frame.copy(deep=True)
    rule = DecisionRule("choose", "decision", 1, _condition(), "select")
    first = simulate_decision_strategy(frame, _config(rule))
    second = simulate_decision_strategy(frame, _config(rule))
    assert first.row_decisions["final_action_name"].tolist() == [
        "select",
        "select",
        "review",
    ]
    assert first.row_decisions["decision_reason"].tolist() == [
        "default_action_applied",
        "computed",
        "unknown_condition",
    ]
    assert first.decided_n_rows == 3
    assert first.unavailable_n_rows == 0
    pd.testing.assert_frame_equal(first.row_decisions, second.row_decisions)
    assert first.strategy_fingerprint == second.strategy_fingerprint
    pd.testing.assert_frame_equal(frame, before)


def test_eligibility_terminal_roles_and_unknown_safety() -> None:
    eligibility = DecisionRule("hard", "eligibility", 1, _condition(value=2), "reject")
    decision = DecisionRule("choose", "decision", 1, _condition(value=1), "select")
    result = simulate_decision_strategy(
        pd.DataFrame({"x": [1.0, 2.0, np.nan]}), _config(eligibility, decision)
    )
    assert result.row_decisions["final_action_name"].tolist() == [
        "select",
        "reject",
        "review",
    ]
    invalid = replace(eligibility, action_name="select")
    with pytest.raises(ValueError, match="eligibility_action_role"):
        simulate_decision_strategy(pd.DataFrame({"x": [1]}), _config(invalid))
    with pytest.raises(ValueError, match="unknown_action_role"):
        simulate_decision_strategy(
            pd.DataFrame({"x": [1]}),
            replace(_config(), unknown_action_name="select"),
        )


def test_overlap_conflict_priority_and_inactive_strategy() -> None:
    rules = (
        DecisionRule("a", "decision", 1, _condition(value=1), "select", False),
        DecisionRule("b", "decision", 2, _condition(value=1), "reject"),
    )
    result = simulate_decision_strategy(pd.DataFrame({"x": [2]}), _config(*rules))
    assert result.row_decisions.loc[0, "final_action_name"] == "review"
    assert result.row_decisions.loc[0, "conflict_rule_count"] == 1
    assert result.rule_evaluations["is_overlap"].tolist() == [False, True]
    with pytest.raises(ValueError, match="duplicate_rule_priority"):
        simulate_decision_strategy(
            pd.DataFrame({"x": [1]}),
            _config(rules[0], replace(rules[1], priority=1)),
        )
    inactive = replace(_config(), evaluation_time=datetime(2024, 1, 1))
    result = simulate_decision_strategy(pd.DataFrame({"x": [1, 2]}), inactive)
    assert result.decided_n_rows == 0
    assert result.unavailable_n_rows == 2
    assert result.row_decisions["decision_status"].tolist() == ["inactive"] * 2


def test_task15_probability_alignment_and_raw_label_privacy() -> None:
    frame = pd.DataFrame({"x": range(8), "target": [0, 1] * 4})
    risk = _risk_result(frame)
    probability_condition = StrategyCondition(
        "atomic", "ge", "event_probability", None, "literal", 0.5
    )
    result = simulate_decision_strategy(
        frame,
        _config(DecisionRule("p", "decision", 1, probability_condition, "reject")),
        risk_validation=risk,
    )
    assert result.row_decisions["final_action_name"].tolist()[:4] == [
        "select",
        "select",
        "select",
        "select",
    ]
    assert "positive_label_type_family" in result.provenance["provenance_key"].tolist()
    assert "positive_label" not in result.provenance["provenance_key"].tolist()
    broken = replace(
        risk,
        predictions=risk.predictions.assign(
            fold_id=lambda table: table["fold_id"].mask(table.index == 0, 99)
        ),
    )
    with pytest.raises(ValueError, match="^strategy source alignment:"):
        simulate_decision_strategy(frame, _config(), risk_validation=broken)


def test_t17_a1_non_time_zero_maturity_with_evaluable_probability_passes() -> None:
    frame = pd.DataFrame({"x": range(8), "target": [0, 1] * 4}, index=[5] * 8)
    risk = _risk_result(frame)
    assert set(risk.folds["validation_mature_n"]) == {0}
    assert all(risk.folds["evaluable_validation_n"] > 0)
    result = simulate_decision_strategy(frame, _config(), risk_validation=risk)
    probability = result.business_summary.loc[
        (result.business_summary["scope_type"] == "overall")
        & result.business_summary["action_role"].isna()
        & (result.business_summary["metric_key"] == "event_probability_mean")
    ]
    assert set(probability["status"]) == {"available"}


def test_t17_a1_time_forward_maturity_alignment_passes() -> None:
    frame = pd.DataFrame({"x": range(8), "target": [0, 1] * 4})
    risk = _as_time_forward(_risk_result(frame))
    result = simulate_decision_strategy(frame, _config(), risk_validation=risk)
    assert result.decided_n_rows == len(frame)
    assert set(
        result.business_summary.loc[
            (result.business_summary["scope_type"] == "overall")
            & result.business_summary["action_role"].isna()
            & (result.business_summary["metric_key"] == "event_probability_mean"),
            "status",
        ]
    ) == {"available"}


@pytest.mark.parametrize(
    ("mode", "column", "value", "key"),
    [
        (
            "stratified_kfold",
            "validation_mature_n",
            1,
            "non_time_maturity_count_nonzero",
        ),
        (
            "time_forward",
            "validation_mature_n",
            0,
            "time_mode_maturity_mismatch",
        ),
        (
            "time_forward",
            "immature_validation_n",
            1,
            "time_mode_maturity_mismatch",
        ),
        (
            "stratified_kfold",
            "validation_n",
            99,
            "maturity_count_mismatch",
        ),
    ],
)
def test_t17_a1_mode_dependent_maturity_failures(
    mode: str, column: str, value: int, key: str
) -> None:
    frame = pd.DataFrame({"x": range(8), "target": [0, 1] * 4})
    risk = _risk_result(frame)
    if mode == "time_forward":
        risk = _as_time_forward(risk)
    folds = risk.folds.copy(deep=True)
    folds.at[0, column] = value
    broken = replace(risk, validation_mode=mode, folds=folds)
    with pytest.raises(ValueError, match=rf"^strategy source alignment: {key}$"):
        simulate_decision_strategy(frame, _config(), risk_validation=broken)


@pytest.mark.parametrize(
    ("mutation", "key"),
    [
        ("fold_union", "prediction_fold_union"),
        ("fold_id", "prediction_fold_id"),
        ("is_evaluable", "evaluable_fold_union"),
        ("duplicate_prediction", "prediction_positions"),
        ("predicted_excluded_overlap", "predicted_excluded_overlap"),
    ],
)
def test_task15_common_source_alignment_failures(mutation: str, key: str) -> None:
    frame = pd.DataFrame({"x": range(8), "target": [0, 1] * 4})
    risk = _risk_result(frame)
    if mutation == "fold_union":
        folds = risk.folds.copy(deep=True)
        positions = folds.at[0, "validation_row_positions"]
        evaluable = folds.at[0, "evaluable_validation_row_positions"]
        folds.at[0, "validation_row_positions"] = positions[1:]
        folds.at[0, "evaluable_validation_row_positions"] = evaluable[1:]
        folds.at[0, "validation_n"] = len(positions) - 1
        folds.at[0, "evaluable_validation_n"] = len(evaluable) - 1
        broken = replace(risk, folds=folds)
    elif mutation == "fold_id":
        predictions = risk.predictions.copy(deep=True)
        predictions.at[0, "fold_id"] = 99
        broken = replace(risk, predictions=predictions)
    elif mutation == "is_evaluable":
        predictions = risk.predictions.copy(deep=True)
        predictions.at[0, "is_evaluable"] = False
        broken = replace(
            risk, predictions=predictions, evaluable_n_rows=risk.evaluable_n_rows - 1
        )
    elif mutation == "duplicate_prediction":
        predictions = risk.predictions.copy(deep=True)
        predictions.at[1, "row_position"] = predictions.at[0, "row_position"]
        broken = replace(risk, predictions=predictions)
    else:
        excluded = pd.DataFrame({"row_position": [0], "reason": ["training_only"]})
        broken = replace(risk, excluded_rows=excluded)
    with pytest.raises(ValueError, match=rf"^strategy source alignment: {key}$"):
        simulate_decision_strategy(frame, _config(), risk_validation=broken)


@pytest.mark.parametrize(
    ("column", "key"),
    [("event_probability", "event_probability"), ("ranking_score", "ranking_score")],
)
def test_task15_numeric_cells_reject_protocol_objects_without_dispatch(
    column: str, key: str
) -> None:
    frame = pd.DataFrame({"x": range(8), "target": [0, 1] * 4})
    risk = _risk_result(frame)
    predictions = risk.predictions.copy(deep=True)
    predictions[column] = predictions[column].astype(object)
    _ProtocolBomb.reset()
    predictions.at[0, column] = _ProtocolBomb()
    with pytest.raises(ValueError, match=rf"^strategy source alignment: {key}$"):
        simulate_decision_strategy(
            frame, _config(), risk_validation=replace(risk, predictions=predictions)
        )
    assert _ProtocolBomb.calls == []


@pytest.mark.parametrize(
    ("metric", "column"),
    (
        ("observed_loss_sum", "value"),
        ("observed_loss_sum", "n_observed_loss_mature_rows"),
        ("observed_loss_sum", "unit"),
        ("exposure_sum", "value"),
        ("exposure_sum", "n_observed_loss_mature_rows"),
        ("exposure_sum", "unit"),
    ),
)
def test_task15_aggregate_cells_reject_protocol_objects_without_dispatch(
    metric: str, column: str
) -> None:
    frame = pd.DataFrame(
        {
            "target": [0, 1] * 4,
            "exposure": np.arange(1.0, 9.0),
            "observed_loss": np.arange(0.0, 8.0),
        }
    )
    risk = _risk_result(frame, business=True)
    table = risk.business_metrics.copy(deep=True)
    table[column] = table[column].astype(object)
    row = table.index[(table["segment_kind"] == "all") & (table["metric"] == metric)][0]
    _ProtocolBomb.reset()
    table.at[row, column] = _ProtocolBomb()
    with pytest.raises(
        ValueError,
        match="^strategy source alignment: observed_loss_evidence$",
    ):
        simulate_decision_strategy(
            frame,
            _config(),
            risk_validation=replace(risk, business_metrics=table),
        )
    assert _ProtocolBomb.calls == []


@pytest.mark.parametrize(
    ("metric", "column"),
    (
        ("observed_loss_sum", "value"),
        ("observed_loss_sum", "n_observed_loss_mature_rows"),
        ("exposure_sum", "value"),
        ("exposure_sum", "n_observed_loss_mature_rows"),
    ),
)
def test_task15_aggregate_rejects_bool_numeric_and_support(
    metric: str, column: str
) -> None:
    frame = pd.DataFrame(
        {
            "target": [0, 1] * 4,
            "exposure": np.arange(1.0, 9.0),
            "observed_loss": np.arange(0.0, 8.0),
        }
    )
    risk = _risk_result(frame, business=True)
    table = risk.business_metrics.copy(deep=True)
    table[column] = table[column].astype(object)
    row = table.index[(table["segment_kind"] == "all") & (table["metric"] == metric)][0]
    table.at[row, column] = True
    with pytest.raises(
        ValueError,
        match="^strategy source alignment: observed_loss_evidence$",
    ):
        simulate_decision_strategy(
            frame,
            _config(),
            risk_validation=replace(risk, business_metrics=table),
        )


def test_task15_aggregate_rejects_unapproved_numpy_scalar() -> None:
    frame = pd.DataFrame(
        {
            "target": [0, 1] * 4,
            "exposure": np.arange(1.0, 9.0),
            "observed_loss": np.arange(0.0, 8.0),
        }
    )
    risk = _risk_result(frame, business=True)
    table = risk.business_metrics.copy(deep=True)
    table["value"] = table["value"].astype(object)
    row = table.index[
        (table["segment_kind"] == "all") & (table["metric"] == "observed_loss_sum")
    ][0]
    table.at[row, "value"] = np.longdouble("1.0")
    with pytest.raises(
        ValueError,
        match="^strategy source alignment: observed_loss_evidence$",
    ):
        simulate_decision_strategy(
            frame,
            _config(),
            risk_validation=replace(risk, business_metrics=table),
        )


def test_task15_common_alignment_precedes_mode_maturity_failures() -> None:
    frame = pd.DataFrame({"x": range(8), "target": [0, 1] * 4})
    risk = _risk_result(frame)
    folds = risk.folds.copy(deep=True)
    validation = folds.at[0, "validation_row_positions"]
    evaluable = folds.at[0, "evaluable_validation_row_positions"]
    folds.at[0, "validation_row_positions"] = validation[1:]
    folds.at[0, "evaluable_validation_row_positions"] = evaluable[1:]
    folds.at[0, "validation_n"] = len(validation) - 1
    folds.at[0, "evaluable_validation_n"] = len(evaluable) - 1
    folds.at[0, "validation_mature_n"] = 1
    with pytest.raises(
        ValueError, match="^strategy source alignment: prediction_fold_union$"
    ):
        simulate_decision_strategy(
            frame, _config(), risk_validation=replace(risk, folds=folds)
        )

    time_risk = _as_time_forward(risk)
    folds = time_risk.folds.copy(deep=True)
    folds.at[0, "validation_n"] = 99
    folds.at[0, "validation_mature_n"] = 0
    with pytest.raises(
        ValueError, match="^strategy source alignment: maturity_count_mismatch$"
    ):
        simulate_decision_strategy(
            frame, _config(), risk_validation=replace(time_risk, folds=folds)
        )


def test_dataframe_ranking_never_creates_probability() -> None:
    config = replace(
        _config(
            DecisionRule(
                "score",
                "decision",
                1,
                StrategyCondition(
                    "atomic", "ge", "ranking_score", None, "literal", 0.5
                ),
                "reject",
            )
        ),
        ranking_score_column="score",
        ranking_score_direction="higher_risk",
    )
    result = simulate_decision_strategy(
        pd.DataFrame({"x": [1, 2], "score": [0.2, 0.8]}), config
    )
    probability = result.business_summary.loc[
        result.business_summary["metric_key"] == "event_probability_mean"
    ]
    assert set(probability["status"]) == {"not_applicable"}
    assert set(probability["reason"]) == {"source_not_requested"}


def test_no_task15_outcome_metrics_are_not_applicable_not_available_zero() -> None:
    result = simulate_decision_strategy(pd.DataFrame({"x": [1, 2]}), _config())
    action = result.action_summary.loc[
        result.action_summary["metric_key"].isin(
            ["evaluable_event_count", "event_count", "event_rate"]
        )
    ]
    assert set(action["status"]) == {"not_applicable"}
    assert set(action["reason"]) == {"source_not_requested"}
    assert action["metric_value"].isna().all()
    business = result.business_summary.loc[
        result.business_summary["metric_key"].isin(
            ["expected_loss_sum", "assumption_based_observed_event_loss_sum"]
        )
    ]
    assert set(business["status"]) == {"not_applicable"}
    assert set(business["reason"]) == {"source_not_requested"}
    assert business[["metric_value", "numerator", "denominator"]].isna().all().all()
    assert set(business["support_n_rows"]) == {0}


def test_ranking_only_and_incomplete_expected_loss_source_precedence() -> None:
    frame = pd.DataFrame(
        {"x": range(8), "target": [0, 1] * 4, "exposure": np.arange(1.0, 9.0)}
    )
    risk = _risk_result(frame)
    predictions = risk.predictions.copy(deep=True)
    predictions["event_probability"] = pd.array([pd.NA] * len(frame), dtype="Float64")
    ranking_only = replace(risk, predictions=predictions, probability_provenance=None)
    config = replace(
        _config(), exposure_column="exposure", loss_fraction=0.5, exposure_unit="units"
    )
    result = simulate_decision_strategy(frame, config, risk_validation=ranking_only)
    expected = result.business_summary.loc[
        result.business_summary["metric_key"] == "expected_loss_sum"
    ]
    assert set(expected["status"]) == {"not_verifiable"}
    assert set(expected["reason"]) == {"probability_unavailable"}
    assert expected["metric_value"].isna().all()

    incomplete = frame.copy(deep=True)
    incomplete.loc[0, "exposure"] = np.nan
    result = simulate_decision_strategy(incomplete, config, risk_validation=risk)
    overall = result.business_summary.loc[
        (result.business_summary["scope_type"] == "overall")
        & result.business_summary["action_role"].isna()
        & (result.business_summary["metric_key"] == "expected_loss_sum")
    ]
    assert overall.iloc[0]["status"] == "not_verifiable"
    assert overall.iloc[0]["reason"] == "exposure_unavailable"
    assert pd.isna(overall.iloc[0]["metric_value"])


def test_expected_loss_zero_denominator_and_constraint_support_precedence() -> None:
    frame = pd.DataFrame({"x": range(8), "target": [0, 1] * 4, "exposure": [0.0] * 8})
    constraint = DecisionConstraint(
        "support", "action_count", "ge", 0.0, action_name="select", minimum_support=9
    )
    config = replace(
        _config(constraints=(constraint,)),
        exposure_column="exposure",
        loss_fraction=0.5,
        exposure_unit="units",
    )
    result = simulate_decision_strategy(
        frame, config, risk_validation=_risk_result(frame)
    )
    rate = result.business_summary.loc[
        (result.business_summary["scope_type"] == "overall")
        & result.business_summary["action_role"].isna()
        & (result.business_summary["metric_key"] == "expected_loss_rate")
    ].iloc[0]
    assert rate["status"] == "undefined"
    assert rate["reason"] == "zero_denominator"
    assert rate["numerator"] == pytest.approx(0.0)
    assert rate["denominator"] == pytest.approx(0.0)
    assert rate["support_n_rows"] == 8
    assert result.constraint_summary.iloc[0]["status"] == "undefined"
    assert result.constraint_summary.iloc[0]["reason"] == "insufficient_support"


def test_constraints_equality_failure_and_no_action_mutation() -> None:
    constraints = (
        DecisionConstraint("equal", "selected_rate", "ge", 1.0, action_role="selected"),
        DecisionConstraint("fail", "rejected_rate", "ge", 0.1, action_role="rejected"),
    )
    result = simulate_decision_strategy(
        pd.DataFrame({"x": [1, 2]}), _config(constraints=constraints)
    )
    assert result.constraint_summary["reason"].tolist() == [
        "constraint_satisfied",
        "constraint_failed",
    ]
    assert result.constraint_summary["actual_value"].tolist() == [1.0, 0.0]
    assert result.row_decisions["final_action_name"].tolist() == ["select", "select"]


def test_action_assumptions_historical_mapping_and_no_raw_values() -> None:
    config = replace(
        _config(),
        historical_action_column="old",
        historical_action_mapping=(("SECRET_RAW", "select"),),
        historical_policy_version="legacy-v1",
        exposure_column="exposure",
        loss_fraction=0.5,
        exposure_unit="units",
        action_assumptions=(
            ("select", 10.0, 1.0),
            ("review", 3.0, 0.5),
        ),
    )
    frame = pd.DataFrame(
        {"x": [1, 2], "old": ["SECRET_RAW", "UNMAPPED"], "exposure": [2.0, 3.0]}
    )
    result = simulate_decision_strategy(frame, config)
    assert result.historical_transitions["row_count"].tolist() == [1, 1]
    payload = " ".join(table.astype(str).to_numpy().ravel().tolist() for table in ())
    payload = " ".join(
        str(value)
        for name in (
            "row_decisions",
            "historical_transitions",
            "provenance",
        )
        for value in getattr(result, name).to_numpy().ravel()
    )
    assert "SECRET_RAW" not in payload
    assert "UNMAPPED" not in payload


def test_segment_time_ordinals_are_anonymous_and_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "segment": ["SECRET_A", "SECRET_B", "SECRET_A", None],
            "period": ["SECRET_T1", "SECRET_T1", "SECRET_T2", "SECRET_T2"],
        },
        index=[9, 9, 8, 8],
    )
    config = replace(
        _config(), segment_columns=("segment",), time_slice_column="period"
    )
    result = simulate_decision_strategy(frame, config)
    assert set(result.business_summary["scope_type"]) == {
        "overall",
        "segment",
        "time_slice",
        "segment_time",
    }
    text = " ".join(map(str, result.business_summary.to_numpy().ravel()))
    assert "SECRET_A" not in text
    assert "SECRET_T1" not in text
    second = simulate_decision_strategy(frame, config)
    pd.testing.assert_frame_equal(result.business_summary, second.business_summary)


def test_invalid_later_condition_and_resource_limits_fail_before_partial_result() -> (
    None
):
    valid = DecisionRule("a", "decision", 1, _condition(), "select")
    invalid = DecisionRule(
        "b",
        "decision",
        2,
        StrategyCondition("atomic", "eq", "column", "x", "literal", object()),
        "select",
    )
    with pytest.raises(ValueError, match="^decision condition is invalid:"):
        simulate_decision_strategy(pd.DataFrame({"x": [2]}), _config(valid, invalid))
    with pytest.raises(ValueError, match="^decision strategy resource limit exceeded:"):
        simulate_decision_strategy(
            pd.DataFrame({"x": [1]}),
            replace(_config(), segment_columns=("a", "b", "c", "d", "e")),
        )


@pytest.mark.parametrize(
    ("operator", "right", "expected"),
    [
        ("eq", 2, ["false", "true", "unknown"]),
        ("ne", 2, ["true", "false", "unknown"]),
        ("lt", 2, ["true", "false", "unknown"]),
        ("le", 2, ["true", "true", "unknown"]),
        ("gt", 2, ["false", "false", "unknown"]),
        ("ge", 2, ["false", "true", "unknown"]),
        ("in", (1, 3), ["true", "false", "unknown"]),
        ("not_in", (1, 3), ["false", "true", "unknown"]),
        ("between", (1, 2), ["true", "true", "unknown"]),
        ("is_missing", None, ["false", "false", "true"]),
        ("is_not_missing", None, ["true", "true", "false"]),
    ],
)
def test_public_atomic_conditions_match_private_kernel(
    operator: str, right: object, expected: list[str]
) -> None:
    frame = pd.DataFrame({"x": [1, 2, None]}, index=[7, 7, 7])
    public = StrategyCondition(
        "atomic",
        operator,  # type: ignore[arg-type]
        "column",
        "x",
        None if right is None else "literal",
        right,
    )
    rule = DecisionRule("r", "decision", 1, public, "select")
    result = simulate_decision_strategy(frame, _config(rule))
    direct = _evaluate_atomic_condition(
        frame,
        operator=operator,
        left=_ConditionOperand("column", "x"),
        right=None if right is None else _ConditionOperand("literal", right),
        root_version="v1",
    )
    detail = result.rule_evaluations
    assert detail["truth"].tolist() == expected == direct.truth.tolist()
    assert detail["status"].tolist() == direct.status.tolist()
    assert detail["reason"].tolist() == direct.reason.tolist()


def test_public_boolean_tree_matches_private_kernel_truth_and_reasons() -> None:
    frame = pd.DataFrame({"x": [1, 2, None]})
    first = StrategyCondition("atomic", "eq", "column", "x", "literal", 1)
    second = StrategyCondition("atomic", "eq", "column", "x", "literal", 2)
    public = StrategyCondition(
        "not", children=(StrategyCondition("or", children=(first, second)),)
    )
    result = simulate_decision_strategy(
        frame, _config(DecisionRule("r", "decision", 1, public, "select"))
    )
    left = _ConditionNode(
        "atomic",
        "eq",
        _ConditionOperand("column", "x"),
        _ConditionOperand("literal", 1),
        (),
        None,
        None,
        None,
    )
    right = replace(left, right=_ConditionOperand("literal", 2))
    joined = _ConditionNode("or", None, None, None, (left, right), None, None, None)
    private = _ConditionNode("not", None, None, None, (joined,), None, None, "v1")
    direct = _evaluate_condition(frame, private)
    assert result.rule_evaluations["truth"].tolist() == direct.truth.tolist()
    assert result.rule_evaluations["reason"].tolist() == direct.reason.tolist()


def test_lower_risk_ranking_flips_ordering_without_becoming_probability() -> None:
    condition = StrategyCondition("atomic", "ge", "ranking_score", None, "literal", 0.5)
    config = _config(
        DecisionRule("low", "decision", 1, condition, "reject"),
        ranking_score_column="score",
        ranking_score_direction="lower_risk",
    )
    result = simulate_decision_strategy(
        pd.DataFrame({"score": [0.4, 0.5, 0.6]}), config
    )
    assert result.row_decisions["final_action_name"].tolist() == [
        "reject",
        "reject",
        "select",
    ]
    probability = result.business_summary.loc[
        result.business_summary["metric_key"] == "event_probability_mean"
    ]
    assert set(probability["status"]) == {"not_applicable"}


def test_rule_window_inherits_strategy_expiry_and_exact_boundaries() -> None:
    end = datetime(2025, 2, 1)
    rule = DecisionRule("r", "decision", 1, _condition(value=1), "reject")
    active = replace(_config(rule), expires_at=end, evaluation_time=START)
    result = simulate_decision_strategy(pd.DataFrame({"x": [2]}), active)
    assert result.row_decisions.at[0, "final_action_name"] == "reject"
    expired = replace(active, evaluation_time=end)
    result = simulate_decision_strategy(pd.DataFrame({"x": [2]}), expired)
    assert result.row_decisions.at[0, "decision_status"] == "inactive"
    assert set(result.action_summary["status"]) == {"inactive"}
    assert set(result.business_summary["status"]) == {"inactive"}


def test_leave_one_out_replays_rule_without_changing_frozen_actions() -> None:
    first = DecisionRule("first", "decision", 1, _condition(value=2), "reject")
    second = DecisionRule("second", "decision", 2, _condition(value=1), "select")
    result = simulate_decision_strategy(
        pd.DataFrame({"x": [1, 2, 3]}), _config(first, second)
    )
    metric = result.rule_summary.loc[
        (result.rule_summary["scope_type"] == "overall")
        & (result.rule_summary["metric_key"] == "leave_one_out_changed_action_count")
    ].set_index("rule_key")["metric_value"]
    assert metric.to_dict() == {"first": 2.0, "second": 0.0}
    assert result.row_decisions["final_action_name"].tolist() == [
        "select",
        "reject",
        "reject",
    ]


def test_all_thirteen_constraints_use_hand_calculated_evidence() -> None:
    frame = pd.DataFrame(
        {
            "target": [0, 1] * 4,
            "exposure": np.arange(1.0, 9.0),
            "observed_loss": np.arange(0.0, 8.0),
        }
    )
    risk = _risk_result(frame, business=True)
    probabilities = np.linspace(0.1, 0.9, 8)
    exposure = np.arange(1.0, 9.0)
    expected_sum = float(np.sum(probabilities * exposure * 0.5))
    expected_rate = expected_sum / float(exposure.sum())
    observed_sum = float(np.arange(0.0, 8.0).sum())
    observed_rate = observed_sum / float(exposure.sum())
    expected = {
        "action_count": 8.0,
        "action_rate": 1.0,
        "selected_rate": 1.0,
        "rejected_rate": 0.0,
        "review_count": 0.0,
        "review_rate": 0.0,
        "request_information_rate": 0.0,
        "selected_exposure_sum": float(exposure.sum()),
        "expected_loss_sum": expected_sum,
        "expected_loss_rate": expected_rate,
        "observed_loss_sum": observed_sum,
        "observed_loss_rate": observed_rate,
        "selected_event_rate": 0.5,
    }
    role = {
        "selected_rate": "selected",
        "rejected_rate": "rejected",
        "review_count": "review",
        "review_rate": "review",
        "request_information_rate": "request_information",
        "selected_exposure_sum": "selected",
        "selected_event_rate": "selected",
    }
    constraints = tuple(
        DecisionConstraint(
            f"c{ordinal}",
            metric,  # type: ignore[arg-type]
            "ge",
            value,
            action_name="select" if metric in {"action_count", "action_rate"} else None,
            action_role=role.get(metric),  # type: ignore[arg-type]
        )
        for ordinal, (metric, value) in enumerate(expected.items())
    )
    config = replace(
        _config(constraints=constraints),
        exposure_column="exposure",
        loss_fraction=0.5,
        exposure_unit="units",
    )
    result = simulate_decision_strategy(frame, config, risk_validation=risk)
    assert result.constraint_summary["metric"].tolist() == list(expected)
    np.testing.assert_allclose(
        result.constraint_summary["actual_value"].astype(float),
        list(expected.values()),
        rtol=0,
        atol=1e-12,
    )
    assert set(result.constraint_summary["reason"]) == {"constraint_satisfied"}


def test_three_loss_claims_and_assumption_payoff_remain_separate() -> None:
    frame = pd.DataFrame(
        {
            "target": [0, 1] * 4,
            "exposure": np.arange(1.0, 9.0),
            "observed_loss": np.arange(0.0, 8.0),
        }
    )
    risk = _risk_result(frame, business=True)
    config = replace(
        _config(),
        exposure_column="exposure",
        loss_fraction=0.5,
        action_assumptions=(("select", 10.0, 1.0), ("review", 3.0, 0.5)),
        exposure_unit="units",
    )
    result = simulate_decision_strategy(frame, config, risk_validation=risk)
    overall = result.business_summary.loc[
        (result.business_summary["scope_type"] == "overall")
        & result.business_summary["action_role"].isna()
    ].set_index("metric_key")
    expected_loss = float(np.sum(np.linspace(0.1, 0.9, 8) * np.arange(1.0, 9.0) * 0.5))
    observed_assumption = float(
        np.sum(np.array([0, 1] * 4) * np.arange(1.0, 9.0) * 0.5)
    )
    assert overall.at["expected_loss_sum", "metric_value"] == pytest.approx(
        expected_loss
    )
    assert overall.at[
        "assumption_based_observed_event_loss_sum", "metric_value"
    ] == pytest.approx(observed_assumption)
    assert overall.at["actual_observed_loss_sum", "metric_value"] == pytest.approx(28.0)
    assert overall.at["assumption_based_payoff_sum", "metric_value"] == pytest.approx(
        8 * 10.0 - 8 * 1.0 - expected_loss
    )


def test_observed_event_loss_uses_only_complete_mature_subset() -> None:
    frame = pd.DataFrame({"target": [0, 1] * 4, "exposure": np.arange(1.0, 9.0)})
    risk = _with_immature_positions(_risk_result(frame), (0,))
    config = replace(
        _config(), exposure_column="exposure", loss_fraction=0.5, exposure_unit="units"
    )
    result = simulate_decision_strategy(frame, config, risk_validation=risk)
    metric = result.business_summary.loc[
        (result.business_summary["scope_type"] == "overall")
        & result.business_summary["action_role"].isna()
        & (
            result.business_summary["metric_key"]
            == "assumption_based_observed_event_loss_sum"
        )
    ].iloc[0]
    assert metric["metric_value"] == pytest.approx(10.0)
    assert metric["support_n_rows"] == 7
    assert metric["status"] == "available"
    assert metric["reason"] == "computed"

    immature_missing = frame.copy(deep=True)
    immature_missing.loc[0, "exposure"] = np.nan
    metric = simulate_decision_strategy(
        immature_missing, config, risk_validation=risk
    ).business_summary
    metric = metric.loc[
        (metric["scope_type"] == "overall")
        & metric["action_role"].isna()
        & (metric["metric_key"] == "assumption_based_observed_event_loss_sum")
    ].iloc[0]
    assert metric["metric_value"] == pytest.approx(10.0)
    assert metric["support_n_rows"] == 7
    assert metric["status"] == "available"


def test_observed_event_loss_zero_mature_and_mature_component_gap() -> None:
    frame = pd.DataFrame({"target": [0, 1] * 4, "exposure": np.arange(1.0, 9.0)})
    config = replace(
        _config(), exposure_column="exposure", loss_fraction=0.5, exposure_unit="units"
    )
    no_mature = _with_immature_positions(_risk_result(frame), tuple(range(8)))
    result = simulate_decision_strategy(frame, config, risk_validation=no_mature)
    metric = result.business_summary.loc[
        (result.business_summary["scope_type"] == "overall")
        & result.business_summary["action_role"].isna()
        & (
            result.business_summary["metric_key"]
            == "assumption_based_observed_event_loss_sum"
        )
    ].iloc[0]
    assert metric["status"] == "not_verifiable"
    assert metric["reason"] == "label_not_evaluable"
    assert metric["support_n_rows"] == 0

    risk = _with_immature_positions(_risk_result(frame), (0,))
    mature_missing = frame.copy(deep=True)
    mature_missing.loc[1, "exposure"] = np.nan
    result = simulate_decision_strategy(mature_missing, config, risk_validation=risk)
    metric = result.business_summary.loc[
        (result.business_summary["scope_type"] == "overall")
        & result.business_summary["action_role"].isna()
        & (
            result.business_summary["metric_key"]
            == "assumption_based_observed_event_loss_sum"
        )
    ].iloc[0]
    assert metric["status"] == "not_verifiable"
    assert metric["reason"] == "exposure_unavailable"


@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate_loss",
        "duplicate_exposure",
        "missing_loss",
        "missing_exposure",
        "unavailable_loss",
        "unavailable_exposure",
        "nonfinite_loss",
        "nonfinite_exposure",
        "support_mismatch",
        "unit_mismatch",
        "non_overall_loss",
    ),
)
def test_actual_observed_loss_requires_compatible_unique_aggregates(
    mutation: str,
) -> None:
    frame = pd.DataFrame(
        {
            "target": [0, 1] * 4,
            "exposure": np.arange(1.0, 9.0),
            "observed_loss": np.arange(0.0, 8.0),
        }
    )
    risk = _risk_result(frame, business=True)
    table = risk.business_metrics.copy(deep=True)
    loss = (table["segment_kind"] == "all") & (table["metric"] == "observed_loss_sum")
    exposure = (table["segment_kind"] == "all") & (table["metric"] == "exposure_sum")
    if mutation == "duplicate_loss":
        table = pd.concat([table, table.loc[loss]], ignore_index=True)
    elif mutation == "duplicate_exposure":
        table = pd.concat([table, table.loc[exposure]], ignore_index=True)
    elif mutation == "missing_loss":
        table = table.loc[~loss].reset_index(drop=True)
    elif mutation == "missing_exposure":
        table = table.loc[~exposure].reset_index(drop=True)
    elif mutation == "unavailable_loss":
        table.loc[loss, "status"] = "not_verifiable"
    elif mutation == "unavailable_exposure":
        table.loc[exposure, "status"] = "not_verifiable"
    elif mutation == "nonfinite_loss":
        table.loc[loss, "value"] = np.inf
    elif mutation == "nonfinite_exposure":
        table.loc[exposure, "value"] = np.inf
    elif mutation == "support_mismatch":
        table.loc[exposure, "n_observed_loss_mature_rows"] = 1
    elif mutation == "unit_mismatch":
        table.loc[exposure, "unit"] = "different"
    else:
        table.loc[loss, "segment_kind"] = "score_band"
    result = simulate_decision_strategy(
        frame, _config(), risk_validation=replace(risk, business_metrics=table)
    )
    metrics = result.business_summary.loc[
        (result.business_summary["scope_type"] == "overall")
        & result.business_summary["action_role"].isna()
        & result.business_summary["metric_key"].isin(
            ["actual_observed_loss_sum", "actual_observed_loss_rate"]
        )
    ]
    assert set(metrics["status"]) == {"not_verifiable"}
    assert metrics["metric_value"].isna().all()
    assert set(metrics["reason"]) <= {
        "observed_loss_not_mature",
        "exposure_unavailable",
    }


def test_empty_result_tables_keep_frozen_columns_and_nullable_dtypes() -> None:
    result = simulate_decision_strategy(pd.DataFrame({"x": []}), _config())
    assert result.row_decisions.empty
    assert result.rule_evaluations.empty
    assert result.rule_summary.empty
    assert result.constraint_summary.empty
    assert result.historical_transitions.empty
    assert tuple(result.constraint_summary.columns) == (
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
    assert str(result.constraint_summary["actual_value"].dtype) == "Float64"


def test_resource_scope_limits_fail_without_truncation() -> None:
    frame = pd.DataFrame({"segment": [f"v{value}" for value in range(101)]})
    with pytest.raises(
        ValueError,
        match="^decision strategy resource limit exceeded: segment_categories$",
    ):
        simulate_decision_strategy(
            frame, replace(_config(), segment_columns=("segment",))
        )


def test_scope_resource_empty_maximum_and_max_plus_one_boundaries() -> None:
    empty = pd.DataFrame({f"s{index}": [] for index in range(4)})
    result = simulate_decision_strategy(
        empty, replace(_config(), segment_columns=("s0", "s1", "s2", "s3"))
    )
    assert set(result.business_summary["scope_type"]) == {"overall"}

    maximum = pd.DataFrame({f"s{index}": list(range(100)) for index in range(4)})
    result = simulate_decision_strategy(
        maximum, replace(_config(), segment_columns=("s0", "s1", "s2", "s3"))
    )
    assert result.business_summary["scope_type"].eq("segment").sum() > 0

    with pytest.raises(
        ValueError,
        match="^decision strategy resource limit exceeded: segment_columns$",
    ):
        simulate_decision_strategy(
            pd.DataFrame({"x": []}),
            replace(_config(), segment_columns=("a", "b", "c", "d", "e")),
        )
    with pytest.raises(
        ValueError,
        match="^decision strategy resource limit exceeded: segment_categories$",
    ):
        simulate_decision_strategy(
            pd.DataFrame({"segment": list(range(101))}),
            replace(_config(), segment_columns=("segment",)),
        )


def test_time_derived_scope_and_summary_resource_boundaries() -> None:
    maximum_time = pd.DataFrame({"time": list(range(100))})
    result = simulate_decision_strategy(
        maximum_time, replace(_config(), time_slice_column="time")
    )
    assert result.business_summary["time_slice_ordinal"].dropna().nunique() == 100
    with pytest.raises(
        ValueError, match="^decision strategy resource limit exceeded: time_slices$"
    ):
        simulate_decision_strategy(
            pd.DataFrame({"time": list(range(101))}),
            replace(_config(), time_slice_column="time"),
        )

    def scope_frame(n_rows: int) -> pd.DataFrame:
        segment = [position % 100 for position in range(n_rows)]
        time = [
            position if position < 100 else (position - 99) % 100
            for position in range(n_rows)
        ]
        frame = pd.DataFrame(
            {
                "s0": segment,
                "s1": segment,
                "s2": segment,
                "s3": segment,
                "time": time,
            }
        )
        if n_rows == 126:
            # s0 has 126 segment-time pairs; s1..s3 each keep 125 by
            # reusing the existing (segment=26, time=26) pair. Therefore:
            # 400 segment + 100 time + (126 + 3 * 125) = 1001 scopes.
            frame.loc[125, ["s1", "s2", "s3"]] = 26
        return frame

    config = replace(
        _config(),
        segment_columns=("s0", "s1", "s2", "s3"),
        time_slice_column="time",
    )
    with pytest.raises(
        ValueError,
        match="^decision strategy resource limit exceeded: scope_summary_rows$",
    ):
        # 400 segment + 100 time + 4 * 125 segment-time = 1000. This
        # passes the derived-scope gate and reaches the later summary gate.
        simulate_decision_strategy(scope_frame(125), config)
    with pytest.raises(
        ValueError, match="^decision strategy resource limit exceeded: derived_scopes$"
    ):
        simulate_decision_strategy(scope_frame(126), config)


def test_scope_summary_row_exact_reachable_boundary() -> None:
    # Frozen independent formula for n non-overall scopes, r rules, a actions:
    # r * (15 + 23n) + a * (11 + 15n) + 4 * (25 + 47n).
    maximum_frame = pd.DataFrame(
        {
            "segment": range(97),
            "x": [0] * 97,
        }
    )
    maximum_config = _resource_config(22, 22, segment_columns=("segment",))
    maximum_projected = 22 * (15 + 23 * 97) + 22 * (11 + 15 * 97) + 4 * (25 + 47 * 97)
    assert maximum_projected == 100_000
    maximum = simulate_decision_strategy(maximum_frame, maximum_config)
    assert (
        len(maximum.rule_summary)
        + len(maximum.action_summary)
        + len(maximum.business_summary)
        == maximum_projected
    )

    over_frame = pd.DataFrame({"segment": range(38), "x": [0] * 38})
    over_config = _resource_config(88, 25, segment_columns=("segment",))
    over_projected = 88 * (15 + 23 * 38) + 25 * (11 + 15 * 38) + 4 * (25 + 47 * 38)
    assert over_projected == 100_001
    with pytest.raises(
        ValueError,
        match="^decision strategy resource limit exceeded: scope_summary_rows$",
    ):
        simulate_decision_strategy(over_frame, over_config)


def test_inventory_key_input_and_rule_evaluation_resource_boundaries() -> None:
    rules = tuple(
        DecisionRule(f"e{priority}", "eligibility", priority, _condition(), "reject")
        for priority in range(50)
    ) + tuple(
        DecisionRule(f"d{priority}", "decision", priority, _condition(), "select")
        for priority in range(50)
    )
    result = simulate_decision_strategy(pd.DataFrame({"x": []}), _config(*rules))
    assert result.requested_rule_count == 100
    with pytest.raises(
        ValueError, match="^decision strategy resource limit exceeded: all_rules$"
    ):
        simulate_decision_strategy(
            pd.DataFrame({"x": []}),
            _config(
                *rules,
                DecisionRule("extra", "decision", 50, _condition(), "select"),
            ),
        )

    constraints = tuple(
        DecisionConstraint(f"c{index}", "action_count", "ge", 0, action_name="select")
        for index in range(50)
    )
    assert (
        simulate_decision_strategy(
            pd.DataFrame({"x": []}), _config(constraints=constraints)
        ).requested_constraint_count
        == 50
    )
    with pytest.raises(
        ValueError, match="^decision strategy resource limit exceeded: constraints$"
    ):
        simulate_decision_strategy(
            pd.DataFrame({"x": []}),
            _config(
                constraints=constraints + (replace(constraints[0], constraint_key="x"),)
            ),
        )

    long_key = "k" * 64
    simulate_decision_strategy(
        pd.DataFrame({"x": []}), replace(_config(), strategy_key=long_key)
    )
    with pytest.raises(
        ValueError,
        match="^decision strategy resource limit exceeded: strategy_key$",
    ):
        simulate_decision_strategy(
            pd.DataFrame({"x": []}), replace(_config(), strategy_key="k" * 65)
        )

    maximum_input = simulate_decision_strategy(
        pd.DataFrame(index=range(100_000)), _config()
    )
    assert maximum_input.input_n_rows == 100_000
    with pytest.raises(
        ValueError, match="^decision strategy resource limit exceeded: input_rows$"
    ):
        simulate_decision_strategy(pd.DataFrame(index=range(100_001)), _config())


def test_rule_evaluation_row_exact_reachable_boundary() -> None:
    maximum_rules = _resource_config(100, 2).rules
    maximum = simulate_decision_strategy(
        pd.DataFrame({"x": [0] * 10_000}),
        _resource_config(100, 2),
    )
    assert len(maximum_rules) * 10_000 == 1_000_000
    assert len(maximum.rule_evaluations) == 1_000_000

    # 1,000,001 factors only as 101 * 9,901 within the row range, so it is
    # unreachable with at most 100 rules. An exhaustive independent integer
    # check freezes 1,000,004 = 53 * 18,868 as the smallest legal excess.
    smallest_excess = min(
        rows * rules
        for rules in range(1, 101)
        for rows in (1_000_000 // rules + 1,)
        if rows <= 100_000
    )
    assert smallest_excess == 1_000_004 == 53 * 18_868
    with pytest.raises(
        ValueError,
        match="^decision strategy resource limit exceeded: rule_evaluation_rows$",
    ):
        simulate_decision_strategy(
            pd.DataFrame({"x": [0] * 18_868}),
            _resource_config(53, 2),
        )


def test_task17_compile_through_inherits_condition_depth_budget() -> None:
    condition = StrategyCondition("atomic", "eq", "column", "x", "literal", 1)
    for _ in range(7):
        condition = StrategyCondition("not", children=(condition,))
    simulate_decision_strategy(
        pd.DataFrame({"x": [1]}),
        _config(DecisionRule("depth8", "decision", 1, condition, "select")),
    )
    too_deep = StrategyCondition("not", children=(condition,))
    with pytest.raises(
        ValueError,
        match="^decision strategy resource limit exceeded: condition_depth_exceeded$",
    ):
        simulate_decision_strategy(
            pd.DataFrame({"x": [1]}),
            _config(DecisionRule("depth9", "decision", 1, too_deep, "select")),
        )


def test_mapping_assumption_and_historical_resource_maxima() -> None:
    action_rules = tuple(
        DecisionRule(f"r{index}", "decision", index, _condition(), f"action{index}")
        for index in range(48)
    )
    roles = (("select", "selected"), ("review", "review")) + tuple(
        (f"action{index}", "other") for index in range(48)
    )
    assumptions = tuple((action, 1.0, 0.0) for action, _ in roles)
    config = replace(
        _config(),
        rules=action_rules,
        action_role_mapping=roles,
        action_assumptions=assumptions,
        exposure_unit="units",
    )
    result = simulate_decision_strategy(pd.DataFrame({"x": []}), config)
    assert len(result.action_summary["action_key"].drop_duplicates()) == 50
    with pytest.raises(
        ValueError,
        match="^decision strategy resource limit exceeded: action_role_mappings$",
    ):
        simulate_decision_strategy(
            pd.DataFrame({"x": []}),
            replace(config, action_role_mapping=roles + (("extra", "other"),)),
        )
    with pytest.raises(
        ValueError,
        match="^decision strategy resource limit exceeded: action_assumptions$",
    ):
        simulate_decision_strategy(
            pd.DataFrame({"x": []}),
            replace(config, action_assumptions=assumptions + (("extra", 1.0, 0.0),)),
        )

    historical = tuple((index, "select") for index in range(50))
    historical_config = replace(
        _config(),
        historical_action_column="old",
        historical_action_mapping=historical,
    )
    simulate_decision_strategy(pd.DataFrame({"old": []}), historical_config)
    with pytest.raises(
        ValueError,
        match="^decision strategy resource limit exceeded: historical_action_mappings$",
    ):
        simulate_decision_strategy(
            pd.DataFrame({"old": []}),
            replace(
                historical_config,
                historical_action_mapping=historical + ((50, "select"),),
            ),
        )


@pytest.mark.parametrize(
    "bad_right",
    ([1, 2], {1, 2}, {"value": 1}, lambda value: value),
)
def test_public_condition_rejects_non_closed_inputs_without_repr(
    bad_right: object,
) -> None:
    condition = StrategyCondition("atomic", "eq", "column", "x", "literal", bad_right)
    with pytest.raises(ValueError, match="^decision condition is invalid:"):
        simulate_decision_strategy(
            pd.DataFrame({"x": [1]}),
            _config(DecisionRule("bad", "decision", 1, condition, "select")),
        )
