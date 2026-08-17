"""Task 19 approved model-governance contract tests."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timedelta
from functools import cache
from inspect import signature
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

import sharper.model_governance as governance
from sharper import (
    BinaryRiskValidationConfig,
    BinaryRiskValidationResult,
    DataAuditResult,
    DecisionRule,
    DecisionStrategyConfig,
    DecisionStrategyResult,
    ExternalRiskPredictions,
    GovernanceAttributionEvidence,
    GovernanceCandidate,
    GovernanceCriterion,
    GovernanceEvidenceRef,
    GovernanceExplanation,
    GovernanceMetadata,
    GovernancePerformanceEvidence,
    GovernancePolicy,
    GovernancePredictionProfile,
    GovernanceResult,
    LifecycleMonitoringResult,
    StrategyCondition,
    audit_data_quality,
    evaluate_governance,
    plot_model_governance,
    simulate_decision_strategy,
    validate_binary_risk,
)


def _risk_result(offset: float = 0.0):
    frame = pd.DataFrame({"x": range(12), "target": [False, True] * 6})
    positions = tuple(range(12))
    probabilities = tuple(
        float(x) for x in np.linspace(0.05 + offset, 0.85 + offset, 12)
    )
    fold_by_position: dict[int, int] = {}
    fit_rows = []
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (train, validation) in enumerate(
        splitter.split(np.arange(12), frame["target"])
    ):
        fit_rows.append((fold, tuple(sorted(int(i) for i in train))))
        for index in validation:
            fold_by_position[int(index)] = fold
    external = ExternalRiskPredictions(
        row_positions=positions,
        fold_ids=tuple(fold_by_position[i] for i in positions),
        fold_fit_row_positions=tuple(fit_rows),
        ranking_scores=probabilities,
        ranking_direction="higher_risk",
        event_probabilities=probabilities,
        probability_positive_label=True,
        probability_provenance="external_declared",
    )
    return validate_binary_risk(
        frame,
        "target",
        config=BinaryRiskValidationConfig(
            validation_mode="stratified_kfold",
            n_splits=3,
            random_state=42,
        ),
        external_predictions=external,
    )


def _metadata() -> GovernanceMetadata:
    return GovernanceMetadata(
        metadata_key="governance_metadata",
        metadata_scope="governance",
        candidate_key=None,
        purpose_key="offline_review",
        owner_key="risk_team",
        materiality="high",
    )


def _candidate(key: str, position: int, role: str, refs=()) -> GovernanceCandidate:
    return GovernanceCandidate(
        candidate_key=key,
        candidate_family="model",
        source_task="task15",
        source_result_position=position,
        source_candidate_key=None,
        expected_source_fingerprint=None,
        version="v1",
        declared_role=role,
        declared_state="approved" if role == "champion" else "candidate",
        evidence_refs=refs,
    )


def _fold_ref(candidate: str, position: int, use: str) -> GovernanceEvidenceRef:
    return GovernanceEvidenceRef(
        source_task="task15",
        source_result_position=position,
        source_table="folds",
        source_use=use,
        candidate_key=candidate,
        expected_source_fingerprint=None,
        fold_id=0,
        field_key="fold_id",
    )


def _policy(*candidates: GovernanceCandidate, pairs=(), criteria=(), explanations=()):
    return GovernancePolicy(
        governance_key="governance_1",
        governance_version="v1",
        analysis_as_of=datetime(2025, 1, 31),
        candidates=tuple(candidates),
        comparison_pairs=pairs,
        criteria=criteria,
        metadata=(_metadata(),),
        explanations=explanations,
    )


def _comparison_output(
    champion_value: float,
    challenger_value: float,
    *,
    challenger_state: str = "candidate",
    second_values: tuple[float, float] | None = None,
    support_mismatch: bool = False,
    unavailable: bool = False,
    required: bool = True,
    minimum: int = 1,
    human_review_mode: str = "promotion_only",
) -> GovernanceResult:
    owners = []
    for position, value in enumerate((champion_value, challenger_value)):
        owner = _t2_time_task15_owner()
        metrics = owner.metrics.copy(deep=True)
        selector = (
            metrics["scope"].eq("fold")
            & metrics["fold_id"].eq(0)
            & metrics["metric"].eq("roc_auc")
            & metrics["statistic"].eq("direct")
        )
        metrics.loc[selector, "value"] = value
        if support_mismatch and position == 1:
            metrics.loc[selector, "n_rows"] = 3
        if unavailable and position == 1:
            metrics.loc[selector, "status"] = "unavailable"
            metrics.loc[selector, "reason"] = "single_class"
        if second_values is not None:
            second = (
                metrics["scope"].eq("fold")
                & metrics["fold_id"].eq(0)
                & metrics["metric"].eq("brier_score")
                & metrics["statistic"].eq("direct")
            )
            metrics.loc[second, "value"] = second_values[position]
        owners.append(replace(owner, metrics=metrics))
    refs = []
    for candidate_key, result_position in (("champion", 0), ("challenger", 1)):
        primary = GovernanceEvidenceRef(
            "task15",
            result_position,
            "metrics",
            "comparison_criterion",
            candidate_key,
            None,
            metric_key="roc_auc",
            scope_key="fold",
            fold_id=0,
            statistic_key="direct",
        )
        values = [primary]
        if second_values is not None:
            values.append(replace(primary, metric_key="brier_score"))
        refs.append(tuple(values))
    champion = _candidate("champion", 0, "champion", refs[0])
    challenger = replace(
        _candidate("challenger", 1, "challenger", refs[1]),
        declared_state=challenger_state,
    )
    criteria = [
        GovernanceCriterion(
            "auc",
            "model",
            "task15",
            "metrics",
            "roc_auc",
            "fold",
            None,
            None,
            "decision",
            required,
            "higher_is_better",
            minimum_support=minimum,
        )
    ]
    if second_values is not None:
        criteria.append(
            GovernanceCriterion(
                "brier",
                "model",
                "task15",
                "metrics",
                "brier_score",
                "fold",
                None,
                None,
                "decision",
                True,
                "lower_is_better",
            )
        )
    policy = replace(
        _policy(
            champion,
            challenger,
            pairs=(("champion", "challenger"),),
            criteria=tuple(criteria),
        ),
        minimum_comparable_criteria=minimum,
        human_review_mode=human_review_mode,
    )
    return evaluate_governance(policy, risk_validations=tuple(owners))


def test_public_inventory_and_signatures_are_exact() -> None:
    assert len(governance._SOURCE_REGISTRY) == 38
    assert len(governance._DIRECTION_REGISTRY) == 92
    assert len(governance._ERROR_KEYS) == 76
    assert len(governance._REASONS) == 15
    assert len(governance._SCHEMAS) == 10
    assert list(signature(evaluate_governance).parameters) == [
        "policy",
        "risk_validations",
        "data_audits",
        "decision_strategies",
        "lifecycle_monitorings",
        "model_attributions",
        "prediction_profiles",
        "performance_evidence",
    ]


def test_ten_typed_tables_and_35_provenance_rows() -> None:
    result = _risk_result()
    output = evaluate_governance(
        _policy(_candidate("champion", 0, "champion")), risk_validations=(result,)
    )
    assert type(output) is GovernanceResult
    tables = [getattr(output, name) for name in governance._SCHEMAS]
    assert len(tables) == 10
    for name, table in zip(governance._SCHEMAS, tables, strict=True):
        assert list(table.columns) == [x for x, _ in governance._SCHEMAS[name]]
        assert all(str(dtype) != "object" for dtype in table.dtypes)
    assert output.provenance["provenance_position"].tolist() == list(range(35))
    assert len(output.governance_summary) == 1


def test_dataclasses_are_frozen() -> None:
    candidate = _candidate("champion", 0, "champion")
    with pytest.raises(FrozenInstanceError):
        candidate.candidate_key = "changed"  # type: ignore[misc]
    assert [field.name for field in fields(GovernanceCandidate)][0] == "candidate_key"


def test_attribution_drift_stability_and_metadata_materialize() -> None:
    owner = _risk_result()
    candidate = _candidate("champion", 0, "champion")
    attribution_ref = _fold_ref("champion", 0, "attribution_context")
    drift_ref = _fold_ref("champion", 0, "drift_context")
    stability_ref = _fold_ref("champion", 0, "stability_context")
    attribution = GovernanceAttributionEvidence(
        "champion",
        "coefficient_direction",
        "feature_1",
        None,
        0.5,
        "positive",
        "not_applicable",
        12,
        None,
        None,
        None,
        datetime(2025, 1, 10),
        attribution_ref,
    )
    boundaries = tuple(float(i) / 10 for i in range(1, 10))
    profiles = (
        GovernancePredictionProfile(
            "champion",
            "reference",
            "reference",
            datetime(2025, 1, 1),
            "event_probability",
            "overall",
            None,
            "a" * 64,
            boundaries,
            (1,) * 10,
            10,
            0,
            2,
            7,
            drift_ref,
        ),
        GovernancePredictionProfile(
            "champion",
            "current",
            "current",
            datetime(2025, 1, 20),
            "event_probability",
            "overall",
            None,
            "a" * 64,
            boundaries,
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 10),
            10,
            0,
            2,
            7,
            drift_ref,
        ),
    )
    performance = (
        GovernancePerformanceEvidence(
            "champion",
            "reference",
            "reference",
            datetime(2024, 12, 1),
            datetime(2024, 12, 10),
            datetime(2024, 12, 10),
            "holdout",
            "overall",
            None,
            (False, True, False, True),
            None,
            (0.1, 0.9, 0.2, 0.8),
            "unknown",
            "verified",
            4,
            3,
            stability_ref,
        ),
        GovernancePerformanceEvidence(
            "champion",
            "current",
            "current",
            datetime(2025, 1, 1),
            datetime(2025, 1, 10),
            datetime(2025, 1, 10),
            "holdout",
            "overall",
            None,
            (False, True, False, True),
            None,
            (0.2, 0.8, 0.3, 0.7),
            "unknown",
            "verified",
            4,
            3,
            stability_ref,
        ),
    )
    output = evaluate_governance(
        _policy(candidate),
        risk_validations=(owner,),
        model_attributions=(attribution,),
        prediction_profiles=profiles,
        performance_evidence=performance,
    )
    assert output.model_attributions.loc[0, "value"] == 0.5
    assert output.prediction_drift.loc[0, "prediction_tvd"] == 0.9
    assert output.performance_stability.loc[0, "metric"] == "brier_score"
    assert len(output.governance_metadata) == 5


def test_explanation_preserves_declaration_order() -> None:
    owner = _risk_result()
    candidate = _candidate("champion", 0, "champion")
    explanations = tuple(
        GovernanceExplanation(
            key,
            "champion",
            "source_lineage",
            _fold_ref("champion", 0, "explanation"),
            None,
            None,
            priority,
            "available",
            None,
        )
        for key, priority in (("z", -10), ("a", 999))
    )
    output = evaluate_governance(
        _policy(candidate, explanations=explanations), risk_validations=(owner,)
    )
    assert output.explanations["explanation_key"].tolist() == ["z", "a"]
    assert output.explanations["explanation_position"].tolist() == [0, 1]


def test_candidate_pair_and_direction_validation() -> None:
    owners = (_t2_time_task15_owner(), _t2_time_task15_owner())
    ref0 = GovernanceEvidenceRef(
        "task15",
        0,
        "metrics",
        "comparison_criterion",
        "champion",
        None,
        metric_key="roc_auc",
        scope_key="fold",
        fold_id=0,
        statistic_key="direct",
    )
    ref1 = GovernanceEvidenceRef(
        "task15",
        1,
        "metrics",
        "comparison_criterion",
        "challenger",
        None,
        metric_key="roc_auc",
        scope_key="fold",
        fold_id=0,
        statistic_key="direct",
    )
    champion = _candidate("champion", 0, "champion", (ref0,))
    challenger = _candidate("challenger", 1, "challenger", (ref1,))
    criterion = GovernanceCriterion(
        "auc",
        "model",
        "task15",
        "metrics",
        "roc_auc",
        "fold",
        None,
        None,
        "decision",
        True,
        "higher_is_better",
    )
    output = evaluate_governance(
        _policy(
            champion,
            challenger,
            pairs=(("champion", "challenger"),),
            criteria=(criterion,),
        ),
        risk_validations=owners,
    )
    assert len(output.candidate_comparisons) == 1
    assert (
        output.candidate_comparisons.loc[0, "delta"]
        == output.candidate_comparisons.loc[0, "challenger_value"]
        - output.candidate_comparisons.loc[0, "champion_value"]
    )
    assert output.recommendations.loc[0, "recommendation"] in {
        "promote_challenger",
        "retain_champion",
        "continue_evaluation",
    }


@pytest.mark.parametrize(
    "direction", ["lower_is_better", "target_range", "not_directional"]
)
def test_wrong_authoritative_direction_rejected(direction: str) -> None:
    owner = _risk_result()
    criterion = GovernanceCriterion(
        "auc",
        "model",
        "task15",
        "metrics",
        "roc_auc",
        "overall",
        None,
        None,
        "decision",
        True,
        direction,
    )
    with pytest.raises(ValueError, match="model governance: invalid_criterion$"):
        evaluate_governance(
            _policy(_candidate("champion", 0, "champion"), criteria=(criterion,)),
            risk_validations=(owner,),
        )


@pytest.mark.parametrize("bad", ["2025-01-01", np.datetime64("2025-01-01"), pd.NaT])
def test_invalid_analysis_as_of_is_rejected(bad) -> None:
    policy = _policy(_candidate("champion", 0, "champion"))
    policy = GovernancePolicy(**{**policy.__dict__, "analysis_as_of": bad})
    with pytest.raises(ValueError, match="model governance: invalid_analysis_as_of$"):
        evaluate_governance(policy)


def test_future_structured_evidence_precedes_resource_gate() -> None:
    candidate = _candidate("champion", 0, "champion")
    item = GovernanceAttributionEvidence(
        "champion",
        "coefficient_direction",
        "feature_1",
        None,
        0.5,
        "positive",
        "not_applicable",
        12,
        None,
        None,
        None,
        datetime(2026, 1, 1),
        _fold_ref("champion", 0, "attribution_context"),
    )
    distinct = tuple(_risk_result(i / 1000) for i in range(17))
    with pytest.raises(ValueError, match="model governance: future_evidence_time$"):
        evaluate_governance(
            _policy(candidate),
            risk_validations=distinct,
            model_attributions=(item,),
        )


class _Hostile:
    calls = 0

    def __repr__(self):
        type(self).calls += 1
        return "secret"

    __str__ = __repr__
    __hash__ = __repr__
    __bool__ = __repr__
    __int__ = __repr__
    __float__ = __repr__
    __iter__ = __repr__
    __eq__ = __repr__
    __lt__ = __repr__
    __array__ = __repr__


class _ProtocolHostile:
    calls: dict[str, int] = {}

    @classmethod
    def _called(cls, name: str):
        cls.calls[name] = cls.calls.get(name, 0) + 1
        raise AssertionError(name)

    def __repr__(self):
        return self._called("__repr__")

    def __str__(self):
        return self._called("__str__")

    def __hash__(self):
        return self._called("__hash__")

    def __bool__(self):
        return self._called("__bool__")

    def __int__(self):
        return self._called("__int__")

    def __float__(self):
        return self._called("__float__")

    def __iter__(self):
        return self._called("__iter__")

    def __eq__(self, other):
        return self._called("__eq__")

    def __lt__(self, other):
        return self._called("__lt__")

    def __array__(self):
        return self._called("__array__")

    def __index__(self):
        return self._called("__index__")


def test_malicious_policy_is_rejected_without_protocol_callbacks() -> None:
    _Hostile.calls = 0
    with pytest.raises(ValueError, match="model governance: invalid_policy_type$"):
        evaluate_governance(_Hostile())  # type: ignore[arg-type]
    assert _Hostile.calls == 0


@pytest.mark.parametrize(
    "field_group",
    (
        "policy",
        "candidate",
        "evidence_ref",
        "criterion",
        "explanation",
        "attribution",
        "prediction_profile",
        "performance_evidence",
        "metadata",
        "analysis_as_of",
        "owner_position",
    ),
)
def test_acceptance_malicious_field_groups_have_zero_protocol_callbacks(
    field_group: str,
) -> None:
    hostile = _ProtocolHostile()
    _ProtocolHostile.calls = {}
    candidate = _candidate("champion", 0, "champion")
    policy = _policy(candidate)
    if field_group == "policy":
        value = hostile
    elif field_group == "candidate":
        value = replace(policy, candidates=(replace(candidate, candidate_key=hostile),))
    elif field_group == "evidence_ref":
        value = replace(
            policy,
            evidence_refs=(
                replace(_fold_ref("champion", 0, "diagnostic"), source_table=hostile),
            ),
        )
    elif field_group == "criterion":
        value = replace(policy, criteria=(hostile,))
    elif field_group == "explanation":
        value = replace(policy, explanations=(hostile,))
    elif field_group == "metadata":
        value = replace(policy, metadata=(hostile,))
    elif field_group == "analysis_as_of":
        value = replace(policy, analysis_as_of=hostile)
    elif field_group == "owner_position":
        value = replace(
            policy, candidates=(replace(candidate, source_result_position=hostile),)
        )
    else:
        container = {
            "attribution": "model_attributions",
            "prediction_profile": "prediction_profiles",
            "performance_evidence": "performance_evidence",
        }[field_group]
        with pytest.raises(ValueError) as caught:
            evaluate_governance(
                policy, risk_validations=(_risk_result(),), **{container: (hostile,)}
            )
        assert str(caught.value).startswith("model governance: invalid_")
        assert _ProtocolHostile.calls == {}
        return
    with pytest.raises(ValueError) as caught:
        evaluate_governance(value, risk_validations=(_risk_result(),))  # type: ignore[arg-type]
    assert str(caught.value).startswith("model governance: invalid_")
    assert _ProtocolHostile.calls == {}


def test_canonical_encoder_is_deterministic_and_rejects_nonfinite() -> None:
    assert governance._fingerprint((1, True, "x")) == governance._fingerprint(
        (1, True, "x")
    )
    with pytest.raises(ValueError, match="model governance: invalid_canonical_value$"):
        governance._canonical_json(float("nan"))


def test_all_five_result_only_plot_kinds() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    owner = _risk_result()
    candidate = _candidate("champion", 0, "champion")
    attribution = GovernanceAttributionEvidence(
        "champion",
        "coefficient_direction",
        "feature_1",
        None,
        0.5,
        "positive",
        "not_applicable",
        12,
        None,
        None,
        None,
        datetime(2025, 1, 10),
        _fold_ref("champion", 0, "attribution_context"),
    )
    result = evaluate_governance(
        _policy(candidate), risk_validations=(owner,), model_attributions=(attribution,)
    )
    # Empty plot sources are rejected before creating a partial figure.
    for kind in ("candidate_comparison", "prediction_drift", "performance_stability"):
        with pytest.raises(ValueError, match="evidence is unavailable"):
            plot_model_governance(result, kind=kind)
    for kind in ("importance", "governance_summary"):
        figure = plot_model_governance(result, kind=kind)
        assert type(figure) is Figure
        plt.close(figure)


# Contract acceptance map: each named test below is intentionally aligned with
# one frozen acceptance obligation.  These are small, table-driven checks; the
# expensive owner kernels are exercised by the focused and full regressions.
def test_acceptance_ten_table_typed_empty_schema_matrix() -> None:
    for name, schema in governance._SCHEMAS.items():
        frame = governance._frame(name, [], aware=False)
        assert list(frame.columns) == [column for column, _ in schema]
        expected = [
            "datetime64[ns]" if dtype == "datetime" else dtype for _, dtype in schema
        ]
        assert [str(dtype) for dtype in frame.dtypes] == expected
        assert all(str(dtype) != "object" for dtype in frame.dtypes)


def test_acceptance_ten_table_populated_schema_order_dtype_and_identity() -> None:
    for name, schema in governance._SCHEMAS.items():
        row = {column: pd.NA for column, _ in schema}
        for column, dtype in schema:
            if dtype == "string":
                row[column] = "safe"
            elif dtype == "Int64":
                row[column] = 0
            elif dtype == "Float64":
                row[column] = 0.0
            elif dtype == "boolean":
                row[column] = True
            elif dtype == "datetime":
                row[column] = datetime(2025, 1, 1)
        frame = governance._frame(name, [row], aware=False)
        assert list(frame.columns) == [column for column, _ in schema]
        expected = [
            "datetime64[ns]" if dtype == "datetime" else dtype for _, dtype in schema
        ]
        assert [str(dtype) for dtype in frame.dtypes] == expected
        assert frame.index.tolist() == [0]


def test_acceptance_ten_table_aware_datetime_schema() -> None:
    for name, schema in governance._SCHEMAS.items():
        if not any(dtype == "datetime" for _, dtype in schema):
            continue
        frame = governance._frame(
            name,
            [
                {
                    column: pd.Timestamp("2025-01-01", tz="UTC")
                    for column, dtype in schema
                    if dtype == "datetime"
                }
            ],
            aware=True,
        )
        for column, dtype in schema:
            if dtype == "datetime":
                assert str(frame[column].dtype) == "datetime64[ns, UTC]"


def test_acceptance_source_registry_38_entry_shape_and_uses() -> None:
    assert len(governance._SOURCE_REGISTRY) == 38
    positions = [position for position, _, _ in governance._SOURCE_REGISTRY]
    assert positions == list(range(1, 39))
    assert len({(task, table) for _, task, table in governance._SOURCE_REGISTRY}) == 38
    allowed = {
        "comparison_criterion",
        "diagnostic",
        "explanation",
        "attribution_context",
        "drift_context",
        "stability_context",
    }
    for position, task, table in governance._SOURCE_REGISTRY:
        assert task in {"task15", "task16", "task17", "task18"}
        assert type(table) is str and table
        assert set(governance._source_uses(position)) <= allowed
        assert governance._source_uses(position)


def test_acceptance_source_registry_deny_matrix() -> None:
    owners = governance._owner_collections((), (), (), ())
    ref = _fold_ref("champion", 0, "diagnostic")
    with pytest.raises(ValueError, match="model governance: invalid_source_binding$"):
        governance._ref_owner(replace(ref, source_result_position=0), owners)
    with pytest.raises(ValueError, match="model governance: unsupported_source$"):
        governance._ref_owner(replace(ref, source_table="not_a_table"), owners)
    with pytest.raises(ValueError, match="model governance: unsupported_source$"):
        governance._ref_owner(replace(ref, source_use="comparison_criterion"), owners)


def test_acceptance_direction_registry_92_and_target_range_inventory() -> None:
    assert len(governance._DIRECTION_REGISTRY) == 92
    assert (
        len(
            {
                (task, table, metric)
                for task, table, metric, _ in governance._DIRECTION_REGISTRY
            }
        )
        == 92
    )
    directions = {direction for *_, direction in governance._DIRECTION_REGISTRY}
    assert directions == {"higher_is_better", "lower_is_better", "target_range"}
    target_range = [
        item for item in governance._DIRECTION_REGISTRY if item[-1] == "target_range"
    ]
    assert len(target_range) == 17


def test_acceptance_error_reason_status_inventories_are_closed() -> None:
    assert len(governance._ERROR_KEYS) == 76
    assert len(set(governance._ERROR_KEYS)) == 76
    assert "invalid_owner_result" not in governance._ERROR_KEYS
    assert "privacy_unsafe_value" not in governance._ERROR_KEYS
    assert all("*" not in key and "{" not in key for key in governance._ERROR_KEYS)
    assert len(governance._REASONS) == 15
    assert len(set(governance._REASONS)) == 15
    assert {
        "available",
        "unavailable",
        "undefined",
        "not_applicable",
        "not_verifiable",
    } <= {
        "available",
        "unavailable",
        "undefined",
        "not_applicable",
        "not_verifiable",
    }


def test_v2_provenance_and_governance_fingerprint_identity() -> None:
    result = evaluate_governance(
        _policy(_candidate("champion", 0, "champion")),
        risk_validations=(_risk_result(),),
    )
    assert result.provenance.loc[0, "provenance_key"] == "contract_version"
    assert result.provenance.loc[0, "provenance_value"] == (
        '{"t":"str","v":"task19-contract-targeted-fixed-v2"}'
    )
    assert len(result.provenance) == 35
    assert len(result.governance_fingerprint) == 64
    assert "task19-contract-targeted-fixed-v1" not in result.provenance.to_csv(
        index=False
    )


def test_v2_invalid_pair_family_mismatch_has_exact_error() -> None:
    champion = _candidate("champion", 0, "champion")
    challenger = GovernanceCandidate(
        "challenger",
        "strategy",
        "task17",
        0,
        "strategy",
        "b" * 64,
        "v1",
        "challenger",
        "candidate",
        (),
    )
    policy = _policy(
        champion,
        challenger,
        pairs=(("champion", "challenger"),),
    )
    owner = simulate_decision_strategy(
        pd.DataFrame({"x": [1]}),
        DecisionStrategyConfig(
            "strategy",
            "v1",
            datetime(2025, 1, 1),
            None,
            datetime(2025, 1, 1),
            (),
            "select",
            "review",
            (("select", "selected"), ("review", "review")),
        ),
    )
    with pytest.raises(ValueError) as caught:
        evaluate_governance(
            policy,
            risk_validations=(_risk_result(),),
            decision_strategies=(owner,),
        )
    assert type(caught.value) is ValueError
    assert str(caught.value) == "model governance: invalid_pair"


def test_v2_duplicate_standalone_evidence_ref_has_exact_error() -> None:
    ref = _fold_ref("champion", 0, "diagnostic")
    policy = replace(
        _policy(_candidate("champion", 0, "champion")),
        evidence_refs=(ref, ref),
    )
    with pytest.raises(ValueError) as caught:
        evaluate_governance(policy, risk_validations=(_risk_result(),))
    assert type(caught.value) is ValueError
    assert str(caught.value) == "model governance: duplicate_evidence_ref"


def test_v2_owner_dtype_status_reason_precedence() -> None:
    owner = _risk_result()
    bad_dtype = owner.metrics.copy(deep=True)
    bad_dtype["value"] = bad_dtype["value"].astype(object)
    bad_dtype.loc[0, "status"] = "bogus"
    bad_dtype.loc[0, "reason"] = "bogus"
    candidate = _candidate("champion", 0, "champion")
    ref = _fold_ref("champion", 0, "diagnostic")
    with pytest.raises(ValueError) as caught:
        evaluate_governance(
            replace(_policy(candidate), evidence_refs=(ref,)),
            risk_validations=(replace(owner, metrics=bad_dtype),),
        )
    assert str(caught.value) == "model governance: invalid_owner_dtype"


def test_v2_owner_status_precedes_reason() -> None:
    owner = _risk_result()
    metrics = owner.metrics.copy(deep=True)
    metrics.loc[0, "status"] = "bogus"
    metrics.loc[0, "reason"] = "bogus"
    ref = _fold_ref("champion", 0, "diagnostic")
    with pytest.raises(ValueError) as caught:
        evaluate_governance(
            replace(
                _policy(_candidate("champion", 0, "champion")),
                evidence_refs=(ref,),
            ),
            risk_validations=(replace(owner, metrics=metrics),),
        )
    assert str(caught.value) == "model governance: invalid_owner_status"


def test_v2_owner_reason_rejects_invalid_status_reason_matrix() -> None:
    owner = _risk_result()
    metrics = owner.metrics.copy(deep=True)
    metrics.loc[0, "status"] = "available"
    metrics.loc[0, "reason"] = "bogus"
    ref = _fold_ref("champion", 0, "diagnostic")
    with pytest.raises(ValueError) as caught:
        evaluate_governance(
            replace(
                _policy(_candidate("champion", 0, "champion")),
                evidence_refs=(ref,),
            ),
            risk_validations=(replace(owner, metrics=metrics),),
        )
    assert str(caught.value) == "model governance: invalid_owner_reason"


def test_v2_source_fingerprint_lexical_and_digest_boundaries() -> None:
    owner = _risk_result()
    candidate = _candidate("champion", 0, "champion")
    bad = replace(candidate, expected_source_fingerprint="not-a-sha")
    with pytest.raises(ValueError) as lexical:
        evaluate_governance(_policy(bad), risk_validations=(owner,))
    assert str(lexical.value) == "model governance: invalid_source_fingerprint"
    ref = replace(
        _fold_ref("champion", 0, "diagnostic"),
        expected_source_fingerprint="a" * 64,
    )
    with pytest.raises(ValueError) as mismatch:
        evaluate_governance(
            replace(_policy(candidate), evidence_refs=(ref,)),
            risk_validations=(owner,),
        )
    assert str(mismatch.value) == "model governance: source_fingerprint_mismatch"


def test_acceptance_canonical_encoder_domain_and_privacy() -> None:
    values = (None, True, 1, 1.5, "秘密", datetime(2025, 1, 1), ("x", 1))
    encoded = governance._canonical_json(values)
    assert "\\u79d8" in encoded
    assert len(governance._fingerprint(values)) == 64
    with pytest.raises(ValueError, match="model governance: invalid_canonical_value$"):
        governance._canonical_json(float("inf"))
    with pytest.raises(ValueError, match="model governance: invalid_canonical_value$"):
        governance._canonical_json([1, 2])


def test_acceptance_repeat_and_deepcopy_determinism() -> None:
    owner = _risk_result()
    policy = _policy(_candidate("champion", 0, "champion"))
    first = evaluate_governance(policy, risk_validations=(owner,))
    second = evaluate_governance(policy, risk_validations=(owner,))
    for name in governance._SCHEMAS:
        pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name))
    assert first.governance_fingerprint == second.governance_fingerprint
    assert first.provenance.equals(second.provenance)


def test_acceptance_successful_call_preserves_owner_and_declarations() -> None:
    owner = _risk_result()
    policy = _policy(_candidate("champion", 0, "champion"))
    owner_before = owner.predictions.copy(deep=True)
    policy_before = policy
    evaluate_governance(policy, risk_validations=(owner,))
    pd.testing.assert_frame_equal(owner.predictions, owner_before)
    assert policy == policy_before


@pytest.mark.parametrize(
    "kind,table_name",
    [
        ("importance", "model_attributions"),
        ("candidate_comparison", "candidate_comparisons"),
        ("prediction_drift", "prediction_drift"),
        ("performance_stability", "performance_stability"),
        ("governance_summary", "governance_summary"),
    ],
)
def test_acceptance_populated_plot_paths_are_result_only(
    kind: str, table_name: str
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    owner = _risk_result()
    result = evaluate_governance(
        _policy(_candidate("champion", 0, "champion")), risk_validations=(owner,)
    )
    table = getattr(result, table_name).copy()
    value_column = {
        "importance": "value",
        "candidate_comparison": "delta",
        "prediction_drift": "prediction_tvd",
        "performance_stability": "delta",
        "governance_summary": "available_criterion_count",
    }[kind]
    if table.empty:
        table = governance._frame(
            table_name,
            [{value_column: 1.0, "status": "available"}],
            aware=False,
        )
    else:
        table.loc[0, "status"] = "available"
        table.loc[0, value_column] = 1.0
    result = replace(result, **{table_name: table})
    figure = plot_model_governance(result, kind=kind)
    assert type(figure) is Figure
    plt.close(figure)


@pytest.mark.parametrize(
    "task,table,metric,direction",
    governance._DIRECTION_REGISTRY,
)
def test_acceptance_direction_registry_92_matching_and_mismatch_paths(
    task: str, table: str, metric: str, direction: str
) -> None:
    champion = _candidate("champion", 0, "champion")
    criterion = GovernanceCriterion(
        "criterion",
        "model",
        task,
        table,
        metric,
        "overall",
        None,
        None,
        "decision",
        True,
        direction,
        0.0 if direction == "target_range" else None,
        1.0 if direction == "target_range" else None,
    )
    policy = _policy(champion, criteria=(criterion,))
    # Zero-challenger evaluation reaches the public assertion branch without
    # requiring source rows for a pair that does not exist.
    accepted = evaluate_governance(policy, risk_validations=(_risk_result(),))
    assert accepted.criterion_count == 1
    mismatch = next(
        item
        for item in ("higher_is_better", "lower_is_better", "target_range")
        if item != direction
    )
    bad = replace(
        criterion,
        direction=mismatch,
        target_low=0.0 if mismatch == "target_range" else None,
        target_high=1.0 if mismatch == "target_range" else None,
    )
    with pytest.raises(ValueError) as rejected:
        evaluate_governance(
            _policy(champion, criteria=(bad,)), risk_validations=(_risk_result(),)
        )
    assert type(rejected.value) is ValueError
    assert str(rejected.value) == "model governance: invalid_criterion"


@pytest.mark.parametrize(
    "ordinal,name,maximum,error",
    [
        (ordinal, name, maximum, error)
        for ordinal, (name, maximum, error) in enumerate(
            governance._RESOURCE_GATES, start=1
        )
    ],
)
def test_acceptance_resource_registry_17_max_and_max_plus_one(
    ordinal: int, name: str, maximum: int, error: str
) -> None:
    assert ordinal in range(1, 18)
    at_max = tuple(
        maximum if item_name == name else 0
        for item_name, _, _ in governance._RESOURCE_GATES
    )
    governance._resource_preflight(at_max)
    above = list(at_max)
    above[ordinal - 1] += 1
    with pytest.raises(ValueError) as caught:
        governance._resource_preflight(tuple(above))
    assert type(caught.value) is ValueError
    assert str(caught.value) == f"model governance: {error}"


def test_acceptance_resource_registry_precedence_and_metadata_gate_13() -> None:
    values = [0] * 17
    values[3] = governance._RESOURCE_GATES[3][1] + 1
    values[16] = governance._RESOURCE_GATES[16][1] + 1
    with pytest.raises(ValueError) as early:
        governance._resource_preflight(tuple(values))
    assert str(early.value) == "model governance: resource_lifecycle_monitoring_results"
    values = [0] * 17
    values[12] = 257
    values[15] = governance._RESOURCE_GATES[15][1] + 1
    with pytest.raises(ValueError) as middle:
        governance._resource_preflight(tuple(values))
    assert str(middle.value) == "model governance: resource_governance_metadata_rows"
    assert governance._RESOURCE_GATES[12] == (
        "governance_metadata_rows",
        256,
        "resource_governance_metadata_rows",
    )


@pytest.mark.parametrize("index", range(9))
def test_acceptance_fixed_invariants_9_defensive_paths(index: int) -> None:
    arguments = {
        "prediction_profiles": 2,
        "source_evidence_rows": 2,
        "evidence_refs": 2,
        "prediction_drift_rows": 1,
        "performance_evidence": 2,
        "performance_stability_rows": 1,
        "pairs": 1,
        "criteria": 1,
        "candidate_comparison_rows": 1,
        "governance_evaluation_rows": 1,
        "recommendation_rows": 1,
        "candidates": 2,
        "governance_summary_rows": 2,
        "provenance_rows": 35,
    }
    governance._validate_fixed_invariants(**arguments)
    mutations = (
        ("prediction_profiles", 3),
        ("source_evidence_rows", 1),
        ("prediction_drift_rows", 0),
        ("performance_stability_rows", 0),
        ("candidate_comparison_rows", 0),
        ("governance_evaluation_rows", 0),
        ("recommendation_rows", 0),
        ("governance_summary_rows", 1),
        ("provenance_rows", 34),
    )
    key, value = mutations[index]
    arguments[key] = value
    with pytest.raises(ValueError) as caught:
        governance._validate_fixed_invariants(**arguments)
    assert (
        str(caught.value)
        == f"model governance: {governance._FIXED_INVARIANTS[index][1]}"
    )


def test_acceptance_deepcopy_and_global_rng_determinism() -> None:
    owner = _risk_result()
    policy = _policy(_candidate("champion", 0, "champion"))
    np.random.seed(871)
    state_before = deepcopy(np.random.get_state())
    first = evaluate_governance(policy, risk_validations=(owner,))
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]
    second = evaluate_governance(deepcopy(policy), risk_validations=(deepcopy(owner),))
    for name in governance._SCHEMAS:
        pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name))
    assert first.governance_fingerprint == second.governance_fingerprint


def test_acceptance_hash_seed_independent_canonical_fingerprint() -> None:
    code = (
        "from sharper.model_governance import _fingerprint;"
        "print(_fingerprint(('z','a',3,True)))"
    )
    outputs = []
    for seed in ("1", "987654"):
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed},
        )
        outputs.append(completed.stdout.strip())
    assert outputs[0] == outputs[1]


def test_acceptance_privacy_scan_all_tables_errors_and_plots() -> None:
    import matplotlib.pyplot as plt

    secrets = (
        "RAW_ENTITY_SECRET_7F92",
        "RAW_SEGMENT_SECRET_4DA1",
        "RAW_COHORT_SECRET_E831",
        "RAW_VINTAGE_SECRET_3AA7",
        "RAW_FEATURE_VALUE_SECRET_0C17",
        "RAW_TARGET_SECRET_991B",
        "MODEL_REPR_SECRET_72D4",
        "PATH_SECRET_1FE8",
    )
    result = evaluate_governance(
        _policy(_candidate("champion", 0, "champion")),
        risk_validations=(_risk_result(),),
    )
    materialized = "".join(
        table.astype("string").to_csv(index=False)
        for table in (getattr(result, name) for name in governance._SCHEMAS)
    )
    for secret in secrets:
        assert secret not in materialized
    invalid = replace(result, governance_key=secrets[0])
    with pytest.raises(ValueError) as caught:
        plot_model_governance(invalid, kind="not_a_kind")  # type: ignore[arg-type]
    assert all(secret not in str(caught.value) for secret in secrets)
    figure = plot_model_governance(result, kind="governance_summary")
    text = "".join(
        item.get_text()
        for item in figure.findobj(match=lambda x: hasattr(x, "get_text"))
    )
    assert all(secret not in text for secret in secrets)
    plt.close(figure)


@pytest.mark.parametrize(
    "expected,kwargs",
    (
        ("promote_challenger", {"champion_value": 0.4, "challenger_value": 0.8}),
        ("retain_champion", {"champion_value": 0.8, "challenger_value": 0.4}),
        ("continue_evaluation", {"champion_value": 0.5, "challenger_value": 0.5}),
        (
            "insufficient_evidence",
            {"champion_value": 0.4, "challenger_value": 0.8, "unavailable": True},
        ),
        (
            "reject_challenger",
            {
                "champion_value": 0.4,
                "challenger_value": 0.8,
                "challenger_state": "rejected",
            },
        ),
    ),
)
def test_acceptance_five_recommendations_public_reachability(
    expected: str, kwargs: dict[str, object]
) -> None:
    result = _comparison_output(**kwargs)  # type: ignore[arg-type]
    assert result.recommendations.loc[0, "recommendation"] == expected
    assert result.recommendations.loc[0, "status"] == "available"
    assert pd.isna(result.recommendations.loc[0, "reason"])


@pytest.mark.parametrize(
    "expected,kwargs",
    (
        (
            "continue_evaluation",
            {
                "champion_value": 0.4,
                "challenger_value": 0.8,
                "second_values": (0.2, 0.4),
            },
        ),
        (
            "insufficient_evidence",
            {"champion_value": 0.4, "challenger_value": 0.8, "support_mismatch": True},
        ),
        (
            "insufficient_evidence",
            {
                "champion_value": 0.4,
                "challenger_value": 0.8,
                "second_values": (0.4, 0.2),
                "unavailable": True,
                "required": False,
                "minimum": 2,
            },
        ),
        (
            "reject_challenger",
            {
                "champion_value": 0.4,
                "challenger_value": 0.8,
                "challenger_state": "retired",
            },
        ),
    ),
)
def test_acceptance_recommendation_blocker_matrix(
    expected: str, kwargs: dict[str, object]
) -> None:
    result = _comparison_output(**kwargs)  # type: ignore[arg-type]
    assert result.recommendations.loc[0, "recommendation"] == expected


def test_acceptance_optional_unavailable_does_not_force_insufficient() -> None:
    result = _comparison_output(
        0.8,
        0.4,
        second_values=(0.4, 0.2),
        unavailable=True,
        required=False,
    )
    assert result.recommendations.loc[0, "recommendation"] == "promote_challenger"


def _source_case(
    position: int,
    task: str,
    table: str,
    ref: dict[str, object],
    row: dict[str, object],
    field: str,
    value: object,
    support: object = pd.NA,
    status: str | None = None,
) -> dict[str, object]:
    return {
        "position": position,
        "task": task,
        "table": table,
        "ref": ref,
        "row": row,
        "field": field,
        "value": value,
        "support": support,
        "status": status,
    }


_REAL_SOURCE_CASES = (
    _source_case(
        1,
        "task15",
        "metrics",
        dict(
            use="comparison_criterion",
            candidate_key="candidate",
            metric_key="roc_auc",
            scope_key="fold",
            fold_id=0,
            statistic_key="direct",
        ),
        dict(scope="fold", fold_id=0, metric="roc_auc", statistic="direct"),
        "value",
        0.314159,
        17,
    ),
    _source_case(
        2,
        "task15",
        "gains",
        dict(
            use="comparison_criterion",
            candidate_key="candidate",
            metric_key="event_rate",
            scope_key="fold",
            fold_id=0,
            numeric_value=0.5,
        ),
        dict(scope="fold", fold_id=0, requested_fraction=0.5),
        "event_rate",
        0.314159,
        17,
    ),
    _source_case(
        3,
        "task15",
        "threshold_analysis",
        dict(
            use="comparison_criterion",
            candidate_key="candidate",
            metric_key="sensitivity",
            scope_key="fold",
            fold_id=0,
            category_key="probability",
            numeric_value=0.5,
        ),
        dict(scope="fold", fold_id=0, threshold_kind="probability", threshold=0.5),
        "sensitivity",
        0.314159,
    ),
    _source_case(
        4,
        "task15",
        "business_metrics",
        dict(
            use="comparison_criterion",
            candidate_key="candidate",
            metric_key="observed_loss_sum",
            scope_key="segment",
            numeric_value=0.5,
        ),
        dict(segment_kind="segment", segment_value=0.5, metric="observed_loss_sum"),
        "value",
        0.314159,
        17,
    ),
    _source_case(
        5,
        "task15",
        "folds",
        dict(use="diagnostic", candidate_key=None, fold_id=0),
        dict(fold_id=0),
        "train_n",
        17,
    ),
    _source_case(
        6,
        "task16",
        "dataset_profile",
        dict(use="diagnostic", candidate_key=None, side_key="current"),
        dict(side="current"),
        "n_rows",
        17,
        17,
    ),
    _source_case(
        7,
        "task16",
        "column_profile",
        dict(use="diagnostic", candidate_key=None, side_key="current", column_key="x"),
        dict(side="current", column="x"),
        "non_missing_count",
        17,
        17,
    ),
    _source_case(
        8,
        "task16",
        "numeric_profile",
        dict(use="diagnostic", candidate_key=None, side_key="current", column_key="x"),
        dict(side="current", column="x"),
        "mean",
        0.314159,
        17,
    ),
    _source_case(
        9,
        "task16",
        "categorical_profile",
        dict(
            use="diagnostic",
            candidate_key=None,
            side_key="current",
            column_key="category",
        ),
        dict(side="current", column="category"),
        "unique_count",
        17,
    ),
    _source_case(
        10,
        "task16",
        "missingness_drift",
        dict(use="diagnostic", candidate_key=None, column_key="x"),
        dict(column="x"),
        "absolute_rate_change",
        0.314159,
    ),
    _source_case(
        11,
        "task16",
        "point_in_time_profile",
        dict(
            use="diagnostic",
            candidate_key=None,
            side_key="current",
            scope_key="feature",
            column_key="x",
        ),
        dict(side="current", scope="feature", column="x"),
        "evaluated_count",
        17,
    ),
    _source_case(
        12,
        "task16",
        "slice_profile",
        dict(
            use="diagnostic",
            candidate_key=None,
            side_key="current",
            slice_role="partition",
            row_kind="data",
            scope_position=0,
        ),
        dict(side="current", slice_role="partition", row_kind="data", slice_ordinal=0),
        "row_count",
        17,
        17,
    ),
    _source_case(
        13,
        "task16",
        "resource_usage",
        dict(
            use="diagnostic",
            candidate_key=None,
            side_key="current",
            category_key="memory",
        ),
        dict(side="current", resource="memory"),
        "requested",
        17,
    ),
    _source_case(
        14,
        "task16",
        "findings",
        dict(use="diagnostic", candidate_key=None, finding_key="finding:source"),
        dict(finding_key="finding:source"),
        "value",
        0.314159,
    ),
    _source_case(
        15,
        "task16",
        "provenance",
        dict(use="diagnostic", candidate_key=None, provenance_key="config_fingerprint"),
        dict(
            provenance_key="config_fingerprint",
            value_type="text",
            text_value="sentinel",
        ),
        "text_value",
        "sentinel",
    ),
    _source_case(
        16,
        "task17",
        "row_decisions",
        dict(use="explanation", candidate_key="candidate", row_position=0),
        dict(row_position=0, final_action_name="selected"),
        "final_action_name",
        "selected",
    ),
    _source_case(
        17,
        "task17",
        "rule_evaluations",
        dict(
            use="explanation",
            candidate_key="candidate",
            row_position=0,
            rule_key="rule",
        ),
        dict(row_position=0, rule_key="rule", truth="true"),
        "truth",
        "true",
    ),
    _source_case(
        18,
        "task17",
        "rule_summary",
        dict(
            use="comparison_criterion",
            candidate_key="candidate",
            scope_key="overall",
            scope_position=0,
            time_position=0,
            phase_key="decision",
            rule_key="rule",
            metric_key="action_count",
        ),
        dict(
            scope_type="overall",
            scope_ordinal=0,
            time_slice_ordinal=0,
            phase="decision",
            rule_key="rule",
            metric_key="action_count",
        ),
        "metric_value",
        0.314159,
        17,
    ),
    _source_case(
        19,
        "task17",
        "action_summary",
        dict(
            use="comparison_criterion",
            candidate_key="candidate",
            scope_key="overall",
            scope_position=0,
            time_position=0,
            action_key="select",
            metric_key="action_count",
        ),
        dict(
            scope_type="overall",
            scope_ordinal=0,
            time_slice_ordinal=0,
            action_key="select",
            metric_key="action_count",
        ),
        "metric_value",
        0.314159,
        17,
    ),
    _source_case(
        20,
        "task17",
        "business_summary",
        dict(
            use="comparison_criterion",
            candidate_key="candidate",
            scope_key="overall",
            scope_position=0,
            time_position=0,
            action_key="select",
            action_role="selected",
            metric_key="selected_rate",
        ),
        dict(
            scope_type="overall",
            scope_ordinal=0,
            time_slice_ordinal=0,
            action_key="select",
            action_role="selected",
            metric_key="selected_rate",
        ),
        "metric_value",
        0.314159,
        17,
    ),
    _source_case(
        21,
        "task17",
        "constraint_summary",
        dict(
            use="explanation",
            candidate_key="candidate",
            constraint_key="limit",
            metric_key="review_rate",
        ),
        dict(constraint_key="limit", metric="review_rate"),
        "actual_value",
        0.314159,
        17,
    ),
    _source_case(
        22,
        "task17",
        "historical_transitions",
        dict(
            use="explanation",
            candidate_key="candidate",
            category_key="old",
            action_key="new",
        ),
        dict(historical_action_name="old", simulated_action_name="new"),
        "row_count",
        17,
        17,
    ),
    _source_case(
        23,
        "task17",
        "provenance",
        dict(
            use="explanation",
            candidate_key="candidate",
            provenance_key="evaluation_time",
        ),
        dict(
            provenance_key="evaluation_time",
            provenance_value='{"t":"datetime","v":"2025-01-01T00:00:00"}',
        ),
        "provenance_value",
        '{"t":"datetime","v":"2025-01-01T00:00:00"}',
    ),
    _source_case(
        24,
        "task18",
        "monitoring_summary",
        dict(
            use="comparison_criterion",
            candidate_key="candidate",
            scenario_key="reference",
            scope_key="scenario",
            scope_position=0,
            metric_key="warning_hit_count",
        ),
        dict(
            scenario_key="reference",
            scope_key="scenario",
            scope_position=0,
            metric="warning_hit_count",
        ),
        "metric_value",
        0.314159,
        17,
    ),
    _source_case(
        25,
        "task18",
        "scenario_comparison",
        dict(
            use="explanation",
            candidate_key="candidate",
            reference_scenario_key="reference",
            comparator_scenario_key="challenger",
            scope_key="scenario",
            scope_position=0,
            metric_key="warning_hit_count",
        ),
        dict(
            reference_scenario_key="reference",
            comparator_scenario_key="challenger",
            scope_key="scenario",
            scope_position=0,
            metric="warning_hit_count",
        ),
        "reference_value",
        0.314159,
        17,
    ),
    _source_case(
        26,
        "task18",
        "lifecycle_summary",
        dict(
            use="explanation",
            candidate_key="candidate",
            scope_key="overall",
            scope_position=0,
            from_state_key="from",
            to_state_key="to",
            metric_key="resolved_episode_count",
        ),
        dict(
            scope_key="overall",
            scope_position=0,
            from_state_key="from",
            to_state_key="to",
            metric="resolved_episode_count",
        ),
        "metric_value",
        0.314159,
        17,
    ),
    _source_case(
        27,
        "task18",
        "rule_evaluations",
        dict(
            use="explanation",
            candidate_key="candidate",
            row_position=0,
            scenario_key="reference",
            rule_key="rule",
        ),
        dict(row_position=0, scenario_key="reference", rule_key="rule", truth="true"),
        "truth",
        "true",
    ),
    _source_case(
        28,
        "task18",
        "alert_episodes",
        dict(
            use="explanation",
            candidate_key="candidate",
            entity_position=0,
            scenario_key="reference",
            rule_key="rule",
            episode_ordinal=0,
        ),
        dict(
            entity_position=0,
            scenario_key="reference",
            rule_key="rule",
            episode_ordinal=0,
        ),
        "raw_hit_count",
        17,
    ),
    _source_case(
        29,
        "task18",
        "state_history",
        dict(use="explanation", candidate_key="candidate", row_position=0),
        dict(row_position=0),
        "effective_state_key",
        "current",
    ),
    _source_case(
        30,
        "task18",
        "state_transitions",
        dict(
            use="explanation",
            candidate_key="candidate",
            from_row_position=0,
            to_row_position=1,
        ),
        dict(from_row_position=0, to_row_position=1, transition_kind="escalation"),
        "transition_kind",
        "escalation",
    ),
    _source_case(
        31,
        "task18",
        "provenance",
        dict(use="diagnostic", candidate_key=None, provenance_key="analysis_as_of"),
        dict(
            provenance_key="analysis_as_of",
            provenance_value='{"t":"datetime","v":"2025-01-01T00:00:00"}',
        ),
        "provenance_value",
        '{"t":"datetime","v":"2025-01-01T00:00:00"}',
    ),
    _source_case(
        32,
        "task16",
        "target_profile",
        dict(
            use="diagnostic",
            candidate_key=None,
            side_key="current",
            category_position=0,
        ),
        dict(side="current", class_position=0),
        "count",
        17,
    ),
    _source_case(
        33,
        "task16",
        "missingness_patterns",
        dict(use="diagnostic", candidate_key=None, pattern_key="p:"),
        dict(pattern_key="p:"),
        "row_count",
        17,
        17,
    ),
    _source_case(
        34,
        "task16",
        "schema_drift",
        dict(use="diagnostic", candidate_key=None, column_key="x"),
        dict(column="x", primary_change="unchanged"),
        "primary_change",
        "unchanged",
    ),
    _source_case(
        35,
        "task16",
        "collinearity",
        dict(use="diagnostic", candidate_key=None, column_key="x", scope_column="y"),
        dict(left_column="x", right_column="y"),
        "pearson_r",
        0.314159,
        17,
    ),
    _source_case(
        36,
        "task18",
        "observation_history",
        dict(use="explanation", candidate_key="candidate", row_position=0),
        dict(row_position=0),
        "active_rule_count",
        17,
    ),
    _source_case(
        37,
        "task18",
        "notifications",
        dict(
            use="explanation",
            candidate_key="candidate",
            entity_position=0,
            scenario_key="reference",
            rule_key="rule",
            episode_ordinal=0,
            notification_ordinal=0,
        ),
        dict(
            entity_position=0,
            scenario_key="reference",
            rule_key="rule",
            episode_ordinal=0,
            notification_ordinal=0,
        ),
        "notification_kind",
        "notification",
    ),
    _source_case(
        38,
        "task18",
        "event_matches",
        dict(
            use="explanation",
            candidate_key="candidate",
            scenario_key="reference",
            entity_position=0,
            event_ordinal=0,
        ),
        dict(scenario_key="reference", entity_position=0, event_ordinal=0),
        "captured",
        True,
    ),
)

for _case in _REAL_SOURCE_CASES:
    if _case["position"] in {1, 2, 3, 7, 8, 9, 10, 12, 16, 29, 36}:
        _case["status"] = "not_verifiable"


def _task17_owner_for_matrix() -> object:
    condition = StrategyCondition("atomic", "ge", "column", "x", "literal", 0)
    config = DecisionStrategyConfig(
        "strategy",
        "v1",
        datetime(2025, 1, 1),
        None,
        datetime(2025, 1, 2),
        (DecisionRule("rule", "decision", 0, condition, "select"),),
        "select",
        "review",
        (("select", "selected"), ("review", "review")),
    )
    return simulate_decision_strategy(pd.DataFrame({"x": [0, 1, 2, 3]}), config)


def _task18_owner_for_matrix() -> object:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "task18_source_matrix_tests", "tests/test_lifecycle_monitoring.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.monitor_lifecycle(module._frame(), module._config())


@cache
def _real_base_owner(task: str) -> object:
    if task == "task15":
        return _risk_result()
    if task == "task16":
        return audit_data_quality(
            pd.DataFrame({"x": [1.0, 2.0, 3.0], "target": [0, 1, 1]})
        )
    if task == "task17":
        return _task17_owner_for_matrix()
    return _task18_owner_for_matrix()


def _source_schema_frame(
    base: pd.DataFrame, overrides: dict[str, object], *, position: int
) -> pd.DataFrame:
    values: dict[str, object] = {}
    for column in base.columns:
        if column in overrides:
            values[column] = overrides[column]
        elif column == "finding_key":
            values[column] = f"finding:matrix:{position}"
        elif column == "provenance_value":
            values[column] = "2025-01-01T00:00:00"
        elif column == "value_type":
            values[column] = "text"
        elif column.endswith("_status") or column == "status":
            values[column] = "available"
        elif column.endswith("_reason") or column == "reason":
            values[column] = pd.NA
        elif column in {"metric", "metric_key"}:
            values[column] = "matrix_metric"
        elif column in {"unit", "support_unit"}:
            values[column] = "rows"
        elif column in {"scope", "scope_type", "scope_key"}:
            values[column] = "overall"
        elif column == "side":
            values[column] = "current"
        elif column == "scope_column":
            values[column] = pd.NA
        elif column.endswith("_time") or column in {
            "cutoff",
            "validation_start",
            "validation_end",
            "analysis_as_of",
        }:
            values[column] = datetime(2025, 1, 1)
        elif column.startswith("is_") or column in {
            "upper_inclusive",
            "aggregated",
            "captured",
            "truncated",
            "constant",
            "near_constant",
            "high_cardinality",
            "suspected_identifier",
            "all_missing",
            "new_all_missing",
            "recovered",
            "override_applied",
            "is_applied",
            "is_overlap",
            "is_conflict",
            "is_repeated",
            "is_allowed",
            "is_consecutive",
            "is_cure",
        }:
            values[column] = False
        elif str(base[column].dtype) == "object":
            values[column] = ()
        elif str(base[column].dtype).startswith(("Float", "float")):
            values[column] = 0.271828
        else:
            values[column] = 0
    frame = pd.DataFrame({column: [values[column]] for column in base.columns})
    for column in base.columns:
        frame[column] = frame[column].astype(base[column].dtype)
    return frame


def _real_owner_case(
    case: dict[str, object],
) -> tuple[object, GovernanceEvidenceRef, object]:
    task = case["task"]
    table_name = case["table"]
    base_owner = _real_base_owner(task)
    base_table = getattr(base_owner, table_name)
    overrides = dict(case["row"])
    overrides[case["field"]] = case["value"]
    if case.get("status") is not None:
        status_column = f"{case['field']}_status"
        status_column = {
            ("dataset_profile", "n_rows"): "feature_status",
            ("column_profile", "non_missing_count"): "missing_status",
            ("numeric_profile", "mean"): "location_status",
            ("categorical_profile", "unique_count"): "cardinality_status",
            ("missingness_drift", "absolute_rate_change"): "absolute_change_status",
            ("slice_profile", "row_count"): "size_status",
            ("target_profile", "count"): "class_status",
            ("missingness_patterns", "row_count"): "count_status",
            ("row_decisions", "final_action_name"): "decision_status",
            ("observation_history", "active_rule_count"): "observation_status",
        }.get((table_name, case["field"]), status_column)
        if status_column not in base_table.columns and "status" in base_table.columns:
            status_column = "status"
        if status_column in base_table.columns:
            overrides[status_column] = case["status"]
    if task == "task15" and table_name == "folds":
        overrides["analysis_as_of"] = pd.NaT
    support = case["support"]
    if support is not pd.NA and not pd.isna(support):
        for support_column in (
            "support_n",
            "support_n_rows",
            "n_rows",
            "n_evaluable_rows",
            "selected_n",
            "valid_n",
            "row_count",
        ):
            if support_column in base_table.columns:
                overrides[support_column] = support
                break
    row = _source_schema_frame(base_table, overrides, position=case["position"])
    owner = replace(base_owner, **{table_name: row})
    fingerprint = governance._owner_fingerprint(task, owner)
    ref_data = dict(case["ref"])
    use = ref_data.pop("use")
    candidate_key = ref_data.pop("candidate_key")
    ref = GovernanceEvidenceRef(
        source_task=task,
        source_result_position=0,
        source_table=table_name,
        source_use=use,
        candidate_key=candidate_key,
        expected_source_fingerprint=fingerprint,
        field_key=case["field"],
        **ref_data,
    )
    return owner, ref, case["value"]


@pytest.mark.parametrize(
    "case",
    _REAL_SOURCE_CASES,
    ids=lambda case: f"{case['position']:02d}-{case['task']}-{case['table']}",
)
def test_acceptance_source_registry_38_real_owner_schema_locator_extractor_allow(
    case: dict[str, object],
) -> None:
    owner, ref, expected_value = _real_owner_case(case)
    expected_type = {
        "task15": BinaryRiskValidationResult,
        "task16": DataAuditResult,
        "task17": DecisionStrategyResult,
        "task18": LifecycleMonitoringResult,
    }[case["task"]]
    assert type(owner) is expected_type
    table = getattr(owner, case["table"])
    base_table = getattr(_real_base_owner(case["task"]), case["table"])
    assert list(table.columns) == list(base_table.columns)
    assert all(
        table[column].dtype == base_table[column].dtype for column in table.columns
    )
    owners = {key: () for key in ("task15", "task16", "task17", "task18")}
    owners[case["task"]] = (owner,)
    resolved = governance._resolve_ref(ref, owners, case["position"] - 1)
    assert resolved["registry_position"] == case["position"]
    assert resolved["status"] == (case["status"] or "available")
    assert pd.isna(resolved["reason"])
    assert resolved["value"] == expected_value
    support = case["support"]
    if support is pd.NA or pd.isna(support):
        assert pd.isna(resolved["support"])
    else:
        assert resolved["support"] == support


def _resolved_real_case(
    position: int,
) -> tuple[object, GovernanceEvidenceRef, dict[str, tuple[object, ...]]]:
    case = next(item for item in _REAL_SOURCE_CASES if item["position"] == position)
    owner, ref, _ = _real_owner_case(case)
    owners = {key: () for key in ("task15", "task16", "task17", "task18")}
    owners[case["task"]] = (owner,)
    return owner, ref, owners


def test_acceptance_source_registry_real_owner_coverage_sets_are_exact() -> None:
    expected = {(task, table) for _, task, table in governance._SOURCE_REGISTRY}
    actual = {(case["task"], case["table"]) for case in _REAL_SOURCE_CASES}
    assert len(_REAL_SOURCE_CASES) == 38
    assert len(actual) == 38
    assert actual == expected
    assert {case["position"] for case in _REAL_SOURCE_CASES} == set(range(1, 39))


def test_acceptance_source_registry_locator_deny_matrix() -> None:
    owner, ref, owners = _resolved_real_case(1)
    with pytest.raises(ValueError, match="model governance: invalid_source_locator$"):
        governance._resolve_ref(replace(ref, fold_id=None), owners, 0)
    with pytest.raises(ValueError, match="model governance: invalid_source_locator$"):
        governance._resolve_ref(replace(ref, row_position=0), owners, 0)
    with pytest.raises(ValueError, match="model governance: invalid_source_locator$"):
        governance._resolve_ref(replace(ref, scope_column="x"), owners, 0)
    with pytest.raises(ValueError, match="model governance: source_not_found$"):
        governance._resolve_ref(replace(ref, fold_id=99), owners, 0)
    duplicate = pd.concat(
        [owner.metrics, owner.metrics.iloc[[0]].copy(deep=True)], ignore_index=True
    )
    duplicate_owner = replace(owner, metrics=duplicate)
    duplicate_owners = {key: () for key in ("task15", "task16", "task17", "task18")}
    duplicate_owners["task15"] = (duplicate_owner,)
    with pytest.raises(ValueError, match="model governance: source_not_unique$"):
        governance._resolve_ref(ref, duplicate_owners, 0)


@pytest.mark.parametrize(
    "case",
    _REAL_SOURCE_CASES,
    ids=lambda case: f"missing-{case['position']:02d}",
)
def test_acceptance_source_registry_every_shape_rejects_missing_required_locator(
    case: dict[str, object],
) -> None:
    _, ref, owners = _resolved_real_case(case["position"])
    required = governance._REQUIRED_REF_FIELDS[case["position"]]
    bad_ref = replace(ref, **{required[0]: None})
    with pytest.raises(ValueError, match="model governance: invalid_source_locator$"):
        governance._resolve_ref(bad_ref, owners, case["position"] - 1)


def _forbidden_locator_field(
    case: dict[str, object], ref: GovernanceEvidenceRef, table: pd.DataFrame
) -> str:
    required = set(governance._REQUIRED_REF_FIELDS[case["position"]])
    aliases = governance._LOCATORS
    for field_name in (
        "row_position",
        "entity_position",
        "scope_key",
        "side_key",
        "finding_key",
        "provenance_key",
        "numeric_value",
        "metric_key",
        "scenario_key",
    ):
        if field_name in required or getattr(ref, field_name) is not None:
            continue
        columns = aliases.get(field_name, ())
        if not any(column in table.columns for column in columns):
            return field_name
    raise AssertionError(f"no forbidden locator representative for {case['position']}")


@pytest.mark.parametrize(
    "case",
    _REAL_SOURCE_CASES,
    ids=lambda case: f"forbidden-{case['position']:02d}",
)
def test_acceptance_source_registry_every_shape_rejects_forbidden_locator(
    case: dict[str, object],
) -> None:
    owner, ref, owners = _resolved_real_case(case["position"])
    table = getattr(owner, case["table"])
    field_name = _forbidden_locator_field(case, ref, table)
    value = 99 if field_name.endswith("position") else "forbidden"
    with pytest.raises(ValueError, match="model governance: invalid_source_locator$"):
        governance._resolve_ref(
            replace(ref, **{field_name: value}), owners, case["position"] - 1
        )


def test_acceptance_source_registry_wrong_task_table_and_use_are_closed() -> None:
    _, ref, owners = _resolved_real_case(1)
    with pytest.raises(ValueError, match="model governance: unsupported_source$"):
        governance._ref_owner(replace(ref, source_task="task16"), owners)
    with pytest.raises(ValueError, match="model governance: unsupported_source$"):
        governance._ref_owner(replace(ref, source_table="calibration"), owners)
    with pytest.raises(ValueError, match="model governance: unsupported_source$"):
        governance._ref_owner(replace(ref, source_use="diagnostic"), owners)
    _, structured_ref, structured_owners = _resolved_real_case(16)
    with pytest.raises(ValueError, match="model governance: unsupported_source$"):
        governance._ref_owner(
            replace(structured_ref, source_use="diagnostic"), structured_owners
        )

    strategy_candidate = GovernanceCandidate(
        "strategy",
        "strategy",
        "task15",
        0,
        None,
        None,
        "v1",
        "champion",
        "approved",
        (),
    )
    with pytest.raises(ValueError, match="model governance: invalid_source_binding$"):
        evaluate_governance(
            _policy(strategy_candidate), risk_validations=(_risk_result(),)
        )
    model_on_strategy = GovernanceCandidate(
        "model-on-strategy",
        "model",
        "task17",
        0,
        None,
        governance._owner_fingerprint("task17", _real_base_owner("task17")),
        "v1",
        "champion",
        "approved",
        (),
    )
    with pytest.raises(ValueError, match="model governance: invalid_source_binding$"):
        evaluate_governance(
            _policy(model_on_strategy),
            decision_strategies=(_real_base_owner("task17"),),
        )
    warning_on_model = GovernanceCandidate(
        "warning-on-model",
        "warning_scenario",
        "task15",
        0,
        "reference",
        None,
        "v1",
        "champion",
        "approved",
        (),
    )
    with pytest.raises(ValueError, match="model governance: invalid_source_binding$"):
        evaluate_governance(
            _policy(warning_on_model), risk_validations=(_risk_result(),)
        )


@pytest.mark.parametrize("position", (6, 16, 24))
def test_acceptance_source_registry_fingerprint_valid_invalid_mismatch(
    position: int,
) -> None:
    owner, ref, owners = _resolved_real_case(position)
    assert governance._resolve_ref(ref, owners, position - 1)["fingerprint"] is not None
    with pytest.raises(
        ValueError, match="model governance: invalid_source_fingerprint$"
    ):
        governance._resolve_ref(
            replace(ref, expected_source_fingerprint="bad"), owners, position - 1
        )
    with pytest.raises(
        ValueError, match="model governance: source_fingerprint_mismatch$"
    ):
        governance._resolve_ref(
            replace(ref, expected_source_fingerprint="a" * 64), owners, position - 1
        )


def test_acceptance_task15_fingerprint_is_unavailable_not_required() -> None:
    _, ref, owners = _resolved_real_case(1)
    assert governance._owner_fingerprint("task15", owners["task15"][0]) is None
    assert governance._resolve_ref(ref, owners, 0)["fingerprint"] is None


def test_acceptance_task16_config_fingerprint_does_not_upgrade_snapshot_proof() -> None:
    owner, ref, owners = _resolved_real_case(6)
    assert (
        governance._resolve_ref(ref, owners, 5)["fingerprint"]
        == owner.config_fingerprint
    )
    assert governance._owner_time("task16", owner, ref) is None


@pytest.mark.parametrize(
    "case",
    _REAL_SOURCE_CASES,
    ids=lambda case: f"time-{case['position']:02d}",
)
def test_acceptance_source_registry_authoritative_time_sources(
    case: dict[str, object],
) -> None:
    owner, ref, _ = _real_owner_case(case)
    value = governance._owner_time(case["task"], owner, ref)
    if case["task"] in {"task17", "task18"}:
        assert isinstance(value, datetime)
    else:
        assert value is None


@pytest.mark.parametrize(
    "left_task,left_position,right_task,right_position,required,expected",
    (
        ("task18", 0, "task18", 0, True, ("verified", "verified")),
        ("task18", 0, "task18", 1, True, ("unverified", "unverified")),
        ("task17", 0, "task18", 0, True, ("unverified", "unverified")),
        ("task15", 0, "task15", 1, False, ("unverified", "not_applicable")),
        ("task16", 0, None, None, False, ("not_applicable", "not_applicable")),
    ),
)
def test_acceptance_snapshot_alignment_direct_matrix(
    left_task: str,
    left_position: int,
    right_task: str | None,
    right_position: int | None,
    required: bool,
    expected: tuple[str, str],
) -> None:
    assert (
        governance._proof_status(
            left_task,
            left_position,
            right_task,
            right_position,
            alignment_required=required,
        )
        == expected
    )


def test_acceptance_time_snapshot_alignment_are_independent() -> None:
    result = _comparison_output(0.4, 0.8)
    row = result.candidate_comparisons.iloc[0]
    assert row["champion_time_status"] == "verified"
    assert row["challenger_time_status"] == "verified"
    assert row["source_snapshot_status"] == "not_applicable"
    assert row["entity_alignment_status"] == "not_applicable"


def test_acceptance_global_multi_invalid_precedence_public_paths() -> None:
    owner = _risk_result()
    future = GovernanceAttributionEvidence(
        "champion",
        "coefficient_direction",
        "feature",
        None,
        0.1,
        "positive",
        "not_applicable",
        1,
        None,
        None,
        None,
        datetime(2026, 1, 1),
        _fold_ref("champion", 0, "attribution_context"),
    )
    cases = (
        (
            replace(_policy(_candidate("champion", 0, "champion")), candidates=[]),
            {"risk_validations": (owner,), "model_attributions": (future,)},
            "invalid_candidate_container",
        ),
        (
            _policy(replace(_candidate("champion", 0, "champion"), candidate_key="")),
            {"risk_validations": (owner,), "model_attributions": (future,)},
            "invalid_candidate",
        ),
        (
            replace(
                _policy(_candidate("champion", 0, "champion")),
                comparison_pairs=(("x", "y"),),
            ),
            {"risk_validations": (owner,), "model_attributions": (future,)},
            "invalid_pair_coverage",
        ),
        (
            replace(_policy(_candidate("champion", 0, "champion")), criteria=("bad",)),
            {"risk_validations": (owner,), "model_attributions": (future,)},
            "invalid_criterion_container",
        ),
        (
            _policy(
                replace(
                    _candidate("champion", 0, "champion"),
                    expected_source_fingerprint="a" * 64,
                )
            ),
            {"risk_validations": (owner,), "model_attributions": (future,)},
            "source_fingerprint_mismatch",
        ),
    )
    for policy, kwargs, expected in cases:
        with pytest.raises(ValueError) as caught:
            evaluate_governance(policy, **kwargs)  # type: ignore[arg-type]
        assert str(caught.value) == f"model governance: {expected}"


def test_acceptance_future_precedes_resource_and_resource_precedes_later_math() -> None:
    candidate = _candidate("champion", 0, "champion")
    future = GovernanceAttributionEvidence(
        "champion",
        "coefficient_direction",
        "feature",
        None,
        0.1,
        "positive",
        "not_applicable",
        1,
        None,
        None,
        None,
        datetime(2026, 1, 1),
        _fold_ref("champion", 0, "attribution_context"),
    )
    owners = tuple(_risk_result(i / 1000) for i in range(17))
    with pytest.raises(ValueError) as future_error:
        evaluate_governance(
            _policy(candidate),
            risk_validations=owners,
            model_attributions=(future,),
        )
    assert str(future_error.value) == "model governance: future_evidence_time"
    projection = [0] * 17
    projection[0] = 17
    projection[-1] = 2_000_001
    with pytest.raises(ValueError) as resource_error:
        governance._resource_preflight(tuple(projection))
    assert (
        str(resource_error.value)
        == "model governance: resource_risk_validation_results"
    )


def test_wave1_authoritative_time_missing_precedes_resource_public_path() -> None:
    """A valid owner with a missing authoritative clock fails before resources."""
    owner = _risk_result()
    folds = owner.folds.copy(deep=True)
    folds["analysis_as_of"] = pd.Series([pd.NA] * len(folds), dtype="object")
    missing_time = replace(owner, validation_mode="time_forward", folds=folds)
    candidate = _candidate("champion", 0, "champion")
    # The 17-owner count is a later resource overage; the missing source time
    # must remain the first public error.
    owners = tuple(_risk_result(i / 1000) for i in range(17))
    owners = (missing_time, *owners[1:])
    ref = _fold_ref("champion", 0, "diagnostic")
    with pytest.raises(ValueError) as caught:
        evaluate_governance(
            replace(_policy(candidate), evidence_refs=(ref,)), risk_validations=owners
        )
    assert str(caught.value) == "model governance: authoritative_time_missing"


def test_wave1_authoritative_time_mismatch_precedes_resource_public_path() -> None:
    """A malformed Task 18 authoritative provenance value fails before resources."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "task18_contract_tests", "tests/test_lifecycle_monitoring.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    owner = module.monitor_lifecycle(module._frame(), module._config())
    provenance = owner.provenance.copy(deep=True)
    provenance.loc[
        provenance["provenance_key"] == "analysis_as_of", "provenance_value"
    ] = "not-a-datetime"
    malformed = replace(owner, provenance=provenance)
    candidate = GovernanceCandidate(
        "scenario",
        "warning_scenario",
        "task18",
        0,
        "reference",
        owner.monitoring_fingerprint,
        "v1",
        "champion",
        "approved",
        (),
    )
    with pytest.raises(ValueError) as caught:
        evaluate_governance(
            _policy(candidate),
            lifecycle_monitorings=tuple(
                replace(malformed, provenance=malformed.provenance.copy(deep=True))
                for _ in range(17)
            ),
        )
    assert str(caught.value) == "model governance: authoritative_time_mismatch"


def test_wave1_resource_precedes_later_owner_value_and_recommendation_math() -> None:
    """Resource preflight remains earlier than comparison/value materialization."""
    owners = [_risk_result(i / 1000) for i in range(17)]
    bad_metrics = owners[0].metrics.copy(deep=True)
    selector = (
        bad_metrics["scope"].eq("fold")
        & bad_metrics["fold_id"].eq(0)
        & bad_metrics["metric"].eq("roc_auc")
        & bad_metrics["statistic"].eq("direct")
    )
    bad_metrics.loc[selector, "value"] = np.inf
    owners[0] = replace(owners[0], metrics=bad_metrics)
    ref0 = GovernanceEvidenceRef(
        "task15",
        0,
        "metrics",
        "comparison_criterion",
        "champion",
        None,
        metric_key="roc_auc",
        scope_key="fold",
        fold_id=0,
        statistic_key="direct",
    )
    ref1 = replace(ref0, source_result_position=1, candidate_key="challenger")
    policy = _policy(
        _candidate("champion", 0, "champion", (ref0,)),
        replace(
            _candidate("challenger", 1, "challenger", (ref1,)),
            declared_state="candidate",
        ),
        pairs=(("champion", "challenger"),),
        criteria=(
            GovernanceCriterion(
                "auc",
                "model",
                "task15",
                "metrics",
                "roc_auc",
                "fold",
                None,
                None,
                "decision",
                True,
                "higher_is_better",
                minimum_support=1,
            ),
        ),
    )
    with pytest.raises(ValueError) as caught:
        evaluate_governance(policy, risk_validations=tuple(owners))
    # Owner value validation is an owner-phase gate and therefore precedes
    # resource preflight under the v2 owner error partition.
    assert str(caught.value) == "model governance: invalid_owner_value"


# ---------------------------------------------------------------------------
# Wave 1B: executable 76-error / status / reason acceptance matrix.
# These cases intentionally point at production public paths or shared
# production validation kernels; they never call a test-only error raiser.
# ---------------------------------------------------------------------------


@cache
def _h_base_owner() -> BinaryRiskValidationResult:
    return _risk_result()


def _h_base_policy() -> GovernancePolicy:
    return _policy(_candidate("champion", 0, "champion"))


def _h_pair_case(
    *,
    owners: tuple[BinaryRiskValidationResult, BinaryRiskValidationResult] | None = None,
    aware: bool = False,
) -> tuple[GovernancePolicy, tuple[BinaryRiskValidationResult, ...]]:
    if owners is None:
        owners = (
            _t2_time_task15_owner(aware=aware),
            _t2_time_task15_owner(aware=aware),
        )
    ref0 = GovernanceEvidenceRef(
        "task15",
        0,
        "metrics",
        "comparison_criterion",
        "champion",
        None,
        metric_key="roc_auc",
        scope_key="fold",
        fold_id=0,
        statistic_key="direct",
    )
    ref1 = replace(ref0, source_result_position=1, candidate_key="challenger")
    policy = _policy(
        _candidate("champion", 0, "champion", (ref0,)),
        _candidate("challenger", 1, "challenger", (ref1,)),
        pairs=(("champion", "challenger"),),
        criteria=(
            GovernanceCriterion(
                "auc",
                "model",
                "task15",
                "metrics",
                "roc_auc",
                "fold",
                None,
                None,
                "decision",
                True,
                "higher_is_better",
            ),
        ),
    )
    return policy, owners


def _h_warning_case(
    owner: LifecycleMonitoringResult,
) -> tuple[GovernancePolicy, tuple[LifecycleMonitoringResult, ...]]:
    candidate = GovernanceCandidate(
        "scenario",
        "warning_scenario",
        "task18",
        0,
        "reference",
        owner.monitoring_fingerprint,
        "v1",
        "champion",
        "approved",
        (),
    )
    return _policy(candidate), (owner,)


def _h_task18_owner() -> LifecycleMonitoringResult:
    return _task18_owner_for_matrix()


def _h_task18_case_with_provenance(
    raw: str,
) -> tuple[GovernancePolicy, tuple[LifecycleMonitoringResult, ...]]:
    owner = _h_task18_owner()
    provenance = owner.provenance.copy(deep=True)
    provenance.loc[
        provenance["provenance_key"] == "analysis_as_of", "provenance_value"
    ] = raw
    owner = replace(owner, provenance=provenance)
    return _h_warning_case(owner)


def _h_resource_case(error_key: str) -> None:
    projected = [0] * len(governance._RESOURCE_GATES)
    for index, (_, maximum, key) in enumerate(governance._RESOURCE_GATES):
        if key == error_key:
            projected[index] = maximum + 1
            break
    else:
        raise AssertionError(error_key)
    governance._resource_preflight(tuple(projected))


def _h_fixed_case(error_key: str) -> None:
    values = dict(
        prediction_profiles=0,
        source_evidence_rows=0,
        evidence_refs=0,
        prediction_drift_rows=0,
        performance_evidence=0,
        performance_stability_rows=0,
        pairs=0,
        criteria=0,
        candidate_comparison_rows=0,
        governance_evaluation_rows=0,
        recommendation_rows=0,
        candidates=0,
        governance_summary_rows=0,
        provenance_rows=35,
    )
    if error_key == "resource_prediction_profile_bin_count":
        values["prediction_profiles"] = 1
    elif error_key == "resource_source_evidence_rows":
        values["source_evidence_rows"] = 1
    elif error_key == "resource_prediction_drift_rows":
        values["prediction_profiles"] = 2
        values["prediction_drift_rows"] = 0
    elif error_key == "resource_performance_stability_rows":
        values["performance_evidence"] = 2
        values["performance_stability_rows"] = 0
    elif error_key == "resource_candidate_comparison_rows":
        values.update(pairs=1, criteria=1, candidate_comparison_rows=0)
    elif error_key == "resource_governance_evaluation_rows":
        values.update(
            pairs=1,
            criteria=1,
            candidate_comparison_rows=1,
            governance_evaluation_rows=0,
        )
    elif error_key == "resource_recommendation_rows":
        values.update(pairs=1, recommendation_rows=0)
    elif error_key == "resource_governance_summary_rows":
        values.update(candidates=1, governance_summary_rows=0)
    elif error_key == "resource_provenance_rows":
        values["provenance_rows"] = 34
    else:
        raise AssertionError(error_key)
    governance._validate_fixed_invariants(**values)


def _h_make_attribution(
    *, candidate_key: str = "missing"
) -> GovernanceAttributionEvidence:
    ref = GovernanceEvidenceRef(
        "task15",
        0,
        "metrics",
        "attribution_context",
        candidate_key,
        None,
        metric_key="roc_auc",
        scope_key="fold",
        fold_id=0,
        statistic_key="direct",
    )
    return GovernanceAttributionEvidence(
        candidate_key,
        "coefficient_direction",
        "feature",
        None,
        1.0,
        "positive",
        "not_applicable",
        10,
        None,
        None,
        None,
        datetime(2025, 1, 1),
        ref,
    )


def _h_make_profile(*, candidate_key: str = "missing") -> GovernancePredictionProfile:
    ref = GovernanceEvidenceRef(
        "task15",
        0,
        "metrics",
        "drift_context",
        candidate_key,
        None,
        metric_key="roc_auc",
        scope_key="fold",
        fold_id=0,
        statistic_key="direct",
    )
    return GovernancePredictionProfile(
        candidate_key,
        "reference",
        "reference",
        datetime(2025, 1, 1),
        "ranking_score",
        "fold",
        None,
        "a" * 64,
        tuple(i / 10 for i in range(1, 10)),
        (1,) * 10,
        10,
        0,
        2,
        0,
        ref,
    )


def _h_make_performance(
    *, candidate_key: str = "missing"
) -> GovernancePerformanceEvidence:
    ref = GovernanceEvidenceRef(
        "task15",
        0,
        "metrics",
        "stability_context",
        candidate_key,
        None,
        metric_key="roc_auc",
        scope_key="fold",
        fold_id=0,
        statistic_key="direct",
    )
    return GovernancePerformanceEvidence(
        candidate_key,
        "reference",
        "reference",
        datetime(2025, 1, 1),
        datetime(2025, 1, 2),
        datetime(2025, 1, 3),
        "holdout",
        "fold",
        None,
        (False, True),
        (0.2, 0.8),
        None,
        "randomized",
        "verified",
        2,
        0,
        ref,
    )


def _h_trigger(key: str) -> None:
    """Execute one real production branch for the authoritative error key."""
    owner = _h_base_owner()
    base = _h_base_policy()
    if key == "invalid_policy_type":
        evaluate_governance(object())
    elif key == "invalid_governance_key":
        evaluate_governance(replace(base, governance_key=""), risk_validations=(owner,))
    elif key == "invalid_governance_version":
        evaluate_governance(
            replace(base, governance_version=""), risk_validations=(owner,)
        )
    elif key == "invalid_analysis_as_of":
        evaluate_governance(
            replace(base, analysis_as_of="bad"), risk_validations=(owner,)
        )
    elif key == "datetime_awareness_mismatch":
        owner = _h_task18_owner()
        policy, owners = _h_warning_case(owner)
        evaluate_governance(
            replace(policy, analysis_as_of=pd.Timestamp("2025-01-31", tz="UTC")),
            lifecycle_monitorings=owners,
        )
    elif key == "invalid_candidate_container":
        evaluate_governance(replace(base, candidates=[]), risk_validations=(owner,))
    elif key == "invalid_pair_container":
        evaluate_governance(
            replace(base, comparison_pairs=[]), risk_validations=(owner,)
        )
    elif key == "invalid_criterion_container":
        evaluate_governance(replace(base, criteria=[]), risk_validations=(owner,))
    elif key == "invalid_metadata_container":
        evaluate_governance(replace(base, metadata=[]), risk_validations=(owner,))
    elif key == "invalid_evidence_ref_container":
        evaluate_governance(replace(base, evidence_refs=[]), risk_validations=(owner,))
    elif key == "invalid_explanation_container":
        evaluate_governance(replace(base, explanations=[]), risk_validations=(owner,))
    elif key == "invalid_attribution_container":
        evaluate_governance(base, risk_validations=(owner,), model_attributions=[])
    elif key == "invalid_prediction_profile_container":
        evaluate_governance(base, risk_validations=(owner,), prediction_profiles=[])
    elif key == "invalid_performance_evidence_container":
        evaluate_governance(base, risk_validations=(owner,), performance_evidence=[])
    elif key == "invalid_owner_result_container":
        evaluate_governance(base, risk_validations=[])
    elif key == "invalid_human_review_mode":
        evaluate_governance(
            replace(base, human_review_mode="bad"), risk_validations=(owner,)
        )
    elif key == "invalid_entity_alignment":
        evaluate_governance(
            replace(base, entity_alignment="bad"), risk_validations=(owner,)
        )
    elif key == "invalid_candidate":
        evaluate_governance(
            replace(base, candidates=(replace(base.candidates[0], candidate_key=""),)),
            risk_validations=(owner,),
        )
    elif key == "duplicate_candidate":
        duplicate = replace(
            base.candidates[0], declared_role="challenger", declared_state="candidate"
        )
        evaluate_governance(
            replace(base, candidates=(base.candidates[0], duplicate)),
            risk_validations=(owner,),
        )
    elif key == "invalid_champion":
        evaluate_governance(
            replace(
                base,
                candidates=(
                    replace(
                        base.candidates[0],
                        declared_role="challenger",
                        declared_state="candidate",
                    ),
                ),
            ),
            risk_validations=(owner,),
        )
    elif key == "invalid_pair":
        challenger = GovernanceCandidate(
            "challenger",
            "strategy",
            "task17",
            0,
            None,
            None,
            "v1",
            "challenger",
            "candidate",
            (),
        )
        evaluate_governance(
            replace(
                base,
                candidates=(base.candidates[0], challenger),
                comparison_pairs=(("champion", "challenger"),),
            ),
            risk_validations=(owner,),
        )
    elif key == "invalid_pair_coverage":
        challenger = _candidate("challenger", 0, "challenger")
        evaluate_governance(
            replace(base, candidates=(base.candidates[0], challenger)),
            risk_validations=(owner,),
        )
    elif key == "duplicate_pair":
        policy, owners = _h_pair_case()
        evaluate_governance(
            replace(
                policy,
                comparison_pairs=(
                    ("champion", "challenger"),
                    ("champion", "challenger"),
                ),
            ),
            risk_validations=owners,
        )
    elif key == "invalid_criterion":
        criterion = GovernanceCriterion(
            "bad",
            "model",
            "task15",
            "metrics",
            "roc_auc",
            "fold",
            None,
            None,
            "decision",
            True,
            "higher_is_better",
            minimum_support=0,
        )
        evaluate_governance(
            replace(base, criteria=(criterion,)), risk_validations=(owner,)
        )
    elif key == "duplicate_criterion":
        criterion = GovernanceCriterion(
            "same",
            "model",
            "task15",
            "metrics",
            "roc_auc",
            "fold",
            None,
            None,
            "decision",
            True,
            "higher_is_better",
        )
        evaluate_governance(
            replace(base, criteria=(criterion, replace(criterion))),
            risk_validations=(owner,),
        )
    elif key == "unsupported_criterion":
        criterion = GovernanceCriterion(
            "bad",
            "model",
            "task15",
            "metrics",
            "not_a_metric",
            "fold",
            None,
            None,
            "decision",
            True,
            "higher_is_better",
        )
        evaluate_governance(
            replace(base, criteria=(criterion,)), risk_validations=(owner,)
        )
    elif key == "invalid_metadata":
        evaluate_governance(
            replace(base, metadata=(replace(base.metadata[0], owner_key=""),)),
            risk_validations=(owner,),
        )
    elif key == "invalid_evidence_ref":
        ref = _fold_ref("champion", 0, "diagnostic")
        evaluate_governance(
            replace(base, evidence_refs=(replace(ref, source_task="bad"),)),
            risk_validations=(owner,),
        )
    elif key == "duplicate_evidence_ref":
        ref = _fold_ref("champion", 0, "diagnostic")
        evaluate_governance(
            replace(base, evidence_refs=(ref, ref)), risk_validations=(owner,)
        )
    elif key == "invalid_explanation":
        ref = _fold_ref("champion", 0, "explanation")
        item = GovernanceExplanation(
            "e", "missing", "reason_trace", ref, None, None, 0, "available", None
        )
        evaluate_governance(
            replace(base, explanations=(item,)), risk_validations=(owner,)
        )
    elif key == "invalid_attribution":
        evaluate_governance(
            base, risk_validations=(owner,), model_attributions=(_h_make_attribution(),)
        )
    elif key == "invalid_prediction_profile":
        evaluate_governance(
            base, risk_validations=(owner,), prediction_profiles=(_h_make_profile(),)
        )
    elif key == "invalid_performance_evidence":
        evaluate_governance(
            base,
            risk_validations=(owner,),
            performance_evidence=(_h_make_performance(),),
        )
    elif key == "duplicate_owner_source":
        evaluate_governance(base, risk_validations=(owner, owner))
    elif key == "invalid_source_binding":
        candidate = replace(
            base.candidates[0], candidate_family="strategy", source_task="task15"
        )
        evaluate_governance(
            replace(base, candidates=(candidate,)), risk_validations=(owner,)
        )
    elif key == "unsupported_source":
        ref = _fold_ref("champion", 0, "diagnostic")
        evaluate_governance(
            replace(base, evidence_refs=(replace(ref, source_table="unknown"),)),
            risk_validations=(owner,),
        )
    elif key in {"invalid_source_locator", "source_not_found"}:
        ref = _fold_ref("champion", 0, "diagnostic")
        bad = (
            replace(ref, fold_id=None)
            if key == "invalid_source_locator"
            else replace(ref, fold_id=99)
        )
        evaluate_governance(
            replace(base, evidence_refs=(bad,)), risk_validations=(owner,)
        )
    elif key == "source_not_unique":
        metrics = pd.concat(
            [owner.metrics, owner.metrics.iloc[[0]].copy(deep=True)], ignore_index=True
        )
        duplicate_owner = replace(owner, metrics=metrics)
        ref = GovernanceEvidenceRef(
            "task15",
            0,
            "metrics",
            "explanation",
            None,
            None,
            metric_key="roc_auc",
            scope_key="fold",
            fold_id=0,
            statistic_key="direct",
        )
        evaluate_governance(
            replace(base, evidence_refs=(ref,)), risk_validations=(duplicate_owner,)
        )
    elif key == "invalid_owner_schema":
        ref = GovernanceEvidenceRef(
            "task15",
            0,
            "metrics",
            "explanation",
            None,
            None,
            metric_key="roc_auc",
            scope_key="fold",
            fold_id=0,
            statistic_key="direct",
        )
        evaluate_governance(
            replace(base, evidence_refs=(ref,)),
            risk_validations=(replace(owner, metrics=None),),
        )
    elif key == "invalid_owner_dtype":
        metrics = owner.metrics.copy(deep=True)
        metrics["value"] = metrics["value"].astype(object)
        evaluate_governance(base, risk_validations=(replace(owner, metrics=metrics),))
    elif key == "invalid_owner_status":
        metrics = owner.metrics.copy(deep=True)
        metrics.loc[0, "status"] = "bogus"
        evaluate_governance(base, risk_validations=(replace(owner, metrics=metrics),))
    elif key == "invalid_owner_reason":
        metrics = owner.metrics.copy(deep=True)
        metrics.loc[0, "reason"] = "bogus"
        evaluate_governance(base, risk_validations=(replace(owner, metrics=metrics),))
    elif key == "invalid_owner_value":
        policy, owners = _h_pair_case()
        metrics = owners[0].metrics.copy(deep=True)
        metrics.loc[0, "value"] = np.inf
        evaluate_governance(
            policy, risk_validations=(replace(owners[0], metrics=metrics), owners[1])
        )
    elif key == "invalid_source_fingerprint":
        candidate = replace(base.candidates[0], expected_source_fingerprint="bad")
        evaluate_governance(
            replace(base, candidates=(candidate,)), risk_validations=(owner,)
        )
    elif key == "source_fingerprint_mismatch":
        candidate = replace(base.candidates[0], expected_source_fingerprint="a" * 64)
        evaluate_governance(
            replace(base, candidates=(candidate,)), risk_validations=(owner,)
        )
    elif key == "authoritative_time_missing":
        policy, owners = _h_warning_case(_h_task18_owner())
        provenance = (
            owners[0]
            .provenance.loc[owners[0].provenance["provenance_key"] != "analysis_as_of"]
            .copy(deep=True)
        )
        evaluate_governance(
            policy, lifecycle_monitorings=(replace(owners[0], provenance=provenance),)
        )
    elif key == "authoritative_time_mismatch":
        policy, owners = _h_task18_case_with_provenance("not-a-datetime")
        evaluate_governance(policy, lifecycle_monitorings=owners)
    elif key == "future_evidence_time":
        policy, owners = _h_task18_case_with_provenance(
            '{"__datetime__":"2030-01-01T00:00:00.000000000"}'
        )
        evaluate_governance(policy, lifecycle_monitorings=owners)
    elif key == "invalid_canonical_value":
        candidate = replace(base.candidates[0], version=object())
        evaluate_governance(
            replace(base, candidates=(candidate,)), risk_validations=(owner,)
        )
    elif key.startswith("resource_"):
        if key in {x[2] for x in governance._RESOURCE_GATES}:
            _h_resource_case(key)
        else:
            _h_fixed_case(key)
    else:
        raise AssertionError(key)


_H_SEMANTICS = {
    "invalid_policy_type": "caller object is not GovernancePolicy",
    "invalid_governance_key": "governance_key is empty or unsafe",
    "invalid_governance_version": "governance_version is empty or unsafe",
    "invalid_analysis_as_of": "analysis_as_of has an invalid exact datetime type",
    "datetime_awareness_mismatch": "policy and owner time awareness differ",
    "invalid_candidate_container": "candidates is not an exact typed tuple",
    "invalid_pair_container": "comparison_pairs is not a tuple of string pairs",
    "invalid_criterion_container": "criteria is not an exact typed tuple",
    "invalid_metadata_container": "metadata is not an exact typed tuple",
    "invalid_evidence_ref_container": "policy refs are not an exact typed tuple",
    "invalid_explanation_container": "explanations are not an exact typed tuple",
    "invalid_attribution_container": "attributions are not an exact typed tuple",
    "invalid_prediction_profile_container": "profiles are not an exact typed tuple",
    "invalid_performance_evidence_container": "performance evidence is not typed",
    "invalid_owner_result_container": "owner results are not an exact typed tuple",
    "invalid_human_review_mode": "human_review_mode is outside its vocabulary",
    "invalid_entity_alignment": "entity_alignment is outside its vocabulary",
    "invalid_candidate": "candidate scalar or challenger state is invalid",
    "duplicate_candidate": "candidate identity is declared twice",
    "invalid_champion": "approved champion inventory is invalid",
    "invalid_pair": "paired candidate families differ",
    "invalid_pair_coverage": "pairs do not cover all challengers",
    "duplicate_pair": "champion/challenger pair is duplicated",
    "invalid_criterion": "criterion scalar or direction predicate is invalid",
    "duplicate_criterion": "criterion identity is declared twice",
    "unsupported_criterion": "criterion metric is absent from direction registry",
    "invalid_metadata": "metadata identity or scalar violates contract",
    "invalid_evidence_ref": "evidence ref shape is invalid",
    "duplicate_evidence_ref": "standalone carrier repeats a canonical ref",
    "invalid_explanation": "explanation binding or source-use is invalid",
    "invalid_attribution": "attribution method or identity is invalid",
    "invalid_prediction_profile": "profile bins or grouping is invalid",
    "invalid_performance_evidence": "performance window or vector is invalid",
    "duplicate_owner_source": "one owner object occurs twice",
    "invalid_source_binding": "candidate/ref binds the wrong owner family",
    "unsupported_source": "task/table/use is outside the source registry",
    "invalid_source_locator": "required or extra locator fields are invalid",
    "source_not_found": "locator resolves to zero owner rows",
    "source_not_unique": "locator resolves to multiple owner rows",
    "invalid_owner_schema": "owner table schema is missing or mismatched",
    "invalid_owner_dtype": "owner numeric field has object dtype",
    "invalid_owner_status": "owner status token is outside vocabulary",
    "invalid_owner_reason": "owner reason token is outside vocabulary",
    "invalid_owner_value": "owner value is non-finite or non-numeric",
    "invalid_source_fingerprint": "expected fingerprint is not lowercase 64-hex",
    "source_fingerprint_mismatch": "expected digest differs from owner digest",
    "authoritative_time_missing": "required owner timestamp is absent",
    "authoritative_time_mismatch": "owner timestamp cannot be parsed",
    "future_evidence_time": "evidence time is later than governance as-of",
    "invalid_canonical_value": "value is outside canonical encoder domain",
}
_H_RESOURCE_SEMANTICS = {
    key: f"resource {key.removeprefix('resource_')} exceeds its frozen bound"
    for key in governance._ERROR_KEYS
    if key.startswith("resource_")
}
_H_SEMANTICS.update(_H_RESOURCE_SEMANTICS)

_H_PHASES = {}
for _key in governance._ERROR_KEYS:
    if _key in {
        "invalid_policy_type",
        "invalid_governance_key",
        "invalid_governance_version",
        "invalid_analysis_as_of",
        "datetime_awareness_mismatch",
    }:
        _H_PHASES[_key] = 1
    elif _key.endswith("_container") or _key in {
        "invalid_human_review_mode",
        "invalid_entity_alignment",
    }:
        _H_PHASES[_key] = 2
    elif _key in {
        "invalid_candidate",
        "duplicate_candidate",
        "invalid_champion",
        "invalid_pair",
        "invalid_pair_coverage",
        "duplicate_pair",
    }:
        _H_PHASES[_key] = 3
    elif _key in {
        "invalid_criterion",
        "duplicate_criterion",
        "unsupported_criterion",
        "invalid_metadata",
        "invalid_evidence_ref",
        "duplicate_evidence_ref",
        "invalid_explanation",
        "invalid_attribution",
        "invalid_prediction_profile",
        "invalid_performance_evidence",
    }:
        _H_PHASES[_key] = 4
    elif _key in {
        "duplicate_owner_source",
        "invalid_source_binding",
        "unsupported_source",
        "invalid_source_locator",
        "source_not_found",
        "source_not_unique",
        "invalid_owner_schema",
        "invalid_owner_dtype",
        "invalid_owner_status",
        "invalid_owner_reason",
        "invalid_owner_value",
        "invalid_source_fingerprint",
        "source_fingerprint_mismatch",
    }:
        _H_PHASES[_key] = 5
    elif _key in {
        "authoritative_time_missing",
        "authoritative_time_mismatch",
        "future_evidence_time",
    }:
        _H_PHASES[_key] = 6
    elif _key == "invalid_canonical_value":
        _H_PHASES[_key] = 7
    else:
        _H_PHASES[_key] = 8

_H_ERROR_CASES = tuple(
    {
        "error_key": key,
        "semantic_condition": _H_SEMANTICS[key],
        "phase": _H_PHASES[key],
        "production_branch": _h_trigger,
        "path_class": "shared_production_kernel"
        if key.startswith("resource_")
        else "evaluate_governance_public",
    }
    for key in governance._ERROR_KEYS
)


def test_task19_gap_h_error_mapping_is_authoritative_and_complete() -> None:
    keys = tuple(case["error_key"] for case in _H_ERROR_CASES)
    assert len(keys) == 76
    assert len(set(keys)) == 76
    assert set(keys) == set(governance._ERROR_KEYS)
    assert {"invalid_owner_result", "privacy_unsafe_value"}.isdisjoint(keys)
    assert all(case["semantic_condition"] and case["phase"] for case in _H_ERROR_CASES)


@pytest.mark.parametrize("case", _H_ERROR_CASES, ids=lambda case: case["error_key"])
def test_task19_gap_h_each_error_executes_exact_production_branch(
    case: dict[str, object],
) -> None:
    with pytest.raises(ValueError) as caught:
        case["production_branch"](case["error_key"])
    assert type(caught.value) is ValueError
    assert str(caught.value) == f"model governance: {case['error_key']}"


@pytest.mark.parametrize("key", governance._ERROR_KEYS)
def test_task19_gap_h_each_error_has_specific_semantic_mapping(key: str) -> None:
    row = next(case for case in _H_ERROR_CASES if case["error_key"] == key)
    assert row["semantic_condition"] != f"invalid {key}"
    assert row["phase"] is not None


def test_task19_gap_h_removed_error_keys_are_absent_everywhere() -> None:
    removed = {"invalid_owner_result", "privacy_unsafe_value"}
    assert removed.isdisjoint({case["error_key"] for case in _H_ERROR_CASES})
    assert removed.isdisjoint(governance._ERROR_KEYS)


def test_task19_gap_h_production_branch_partition_is_mechanical() -> None:
    public = [
        row
        for row in _H_ERROR_CASES
        if row["path_class"] == "evaluate_governance_public"
    ]
    shared = [
        row for row in _H_ERROR_CASES if row["path_class"] == "shared_production_kernel"
    ]
    assert len(public) == 50
    assert len(shared) == 26
    assert len(public) + len(shared) == 76
    names = governance.evaluate_governance.__code__.co_names
    assert "_resource_preflight" in names
    assert "_validate_fixed_invariants" in names
    assert not any(row["path_class"] == "test_only" for row in _H_ERROR_CASES)


def _h_explanation_result(
    status: str, reason: str | None, *, task16_time: bool = False
) -> GovernanceResult:
    candidate = _candidate("champion", 0, "champion")
    if task16_time:
        owner = _h_clean_task16_owner()
        ref = GovernanceEvidenceRef(
            "task16",
            0,
            "dataset_profile",
            "explanation",
            "champion",
            owner.config_fingerprint,
            side_key="current",
            field_key="n_rows",
        )
        kwargs = {"risk_validations": (_h_base_owner(),), "data_audits": (owner,)}
    else:
        owner = _h_base_owner()
        ref = GovernanceEvidenceRef(
            "task15",
            0,
            "metrics",
            "explanation",
            "champion",
            None,
            metric_key="roc_auc",
            scope_key="fold",
            fold_id=0,
            statistic_key="direct",
        )
        if reason in {
            "source_unavailable",
            "source_undefined",
            "source_not_verifiable",
        }:
            metrics = owner.metrics.copy(deep=True)
            selector = (
                metrics["scope"].eq("fold")
                & metrics["fold_id"].eq(0)
                & metrics["metric"].eq("roc_auc")
                & metrics["statistic"].eq("direct")
            )
            metrics.loc[selector, "status"] = status
            metrics.loc[selector, "reason"] = reason
            owner = replace(owner, metrics=metrics)
        kwargs = {"risk_validations": (owner,)}
    explanation = GovernanceExplanation(
        "explanation",
        "champion",
        "reason_trace",
        ref,
        None,
        None,
        0,
        status,
        reason,
    )
    return evaluate_governance(
        replace(_policy(candidate), explanations=(explanation,)), **kwargs
    )


@cache
def _h_clean_task16_owner() -> DataAuditResult:
    owner = _real_base_owner("task16")
    updates = {}
    for name, table in vars(owner).items():
        if not isinstance(table, pd.DataFrame):
            continue
        clean = table.copy(deep=True)
        for column in clean.columns:
            if column == "reason" or column.endswith("_reason"):
                clean[column] = clean[column].map(
                    lambda value: (
                        value
                        if pd.isna(value)
                        or value in governance._OWNER_REASON_VOCABULARY
                        else "source_not_requested"
                    )
                )
        updates[name] = clean
    return replace(owner, **updates)


def _h_diagnostic_result(
    *, entity_alignment: str = "not_requested"
) -> GovernanceResult:
    policy, owners = _h_pair_case()
    criterion = replace(
        policy.criteria[0],
        criterion_role="diagnostic",
        required_for_promotion=False,
        direction="not_directional",
    )
    return evaluate_governance(
        replace(policy, criteria=(criterion,), entity_alignment=entity_alignment),
        risk_validations=owners,
    )


def _h_profile_status_result() -> GovernanceResult:
    ref0 = GovernanceEvidenceRef(
        "task15",
        0,
        "metrics",
        "drift_context",
        "champion",
        None,
        metric_key="roc_auc",
        scope_key="fold",
        fold_id=0,
        statistic_key="direct",
    )
    ref1 = replace(ref0)
    reference = GovernancePredictionProfile(
        "champion",
        "reference",
        "reference",
        datetime(2025, 1, 1),
        "ranking_score",
        "fold",
        None,
        "a" * 64,
        tuple(i / 10 for i in range(1, 10)),
        (0,) * 10,
        0,
        0,
        2,
        0,
        ref0,
    )
    current = replace(
        reference, snapshot_key="current", snapshot_role="current", source_ref=ref1
    )
    policy = _policy(_candidate("champion", 0, "champion"))
    return evaluate_governance(
        policy,
        risk_validations=(_h_base_owner(),),
        prediction_profiles=(reference, current),
    )


def _h_stability_result(
    *,
    common_support: str = "verified",
    target_values: tuple[bool, ...] = (False, True),
    bootstrap_repeats: int = 2,
) -> GovernanceResult:
    base = _h_make_performance(candidate_key="champion")
    reference = replace(
        base,
        snapshot_key="reference",
        snapshot_role="reference",
        target_values=target_values,
        ranking_scores=tuple(
            0.2 + 0.6 * i / max(1, len(target_values) - 1)
            for i in range(len(target_values))
        ),
        common_support=common_support,
        bootstrap_repeats=bootstrap_repeats,
    )
    current = replace(
        reference,
        snapshot_key="current",
        snapshot_role="current",
        window_start=datetime(2025, 1, 2),
        window_end=datetime(2025, 1, 3),
        evidence_as_of=datetime(2025, 1, 4),
    )
    return evaluate_governance(
        _policy(_candidate("champion", 0, "champion")),
        risk_validations=(_h_base_owner(),),
        performance_evidence=(reference, current),
    )


def _h_source_status_comparison(
    *, status: str, reason: str, entity_alignment: str = "not_requested"
) -> GovernanceResult:
    policy, owners = _h_pair_case()
    modified = []
    for index, owner in enumerate(owners):
        metrics = owner.metrics.copy(deep=True)
        selector = (
            metrics["scope"].eq("fold")
            & metrics["fold_id"].eq(0)
            & metrics["metric"].eq("roc_auc")
            & metrics["statistic"].eq("direct")
        )
        metrics.loc[selector, "status"] = status
        metrics.loc[selector, "reason"] = reason
        modified.append(replace(owner, metrics=metrics))
        if index == 0:
            break
    return evaluate_governance(
        replace(policy, entity_alignment=entity_alignment),
        risk_validations=(modified[0], owners[1]),
    )


def test_task19_gap_h_status_inventory_and_materialized_reachability() -> None:
    statuses = (
        "available",
        "unavailable",
        "undefined",
        "not_applicable",
        "not_verifiable",
    )
    assert len(set(statuses)) == 5
    available = _h_explanation_result("available", None)
    unavailable = _h_explanation_result("unavailable", "source_unavailable")
    undefined = _h_explanation_result("undefined", "source_undefined")
    not_verifiable = _h_explanation_result("not_verifiable", "source_not_verifiable")
    not_applicable = _h_diagnostic_result()
    assert available.explanations.iloc[0]["status"] == "available"
    assert pd.isna(available.explanations.iloc[0]["reason"])
    assert unavailable.explanations.iloc[0]["status"] == "unavailable"
    assert undefined.explanations.iloc[0]["status"] == "undefined"
    assert not_verifiable.explanations.iloc[0]["status"] == "not_verifiable"
    assert not_applicable.candidate_comparisons.iloc[0]["status"] == "not_applicable"
    assert not_applicable.candidate_comparisons.iloc[0]["reason"] == "diagnostic_only"


@pytest.mark.parametrize(
    "reason,expected_status,table_name",
    (
        ("source_unavailable", "unavailable", "explanations"),
        ("source_undefined", "undefined", "explanations"),
        ("source_not_verifiable", "not_verifiable", "explanations"),
        ("support_not_comparable", "not_verifiable", "candidate_comparisons"),
        ("insufficient_support", "undefined", "prediction_drift"),
        ("maturity_not_comparable", "not_verifiable", "candidate_comparisons"),
        ("snapshot_unverified", "not_verifiable", "candidate_comparisons"),
        ("alignment_unverified", "not_verifiable", "candidate_comparisons"),
        ("time_unverified", "not_verifiable", "explanations"),
        ("common_support_unverified", "not_verifiable", "performance_stability"),
        ("insufficient_bootstrap_support", "undefined", "performance_stability"),
        ("zero_denominator", "undefined", "candidate_comparisons"),
        ("single_class", "undefined", "performance_stability"),
        ("operation_not_applicable", "not_applicable", "candidate_comparisons"),
        ("diagnostic_only", "not_applicable", "candidate_comparisons"),
    ),
)
def test_task19_gap_h_reason_is_materialized_in_approved_table(
    reason: str, expected_status: str, table_name: str
) -> None:
    if reason in {"source_unavailable", "source_undefined", "source_not_verifiable"}:
        result = _h_explanation_result(expected_status, reason)
    elif reason == "support_not_comparable":
        policy, owners = _h_pair_case()
        metrics = owners[1].metrics.copy(deep=True)
        selector = (
            metrics["scope"].eq("fold")
            & metrics["fold_id"].eq(0)
            & metrics["metric"].eq("roc_auc")
            & metrics["statistic"].eq("direct")
        )
        metrics.loc[selector, "n_rows"] = 3
        result = evaluate_governance(
            policy, risk_validations=(owners[0], replace(owners[1], metrics=metrics))
        )
    elif reason == "insufficient_support":
        result = _h_profile_status_result()
    elif reason == "maturity_not_comparable":
        result = _h_source_status_comparison(status="not_verifiable", reason=reason)
    elif reason == "snapshot_unverified":
        result = _h_source_status_comparison(status="not_verifiable", reason=reason)
    elif reason == "alignment_unverified":
        result = _h_source_status_comparison(status="not_verifiable", reason=reason)
    elif reason == "time_unverified":
        result = _h_explanation_result(expected_status, reason, task16_time=True)
    elif reason == "common_support_unverified":
        result = _h_stability_result(common_support="unverified")
    elif reason == "insufficient_bootstrap_support":
        result = _h_stability_result(target_values=(False, True), bootstrap_repeats=2)
    elif reason == "single_class":
        result = _h_stability_result(target_values=(True, True), bootstrap_repeats=2)
    elif reason == "operation_not_applicable":
        result = _h_source_status_comparison(
            status="not_applicable", reason="source_not_requested"
        )
    elif reason == "diagnostic_only":
        result = _h_diagnostic_result()
    elif reason == "zero_denominator":
        result = _h_source_status_comparison(status="undefined", reason=reason)
    else:
        raise AssertionError(reason)
    table = getattr(result, table_name)
    row = table.loc[table["reason"] == reason].iloc[0]
    assert row["status"] == expected_status
    assert row["reason"] == reason


def test_task19_gap_h_invalid_source_still_raises_not_reason_row() -> None:
    for key in (
        "unsupported_source",
        "invalid_source_locator",
        "invalid_source_fingerprint",
        "source_fingerprint_mismatch",
    ):
        with pytest.raises(ValueError) as caught:
            _h_trigger(key)
        assert str(caught.value) == f"model governance: {key}"


# ---------------------------------------------------------------------------
# Wave 1C: bounded global validation-precedence acceptance matrix.
# Each builder constructs two simultaneous, independently observable invalid
# conditions and exercises the public evaluator.  The matrix deliberately
# stops at the caller-controlled/resource phase; recommendation and public
# table materialization have no independent caller-invalid branch.
# ---------------------------------------------------------------------------


def _e_future_attribution() -> GovernanceAttributionEvidence:
    return replace(
        _h_make_attribution(candidate_key="champion"),
        evidence_as_of=datetime(2026, 1, 1),
    )


def _e_p01() -> dict[str, object]:
    policy = replace(_h_base_policy(), governance_key="")
    future = _e_future_attribution()
    assert policy.governance_key == ""
    assert future.evidence_as_of > policy.analysis_as_of
    return {
        "invoke": lambda: evaluate_governance(
            policy, risk_validations=(_h_base_owner(),), model_attributions=(future,)
        ),
        "earlier": lambda: policy.governance_key == "",
        "later": lambda: future.evidence_as_of > policy.analysis_as_of,
    }


def _e_p02() -> dict[str, object]:
    policy = replace(
        _h_base_policy(),
        candidates=(replace(_h_base_policy().candidates[0], candidate_key=""),),
    )
    future = _e_future_attribution()
    assert policy.candidates[0].candidate_key == ""
    assert future.evidence_as_of > policy.analysis_as_of
    return {
        "invoke": lambda: evaluate_governance(
            policy, risk_validations=(_h_base_owner(),), model_attributions=(future,)
        ),
        "earlier": lambda: policy.candidates[0].candidate_key == "",
        "later": lambda: future.evidence_as_of > policy.analysis_as_of,
    }


def _e_p03() -> dict[str, object]:
    policy, owners = _h_pair_case()
    policy = replace(policy, comparison_pairs=(("champion", "missing"),))
    future = _e_future_attribution()
    assert policy.comparison_pairs != (("champion", "challenger"),)
    assert future.evidence_as_of > policy.analysis_as_of
    return {
        "invoke": lambda: evaluate_governance(
            policy, risk_validations=owners, model_attributions=(future,)
        ),
        "earlier": lambda: policy.comparison_pairs != (("champion", "challenger"),),
        "later": lambda: future.evidence_as_of > policy.analysis_as_of,
    }


def _e_p04() -> dict[str, object]:
    policy, owners = _h_pair_case()
    policy = replace(
        policy,
        criteria=(replace(policy.criteria[0], criterion_key=""),),
    )
    future = _e_future_attribution()
    assert policy.criteria[0].criterion_key == ""
    assert future.evidence_as_of > policy.analysis_as_of
    return {
        "invoke": lambda: evaluate_governance(
            policy, risk_validations=owners, model_attributions=(future,)
        ),
        "earlier": lambda: policy.criteria[0].criterion_key == "",
        "later": lambda: future.evidence_as_of > policy.analysis_as_of,
    }


def _e_p05() -> dict[str, object]:
    policy, owners = _h_pair_case()
    policy = replace(
        policy,
        candidates=(
            replace(policy.candidates[0], expected_source_fingerprint="a" * 64),
        )
        + policy.candidates[1:],
    )
    future = _e_future_attribution()
    assert policy.candidates[0].expected_source_fingerprint == "a" * 64
    assert future.evidence_as_of > policy.analysis_as_of
    return {
        "invoke": lambda: evaluate_governance(
            policy, risk_validations=owners, model_attributions=(future,)
        ),
        "earlier": lambda: policy.candidates[0].expected_source_fingerprint == "a" * 64,
        "later": lambda: future.evidence_as_of > policy.analysis_as_of,
    }


def _e_p06() -> dict[str, object]:
    owner = _risk_result()
    folds = owner.folds.copy(deep=True)
    folds["analysis_as_of"] = pd.Series([pd.NA] * len(folds), dtype="object")
    missing_time = replace(owner, validation_mode="time_forward", folds=folds)
    owners = tuple(_risk_result(i / 1000) for i in range(17))
    owners = (missing_time, *owners[1:])
    ref = _fold_ref("champion", 0, "diagnostic")
    policy = replace(_h_base_policy(), evidence_refs=(ref,))
    assert pd.isna(missing_time.folds["analysis_as_of"]).all()
    assert len(owners) == 17
    return {
        "invoke": lambda: evaluate_governance(policy, risk_validations=owners),
        "earlier": lambda: pd.isna(missing_time.folds["analysis_as_of"]).all(),
        "later": lambda: len(owners) > governance._RESOURCE_GATES[0][1],
    }


def _e_p07() -> dict[str, object]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "task18_precedence_fixture", "tests/test_lifecycle_monitoring.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    owner = module.monitor_lifecycle(module._frame(), module._config())
    provenance = owner.provenance.copy(deep=True)
    provenance.loc[
        provenance["provenance_key"] == "analysis_as_of", "provenance_value"
    ] = "not-a-datetime"
    malformed = replace(owner, provenance=provenance)
    candidate = GovernanceCandidate(
        "scenario",
        "warning_scenario",
        "task18",
        0,
        "reference",
        owner.monitoring_fingerprint,
        "v1",
        "champion",
        "approved",
        (),
    )
    owners = tuple(
        replace(malformed, provenance=malformed.provenance.copy(deep=True))
        for _ in range(17)
    )
    policy = _policy(candidate)
    assert "not-a-datetime" in set(
        malformed.provenance.loc[
            malformed.provenance["provenance_key"] == "analysis_as_of",
            "provenance_value",
        ]
    )
    assert len(owners) == 17
    return {
        "invoke": lambda: evaluate_governance(policy, lifecycle_monitorings=owners),
        "earlier": lambda: (
            "not-a-datetime"
            in set(
                malformed.provenance.loc[
                    malformed.provenance["provenance_key"] == "analysis_as_of",
                    "provenance_value",
                ]
            )
        ),
        "later": lambda: len(owners) > governance._RESOURCE_GATES[3][1],
    }


def _e_p08() -> dict[str, object]:
    policy = _h_base_policy()
    future = _e_future_attribution()
    owners = tuple(_risk_result(i / 1000) for i in range(17))
    assert future.evidence_as_of > policy.analysis_as_of
    assert len(owners) == 17
    return {
        "invoke": lambda: evaluate_governance(
            policy, risk_validations=owners, model_attributions=(future,)
        ),
        "earlier": lambda: future.evidence_as_of > policy.analysis_as_of,
        "later": lambda: len(owners) > governance._RESOURCE_GATES[0][1],
    }


def _e_p09() -> dict[str, object]:
    policy = _h_base_policy()
    risk = tuple(_risk_result(i / 1000) for i in range(17))
    audit = _e_clean_audit_owner()
    audits = tuple(replace(audit) for _ in range(17))
    assert len(risk) == 17
    assert len(audits) == 17
    return {
        "invoke": lambda: evaluate_governance(
            policy, risk_validations=risk, data_audits=audits
        ),
        "earlier": lambda: len(risk) > governance._RESOURCE_GATES[0][1],
        "later": lambda: len(audits) > governance._RESOURCE_GATES[1][1],
    }


def _e_clean_audit_owner() -> DataAuditResult:
    """Return a real Task 16 result with only contract-valid status cells."""
    owner = _real_base_owner("task16")
    tables: dict[str, pd.DataFrame] = {}
    for name, table in vars(owner).items():
        if type(table) is not pd.DataFrame:
            continue
        cleaned = table.copy(deep=True)
        for column in cleaned.columns:
            if column == "status" or column.endswith("_status"):
                cleaned[column] = "available"
            elif column == "reason" or column.endswith("_reason"):
                cleaned[column] = pd.NA
        tables[name] = cleaned
    return replace(owner, **tables)


def _e_p10() -> dict[str, object]:
    metadata = replace(
        _metadata(),
        assumption_keys=tuple(f"assumption_{i}" for i in range(252)),
    )
    template = GovernanceEvidenceRef(
        "task15",
        0,
        "metrics",
        "comparison_criterion",
        "champion",
        None,
        metric_key="roc_auc",
        scope_key="fold",
        fold_id=0,
        statistic_key="direct",
    )
    refs = tuple(replace(template, field_key=f"padding_{i}") for i in range(8193))
    policy = replace(_h_base_policy(), metadata=(metadata,), evidence_refs=refs)
    assert 5 + len(metadata.assumption_keys) > governance._RESOURCE_GATES[12][1]
    assert len(refs) > governance._RESOURCE_GATES[13][1]
    return {
        "invoke": lambda: evaluate_governance(
            policy, risk_validations=(_h_base_owner(),)
        ),
        "earlier": lambda: (
            5 + len(metadata.assumption_keys) > governance._RESOURCE_GATES[12][1]
        ),
        "later": lambda: len(refs) > governance._RESOURCE_GATES[13][1],
    }


def _e_p11() -> dict[str, object]:
    owners = [_risk_result(i / 1000) for i in range(17)]
    bad_metrics = owners[0].metrics.copy(deep=True)
    selector = (
        bad_metrics["scope"].eq("fold")
        & bad_metrics["fold_id"].eq(0)
        & bad_metrics["metric"].eq("roc_auc")
        & bad_metrics["statistic"].eq("direct")
    )
    bad_metrics.loc[selector, "value"] = np.inf
    owners[0] = replace(owners[0], metrics=bad_metrics)
    ref0 = GovernanceEvidenceRef(
        "task15",
        0,
        "metrics",
        "comparison_criterion",
        "champion",
        None,
        metric_key="roc_auc",
        scope_key="fold",
        fold_id=0,
        statistic_key="direct",
    )
    ref1 = replace(ref0, source_result_position=1, candidate_key="challenger")
    criterion = GovernanceCriterion(
        "auc",
        "model",
        "task15",
        "metrics",
        "roc_auc",
        "fold",
        None,
        None,
        "decision",
        True,
        "higher_is_better",
        minimum_support=1,
    )
    policy = _policy(
        _candidate("champion", 0, "champion", (ref0,)),
        replace(
            _candidate("challenger", 1, "challenger", (ref1,)),
            declared_state="candidate",
        ),
        pairs=(("champion", "challenger"),),
        criteria=(criterion,),
    )
    assert np.isinf(bad_metrics.loc[selector, "value"].iloc[0])
    assert len(owners) > governance._RESOURCE_GATES[0][1]
    return {
        "invoke": lambda: evaluate_governance(policy, risk_validations=tuple(owners)),
        "earlier": lambda: len(owners) > governance._RESOURCE_GATES[0][1],
        "later": lambda: np.isinf(bad_metrics.loc[selector, "value"].iloc[0]),
    }


_E_PRECEDENCE_CASES = (
    {
        "case_id": "P01",
        "earlier_invalid": "invalid_governance_key",
        "later_invalid": "future_evidence_time",
        "earlier_phase": 1,
        "later_phase": 9,
        "expected_error": "invalid_governance_key",
        "fixture_builder": _e_p01,
    },
    {
        "case_id": "P02",
        "earlier_invalid": "invalid_candidate",
        "later_invalid": "future_evidence_time",
        "earlier_phase": 3,
        "later_phase": 9,
        "expected_error": "invalid_candidate",
        "fixture_builder": _e_p02,
    },
    {
        "case_id": "P03",
        "earlier_invalid": "invalid_pair_coverage",
        "later_invalid": "future_evidence_time",
        "earlier_phase": 3,
        "later_phase": 9,
        "expected_error": "invalid_pair_coverage",
        "fixture_builder": _e_p03,
    },
    {
        "case_id": "P04",
        "earlier_invalid": "invalid_criterion",
        "later_invalid": "future_evidence_time",
        "earlier_phase": 4,
        "later_phase": 9,
        "expected_error": "invalid_criterion",
        "fixture_builder": _e_p04,
    },
    {
        "case_id": "P05",
        "earlier_invalid": "source_fingerprint_mismatch",
        "later_invalid": "future_evidence_time",
        "earlier_phase": 6,
        "later_phase": 9,
        "expected_error": "source_fingerprint_mismatch",
        "fixture_builder": _e_p05,
    },
    {
        "case_id": "P06",
        "earlier_invalid": "authoritative_time_missing",
        "later_invalid": "resource_risk_validation_results",
        "earlier_phase": 7,
        "later_phase": 11,
        "expected_error": "authoritative_time_missing",
        "fixture_builder": _e_p06,
    },
    {
        "case_id": "P07",
        "earlier_invalid": "authoritative_time_mismatch",
        "later_invalid": "resource_lifecycle_monitoring_results",
        "earlier_phase": 8,
        "later_phase": 11,
        "expected_error": "authoritative_time_mismatch",
        "fixture_builder": _e_p07,
    },
    {
        "case_id": "P08",
        "earlier_invalid": "future_evidence_time",
        "later_invalid": "resource_risk_validation_results",
        "earlier_phase": 9,
        "later_phase": 11,
        "expected_error": "future_evidence_time",
        "fixture_builder": _e_p08,
    },
    {
        "case_id": "P09",
        "earlier_invalid": "resource_risk_validation_results",
        "later_invalid": "resource_data_audit_results",
        "earlier_phase": 11,
        "later_phase": 11,
        "expected_error": "resource_risk_validation_results",
        "fixture_builder": _e_p09,
    },
    {
        "case_id": "P10",
        "earlier_invalid": "resource_governance_metadata_rows",
        "later_invalid": "resource_evidence_refs",
        "earlier_phase": 11,
        "later_phase": 11,
        "expected_error": "invalid_evidence_ref",
        "fixture_builder": _e_p10,
    },
    {
        "case_id": "P11",
        "earlier_invalid": "resource_risk_validation_results",
        "later_invalid": "invalid_owner_value",
        "earlier_phase": 11,
        "later_phase": 13,
        "expected_error": "invalid_owner_value",
        "fixture_builder": _e_p11,
    },
)


def test_task19_gap_e_precedence_matrix_inventory_is_exact() -> None:
    case_ids = tuple(case["case_id"] for case in _E_PRECEDENCE_CASES)
    assert case_ids == tuple(f"P{i:02d}" for i in range(1, 12))
    assert len(case_ids) == len(set(case_ids)) == 11
    assert all(
        case["earlier_phase"] <= case["later_phase"] for case in _E_PRECEDENCE_CASES
    )


@pytest.mark.parametrize(
    "case", _E_PRECEDENCE_CASES, ids=lambda case: str(case["case_id"])
)
def test_task19_gap_e_public_precedence_matrix(case: dict[str, object]) -> None:
    fixture = case["fixture_builder"]()
    assert fixture["earlier"]()
    assert fixture["later"]()
    with pytest.raises(ValueError) as caught:
        fixture["invoke"]()
    assert type(caught.value) is ValueError
    assert str(caught.value) == f"model governance: {case['expected_error']}"


def test_task19_gap_e_no_private_precedence_bypass() -> None:
    assert all(callable(case["fixture_builder"]) for case in _E_PRECEDENCE_CASES)
    assert all(case["case_id"].startswith("P") for case in _E_PRECEDENCE_CASES)


def test_task19_gap_e_p12_contractually_not_separately_constructible() -> None:
    """Recommendation/materialization have no independent caller-invalid input."""
    assert "recommendation aggregation" in Path(
        "docs/decisions/task19-explainability-champion-challenger-governance-contract.md"
    ).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Wave 2A: Gap A public ten-table contract acceptance.
# The schema matrix below is transcribed from the approved v2 contract; it is
# intentionally independent of the production schema factory.
# ---------------------------------------------------------------------------


_A_PUBLIC_TABLES = (
    "explanations",
    "model_attributions",
    "prediction_drift",
    "performance_stability",
    "candidate_comparisons",
    "governance_evaluations",
    "recommendations",
    "governance_summary",
    "governance_metadata",
    "provenance",
)


def _a_table_spec(
    columns: tuple[str, ...],
    dtypes: tuple[str, ...],
    identity: tuple[str, ...],
    ordering: str,
    finding: str,
    conditional: str,
) -> dict[str, object]:
    assert len(columns) == len(dtypes)
    return {
        "columns": columns,
        "dtypes": dtypes,
        "identity": identity,
        "ordering": ordering,
        "finding": finding,
        "conditional": conditional,
    }


_A_TABLE_MATRIX = {
    "explanations": _a_table_spec(
        (
            "explanation_position",
            "explanation_key",
            "candidate_position",
            "candidate_family",
            "method",
            "source_ref_position",
            "source_task",
            "source_result_position",
            "source_table",
            "source_registry_position",
            "source_fingerprint",
            "feature_key",
            "relation",
            "priority",
            "evidence_time_status",
            "source_status",
            "source_reason",
            "status",
            "reason",
            "finding_key",
        ),
        (
            "Int64",
            "string",
            "Int64",
            "string",
            "string",
            "Int64",
            "string",
            "Int64",
            "string",
            "Int64",
            "string",
            "string",
            "string",
            "Int64",
            "string",
            "string",
            "string",
            "string",
            "string",
            "string",
        ),
        ("explanation_position",),
        "declaration position ascending; no secondary sort",
        "governance:explanation:<explanation_position>",
        "source/Task19 status and optional feature/relation fields",
    ),
    "model_attributions": _a_table_spec(
        (
            "candidate_position",
            "attribution_position",
            "candidate_family",
            "method",
            "feature_key",
            "metric_key",
            "value",
            "relation",
            "evaluation_scope",
            "support_n",
            "uncertainty_std",
            "permutation_repeats",
            "random_state",
            "evidence_as_of",
            "evidence_time_status",
            "source_task",
            "source_result_position",
            "source_table",
            "source_ref_position",
            "source_fingerprint",
            "source_status",
            "source_reason",
            "status",
            "reason",
            "finding_key",
        ),
        (
            "Int64",
            "Int64",
            "string",
            "string",
            "string",
            "string",
            "Float64",
            "string",
            "string",
            "Int64",
            "Float64",
            "Int64",
            "Int64",
            "datetime",
            "string",
            "string",
            "Int64",
            "string",
            "Int64",
            "string",
            "string",
            "string",
            "string",
            "string",
            "string",
        ),
        ("candidate_position", "attribution_position"),
        "candidate, fixed method order, declaration position",
        "governance:attribution:<candidate_position>:<attribution_position>",
        "method-dependent metric/scope/uncertainty/repeat/seed nullability",
    ),
    "prediction_drift": _a_table_spec(
        (
            "candidate_position",
            "reference_profile_position",
            "current_profile_position",
            "prediction_kind",
            "scope_key",
            "scope_position",
            "reference_snapshot_key",
            "current_snapshot_key",
            "reference_analysis_as_of",
            "current_analysis_as_of",
            "reference_time_status",
            "current_time_status",
            "reference_support_n",
            "current_support_n",
            "reference_missing_n",
            "current_missing_n",
            "bin_count",
            "reference_state_fingerprint",
            "reference_source_fingerprint",
            "current_source_fingerprint",
            "metric",
            "prediction_tvd",
            "direction",
            "uncertainty_std",
            "bootstrap_repeats",
            "random_state",
            "status",
            "reason",
            "finding_key",
        ),
        (
            "Int64",
            "Int64",
            "Int64",
            "string",
            "string",
            "Int64",
            "string",
            "string",
            "datetime",
            "datetime",
            "string",
            "string",
            "Int64",
            "Int64",
            "Int64",
            "Int64",
            "Int64",
            "string",
            "string",
            "string",
            "string",
            "Float64",
            "string",
            "Float64",
            "Int64",
            "Int64",
            "string",
            "string",
            "string",
        ),
        (
            "candidate_position",
            "reference_profile_position",
            "current_profile_position",
        ),
        "candidate then reference profile declaration position",
        "governance:drift:<candidate_position>:<reference_profile_position>:<current_profile_position>",
        "available metric/std versus undefined/not-verifiable numeric NA",
    ),
    "performance_stability": _a_table_spec(
        (
            "candidate_position",
            "reference_evidence_position",
            "current_evidence_position",
            "evaluation_scope",
            "scope_key",
            "scope_position",
            "reference_snapshot_key",
            "current_snapshot_key",
            "reference_window_start",
            "reference_window_end",
            "current_window_start",
            "current_window_end",
            "reference_evidence_as_of",
            "current_evidence_as_of",
            "reference_time_status",
            "current_time_status",
            "metric",
            "reference_value",
            "current_value",
            "delta",
            "direction",
            "reference_uncertainty_std",
            "current_uncertainty_std",
            "reference_bootstrap_repeats",
            "current_bootstrap_repeats",
            "reference_random_state",
            "current_random_state",
            "reference_support_n",
            "current_support_n",
            "reference_assignment_mechanism",
            "current_assignment_mechanism",
            "reference_common_support",
            "current_common_support",
            "reference_source_fingerprint",
            "current_source_fingerprint",
            "status",
            "reason",
            "finding_key",
        ),
        (
            "Int64",
            "Int64",
            "Int64",
            "string",
            "string",
            "Int64",
            "string",
            "string",
            "datetime",
            "datetime",
            "datetime",
            "datetime",
            "datetime",
            "datetime",
            "string",
            "string",
            "string",
            "Float64",
            "Float64",
            "Float64",
            "string",
            "Float64",
            "Float64",
            "Int64",
            "Int64",
            "Int64",
            "Int64",
            "Int64",
            "Int64",
            "string",
            "string",
            "string",
            "string",
            "string",
            "string",
            "string",
            "string",
            "string",
        ),
        (
            "candidate_position",
            "reference_evidence_position",
            "current_evidence_position",
        ),
        "candidate, reference/current declaration positions, metric inventory",
        "governance:stability:<candidate_position>:<reference_evidence_position>:<current_evidence_position>",
        "available values/std/delta versus non-available numeric NA",
    ),
    "candidate_comparisons": _a_table_spec(
        (
            "pair_position",
            "champion_candidate_position",
            "challenger_candidate_position",
            "candidate_family",
            "criterion_position",
            "criterion_role",
            "source_task",
            "source_table",
            "metric_key",
            "scope_key",
            "scope_position",
            "rule_key",
            "champion_source_result_position",
            "challenger_source_result_position",
            "champion_source_ref_position",
            "challenger_source_ref_position",
            "champion_source_fingerprint",
            "challenger_source_fingerprint",
            "champion_time_status",
            "challenger_time_status",
            "champion_source_status",
            "champion_source_reason",
            "challenger_source_status",
            "challenger_source_reason",
            "source_snapshot_status",
            "entity_alignment_status",
            "normalized_support_identity",
            "champion_value",
            "challenger_value",
            "delta",
            "champion_support_n",
            "challenger_support_n",
            "support_unit",
            "direction",
            "target_low",
            "target_high",
            "comparison_outcome",
            "support_comparable",
            "status",
            "reason",
            "finding_key",
        ),
        (
            "Int64",
            "Int64",
            "Int64",
            "string",
            "Int64",
            "string",
            "string",
            "string",
            "string",
            "string",
            "Int64",
            "string",
            "Int64",
            "Int64",
            "Int64",
            "Int64",
            "string",
            "string",
            "string",
            "string",
            "string",
            "string",
            "string",
            "string",
            "string",
            "string",
            "string",
            "Float64",
            "Float64",
            "Float64",
            "Int64",
            "Int64",
            "string",
            "string",
            "Float64",
            "Float64",
            "string",
            "boolean",
            "string",
            "string",
            "string",
        ),
        ("pair_position", "criterion_position"),
        "pair then criterion declaration order",
        "governance:comparison:<pair_position>:<criterion_position>",
        "directional/non-comparable/diagnostic conditional values",
    ),
    "governance_evaluations": _a_table_spec(
        (
            "pair_position",
            "champion_candidate_position",
            "challenger_candidate_position",
            "criterion_position",
            "criterion_role",
            "required_for_promotion",
            "priority",
            "comparison_outcome",
            "comparable",
            "counts_toward_minimum",
            "blocks_promotion",
            "directional_contribution",
            "evidence_time_status",
            "status",
            "reason",
            "finding_key",
        ),
        (
            "Int64",
            "Int64",
            "Int64",
            "Int64",
            "string",
            "boolean",
            "Int64",
            "string",
            "boolean",
            "boolean",
            "boolean",
            "string",
            "string",
            "string",
            "string",
            "string",
        ),
        ("pair_position", "criterion_position"),
        "pair then criterion declaration order",
        "governance:evaluation:<pair_position>:<criterion_position>",
        "decision and diagnostic contribution/boolean fields",
    ),
    "recommendations": _a_table_spec(
        (
            "pair_position",
            "champion_candidate_position",
            "challenger_candidate_position",
            "candidate_family",
            "recommendation",
            "recommendation_basis",
            "hard_veto",
            "human_review_mode",
            "human_review_required",
            "minimum_comparable_criteria",
            "criteria_available_n",
            "criteria_unavailable_n",
            "criteria_better_n",
            "criteria_worse_n",
            "criteria_tied_n",
            "required_incomplete_n",
            "support_comparable",
            "status",
            "reason",
            "finding_key",
        ),
        (
            "Int64",
            "Int64",
            "Int64",
            "string",
            "string",
            "string",
            "boolean",
            "string",
            "boolean",
            "Int64",
            "Int64",
            "Int64",
            "Int64",
            "Int64",
            "Int64",
            "Int64",
            "boolean",
            "string",
            "string",
            "string",
        ),
        ("pair_position",),
        "pair position ascending",
        "governance:recommendation:<pair_position>",
        "status available and reason NA for legal pair",
    ),
    "governance_summary": _a_table_spec(
        (
            "candidate_position",
            "candidate_family",
            "declared_role",
            "declared_state",
            "source_task",
            "source_result_position",
            "source_candidate_position",
            "source_snapshot_status",
            "entity_alignment_status",
            "evidence_time_status",
            "criterion_count",
            "available_criterion_count",
            "unavailable_criterion_count",
            "not_verifiable_criterion_count",
            "attribution_count",
            "prediction_drift_count",
            "performance_stability_count",
            "recommendation_count",
            "human_review_required_count",
            "status",
            "reason",
            "finding_key",
        ),
        (
            "Int64",
            "string",
            "string",
            "string",
            "string",
            "Int64",
            "Int64",
            "string",
            "string",
            "string",
            "Int64",
            "Int64",
            "Int64",
            "Int64",
            "Int64",
            "Int64",
            "Int64",
            "Int64",
            "Int64",
            "string",
            "string",
            "string",
        ),
        ("candidate_position",),
        "candidate declaration position ascending",
        "governance:summary:<candidate_position>",
        "summary always available with reason NA",
    ),
    "governance_metadata": _a_table_spec(
        (
            "metadata_position",
            "metadata_scope",
            "candidate_position",
            "field_position",
            "field_key",
            "item_position",
            "text_value",
            "numeric_value",
            "evidence_time_status",
            "status",
            "reason",
            "finding_key",
        ),
        (
            "Int64",
            "string",
            "Int64",
            "Int64",
            "string",
            "Int64",
            "string",
            "Float64",
            "string",
            "string",
            "string",
            "string",
        ),
        ("metadata_position", "field_position"),
        "metadata declaration then fixed field/item order",
        "governance:metadata:<scope-branch>:<metadata_position>:<field_position>",
        "candidate scope and text/numeric mutual nullability",
    ),
    "provenance": _a_table_spec(
        (
            "provenance_position",
            "provenance_key",
            "provenance_value",
            "status",
            "reason",
            "finding_key",
        ),
        ("Int64", "string", "string", "string", "string", "string"),
        ("provenance_position",),
        "fixed position 0..34",
        "governance:provenance:<provenance_position>",
        "always 35 available rows with reason NA",
    ),
}


_A_PROVENANCE_KEYS = (
    "contract_version",
    "package_version",
    "canonical_encoder_version",
    "source_registry_identity",
    "direction_registry_identity",
    "recommendation_policy_identity",
    "analysis_as_of",
    "time_model",
    "policy_fingerprint",
    "candidate_inventory_fingerprint",
    "criterion_inventory_fingerprint",
    "source_inventory_fingerprint",
    "source_binding_inventory",
    "explanation_inventory_fingerprint",
    "attribution_inventory_fingerprint",
    "prediction_profile_inventory_fingerprint",
    "performance_evidence_inventory_fingerprint",
    "metadata_inventory_fingerprint",
    "structured_evidence_fingerprint",
    "governance_fingerprint",
    "source_snapshot_status",
    "entity_alignment_status",
    "evidence_time_status",
    "candidate_count",
    "comparison_pair_count",
    "criterion_count",
    "risk_validation_result_count",
    "data_audit_result_count",
    "decision_strategy_result_count",
    "lifecycle_monitoring_result_count",
    "task15_owner_fingerprint_statuses",
    "task16_owner_fingerprints",
    "task17_owner_fingerprints",
    "task18_owner_fingerprints",
    "evidence_time_statuses",
)


def _a_time(aware: bool, day: int) -> datetime | pd.Timestamp:
    value = datetime(2025, 1, day)
    return pd.Timestamp(value, tz="UTC") if aware else value


def _a_attributions(aware: bool) -> tuple[GovernanceAttributionEvidence, ...]:
    base = replace(
        _h_make_attribution(candidate_key="champion"),
        evidence_as_of=_a_time(aware, 1),
    )
    return (
        replace(base, feature_key="z_feature"),
        replace(
            base,
            method="native_importance",
            feature_key="a_feature",
            relation="not_directional",
        ),
        replace(
            base,
            method="permutation_importance",
            feature_key="m_feature",
            metric_key="roc_auc",
            evaluation_scope="holdout",
            uncertainty_std=0.1,
            permutation_repeats=2,
            random_state=7,
        ),
    )


def _a_profiles(aware: bool) -> tuple[GovernancePredictionProfile, ...]:
    base = _h_make_profile(candidate_key="champion")
    reference = replace(
        base,
        analysis_as_of=_a_time(aware, 1),
        source_ref=replace(base.source_ref),
    )
    current = replace(
        reference,
        snapshot_key="current",
        snapshot_role="current",
        analysis_as_of=_a_time(aware, 2),
        source_ref=replace(reference.source_ref),
    )
    return reference, current


def _a_performance(aware: bool) -> tuple[GovernancePerformanceEvidence, ...]:
    base = _h_make_performance(candidate_key="champion")
    target_values = (False, True) * 6
    ranking_scores = tuple(
        0.2 + 0.6 * i / (len(target_values) - 1) for i in range(len(target_values))
    )
    reference = replace(
        base,
        window_start=_a_time(aware, 1),
        window_end=_a_time(aware, 2),
        evidence_as_of=_a_time(aware, 3),
        target_values=target_values,
        ranking_scores=ranking_scores,
        source_ref=replace(base.source_ref),
    )
    current = replace(
        reference,
        snapshot_key="current",
        snapshot_role="current",
        window_start=_a_time(aware, 4),
        window_end=_a_time(aware, 5),
        evidence_as_of=_a_time(aware, 6),
        source_ref=replace(reference.source_ref),
    )
    return reference, current


@cache
def _a_rich_result(aware: bool = False) -> GovernanceResult:
    policy, owners = _h_pair_case(aware=aware)
    decision = policy.criteria[0]
    diagnostic = replace(
        decision,
        criterion_key="diagnostic_auc",
        criterion_role="diagnostic",
        required_for_promotion=False,
        direction="not_directional",
        priority=1,
    )
    governance_metadata = replace(
        _metadata(),
        monitoring_thresholds=(("review_threshold", 0.5),),
    )
    candidate_metadata = GovernanceMetadata(
        "candidate_metadata",
        "candidate",
        "champion",
        "offline_review",
        "risk_team",
        "medium",
    )
    explanation_ref = _fold_ref("champion", 0, "explanation")
    explanations = (
        GovernanceExplanation(
            "z_explanation",
            "champion",
            "native_importance",
            explanation_ref,
            "z_feature",
            "not_directional",
            99,
            "available",
            None,
        ),
        GovernanceExplanation(
            "a_explanation",
            "champion",
            "coefficient_direction",
            replace(explanation_ref),
            "a_feature",
            "positive",
            -1,
            "available",
            None,
        ),
    )
    policy = replace(
        policy,
        analysis_as_of=_a_time(aware, 31),
        criteria=(decision, diagnostic),
        metadata=(governance_metadata, candidate_metadata),
        explanations=explanations,
    )
    return evaluate_governance(
        policy,
        risk_validations=owners,
        model_attributions=_a_attributions(aware),
        prediction_profiles=_a_profiles(aware),
        performance_evidence=_a_performance(aware),
    )


@cache
def _a_empty_result(aware: bool = False) -> GovernanceResult:
    policy = replace(_h_base_policy(), analysis_as_of=_a_time(aware, 31))
    return evaluate_governance(policy, risk_validations=(_h_base_owner(),))


def _a_expected_dtype(spec: str, *, aware: bool) -> str:
    if spec == "datetime":
        return "datetime64[ns, UTC]" if aware else "datetime64[ns]"
    return spec


def _a_assert_public_table_contract(result: GovernanceResult, *, aware: bool) -> None:
    for name, spec in _A_TABLE_MATRIX.items():
        table = getattr(result, name)
        columns = spec["columns"]
        dtypes = spec["dtypes"]
        assert list(table.columns) == list(columns)
        assert [str(dtype) for dtype in table.dtypes] == [
            _a_expected_dtype(dtype, aware=aware) for dtype in dtypes
        ]
        assert all(str(dtype) != "object" for dtype in table.dtypes)


def test_task19_gap_a_ten_table_matrix_is_exact_and_contract_frozen() -> None:
    assert tuple(_A_TABLE_MATRIX) == _A_PUBLIC_TABLES
    assert set(_A_TABLE_MATRIX) == set(_A_PUBLIC_TABLES)
    assert len(_A_TABLE_MATRIX) == len(set(_A_TABLE_MATRIX)) == 10
    for spec in _A_TABLE_MATRIX.values():
        assert spec["columns"]
        assert len(spec["columns"]) == len(spec["dtypes"])
        assert spec["identity"]
        assert spec["ordering"]
        assert spec["finding"]
        assert spec["conditional"]


def test_task19_gap_a_all_ten_tables_public_populated() -> None:
    result = _a_rich_result()
    populated = {name for name in _A_PUBLIC_TABLES if len(getattr(result, name)) > 0}
    assert populated == set(_A_PUBLIC_TABLES)
    assert all(len(getattr(result, name)) > 0 for name in _A_PUBLIC_TABLES)


@pytest.mark.parametrize("aware", (False, True), ids=("naive", "aware"))
def test_task19_gap_a_populated_schema_dtype_and_no_object(
    aware: bool,
) -> None:
    _a_assert_public_table_contract(_a_rich_result(aware), aware=aware)


@pytest.mark.parametrize("aware", (False, True), ids=("naive", "aware"))
def test_task19_gap_a_typed_empty_public_contract(aware: bool) -> None:
    result = _a_empty_result(aware)
    typed_empty = {
        "explanations",
        "model_attributions",
        "prediction_drift",
        "performance_stability",
        "candidate_comparisons",
        "governance_evaluations",
        "recommendations",
    }
    assert typed_empty == {
        name for name in _A_PUBLIC_TABLES if len(getattr(result, name)) == 0
    }
    assert len(typed_empty) == 7
    assert len(result.governance_summary) == 1
    assert len(result.governance_metadata) == 5
    assert len(result.provenance) == 35
    _a_assert_public_table_contract(result, aware=aware)


def test_task19_gap_a_status_reason_and_null_sentinel_matrix() -> None:
    result = _a_rich_result()
    for name in _A_PUBLIC_TABLES:
        table = getattr(result, name)
        if "status" not in table.columns or "reason" not in table.columns:
            continue
        available = table.loc[table["status"] == "available", "reason"]
        assert all(pd.isna(value) for value in available)
        assert not any(
            isinstance(value, str) and value in {"", "NA", "None"}
            for value in table["reason"].dropna()
        )


def test_task19_gap_a_explanations_public_order_identity_and_finding() -> None:
    table = _a_rich_result().explanations
    assert table["explanation_position"].tolist() == [0, 1]
    assert table["explanation_key"].tolist() == ["z_explanation", "a_explanation"]
    assert table["priority"].tolist() == [99, -1]
    assert table["method"].tolist() == ["native_importance", "coefficient_direction"]
    assert table["finding_key"].tolist() == [
        "governance:explanation:0",
        "governance:explanation:1",
    ]
    assert table["explanation_position"].is_unique


def test_task19_gap_a_attribution_method_conditional_matrix() -> None:
    table = _a_rich_result().model_attributions
    assert set(table["method"]) == {
        "coefficient_direction",
        "native_importance",
        "permutation_importance",
    }
    assert table["attribution_position"].is_unique
    assert table["finding_key"].is_unique
    for method in ("coefficient_direction", "native_importance"):
        row = table.loc[table["method"] == method].iloc[0]
        assert row["evaluation_scope"] == "not_applicable"
        assert pd.isna(row["metric_key"])
        assert pd.isna(row["uncertainty_std"])
        assert pd.isna(row["permutation_repeats"])
        assert pd.isna(row["random_state"])
        assert pd.notna(row["value"])
    permutation = table.loc[table["method"] == "permutation_importance"].iloc[0]
    assert permutation["evaluation_scope"] in {"holdout", "oof"}
    assert permutation["metric_key"] == "roc_auc"
    assert pd.notna(permutation["uncertainty_std"])
    assert pd.notna(permutation["permutation_repeats"])
    assert pd.notna(permutation["random_state"])


def test_task19_gap_a_drift_available_and_nonavailable_matrix() -> None:
    available = _a_rich_result().prediction_drift.iloc[0]
    assert available["status"] == "available"
    assert pd.isna(available["reason"])
    assert pd.notna(available["prediction_tvd"])
    assert pd.notna(available["uncertainty_std"])
    nonavailable = _h_profile_status_result().prediction_drift.iloc[0]
    assert nonavailable["status"] == "undefined"
    assert nonavailable["reason"] == "insufficient_support"
    assert pd.isna(nonavailable["prediction_tvd"])
    assert pd.isna(nonavailable["uncertainty_std"])


def test_task19_gap_a_stability_available_and_nonavailable_matrix() -> None:
    available = _a_rich_result().performance_stability.iloc[0]
    assert available["status"] == "available"
    assert pd.isna(available["reason"])
    for field in (
        "reference_value",
        "current_value",
        "delta",
        "reference_uncertainty_std",
        "current_uncertainty_std",
    ):
        assert pd.notna(available[field])
    nonavailable = _h_stability_result(
        common_support="unverified"
    ).performance_stability.iloc[0]
    assert nonavailable["status"] == "not_verifiable"
    assert nonavailable["reason"] == "common_support_unverified"
    for field in (
        "reference_value",
        "current_value",
        "delta",
        "reference_uncertainty_std",
        "current_uncertainty_std",
    ):
        assert pd.isna(nonavailable[field])


def test_task19_gap_a_comparison_and_evaluation_conditional_matrix() -> None:
    rich = _a_rich_result()
    decision = rich.candidate_comparisons.loc[
        rich.candidate_comparisons["criterion_role"] == "decision"
    ].iloc[0]
    diagnostic = rich.candidate_comparisons.loc[
        rich.candidate_comparisons["criterion_role"] == "diagnostic"
    ].iloc[0]
    assert decision["status"] == "available"
    assert pd.notna(decision["champion_value"])
    assert pd.notna(decision["challenger_value"])
    assert (
        decision["delta"] == decision["challenger_value"] - decision["champion_value"]
    )
    assert diagnostic["status"] == "not_applicable"
    assert diagnostic["reason"] == "diagnostic_only"
    assert diagnostic["comparison_outcome"] == "not_directional"
    assert pd.notna(diagnostic["delta"])
    not_comparable = _h_source_status_comparison(
        status="not_verifiable", reason="support_not_comparable"
    ).candidate_comparisons.iloc[0]
    assert not_comparable["status"] == "not_verifiable"
    assert not_comparable["reason"] == "support_not_comparable"
    assert pd.isna(not_comparable["delta"])
    assert pd.isna(not_comparable["champion_value"])
    assert pd.isna(not_comparable["challenger_value"])
    evaluations = rich.governance_evaluations
    assert set(evaluations["criterion_role"]) == {"decision", "diagnostic"}
    assert (
        evaluations[["pair_position", "criterion_position"]].drop_duplicates().shape[0]
        == 2
    )
    diagnostic_eval = evaluations.loc[
        evaluations["criterion_role"] == "diagnostic"
    ].iloc[0]
    assert diagnostic_eval["directional_contribution"] == "not_directional"
    assert diagnostic_eval["counts_toward_minimum"] is np.bool_(False)


def test_task19_gap_a_recommendation_summary_metadata_conditional_matrix() -> None:
    rich = _a_rich_result()
    recommendation = rich.recommendations.iloc[0]
    assert len(rich.recommendations) == len(
        rich.candidate_comparisons["pair_position"].unique()
    )
    assert recommendation["status"] == "available"
    assert pd.isna(recommendation["reason"])
    empty = _a_empty_result()
    assert empty.recommendations.empty
    summary = rich.governance_summary
    assert len(summary) == rich.candidate_count
    assert summary["candidate_position"].tolist() == [0, 1]
    assert all(pd.isna(value) for value in summary["reason"])
    metadata = rich.governance_metadata
    assert set(metadata["metadata_scope"]) == {"governance", "candidate"}
    assert metadata["finding_key"].is_unique
    for _, row in metadata.iterrows():
        assert pd.isna(row["text_value"]) ^ pd.isna(row["numeric_value"])
    threshold = metadata.loc[metadata["field_key"] == "threshold"].iloc[0]
    assert threshold["metadata_scope"] == "governance"
    assert pd.isna(threshold["text_value"])
    assert threshold["numeric_value"] == 0.5


def test_task19_gap_a_identity_order_and_finding_matrix() -> None:
    result = _a_rich_result()
    expected_pairs = {
        "candidate_comparisons": ("pair_position", "criterion_position"),
        "governance_evaluations": ("pair_position", "criterion_position"),
    }
    for name, identity_columns in expected_pairs.items():
        table = getattr(result, name)
        assert list(
            table[list(identity_columns)].itertuples(index=False, name=None)
        ) == [(0, 0), (0, 1)]
        assert table["finding_key"].is_unique
    assert list(result.recommendations["pair_position"]) == [0]
    assert list(result.governance_summary["candidate_position"]) == [0, 1]
    metadata_identity = list(
        result.governance_metadata[["metadata_position", "field_position"]].itertuples(
            index=False, name=None
        )
    )
    assert metadata_identity == sorted(metadata_identity)
    assert result.provenance["provenance_position"].tolist() == list(range(35))
    assert result.provenance["finding_key"].tolist() == [
        f"governance:provenance:{i}" for i in range(35)
    ]
    for name, spec in _A_TABLE_MATRIX.items():
        table = getattr(result, name)
        assert table[list(spec["identity"])].drop_duplicates().shape[0] == len(table)


def test_task19_gap_a_provenance_exact_v2_inventory_and_values() -> None:
    result = _a_rich_result()
    provenance = result.provenance
    assert len(provenance) == 35
    assert provenance["provenance_key"].tolist() == list(_A_PROVENANCE_KEYS)
    assert all(provenance["status"] == "available")
    assert all(pd.isna(value) for value in provenance["reason"])
    values = {
        row.provenance_key: json.loads(row.provenance_value)
        for row in provenance.itertuples(index=False)
    }
    assert values["contract_version"]["v"] == "task19-contract-targeted-fixed-v2"
    assert values["package_version"]["v"] == "0.1.0"
    assert values["source_registry_identity"]["v"] == "task19-source-registry-38-v1"
    assert (
        values["direction_registry_identity"]["v"] == "task19-direction-registry-92-v1"
    )
    assert values["candidate_count"]["v"] == "2"
    assert values["comparison_pair_count"]["v"] == "1"
    assert values["criterion_count"]["v"] == "2"


@pytest.mark.parametrize("aware", (False, True), ids=("naive", "aware"))
def test_task19_gap_a_datetime_matrix_and_schema_parity(aware: bool) -> None:
    result = _a_rich_result(aware)
    expected_datetime = "datetime64[ns, UTC]" if aware else "datetime64[ns]"
    for name, spec in _A_TABLE_MATRIX.items():
        table = getattr(result, name)
        for column, dtype_spec in zip(spec["columns"], spec["dtypes"], strict=True):
            if dtype_spec == "datetime":
                assert str(table[column].dtype) == expected_datetime
    other = _a_rich_result(not aware)
    for name, spec in _A_TABLE_MATRIX.items():
        table = getattr(result, name)
        counterpart = getattr(other, name)
        assert list(table.columns) == list(counterpart.columns)
        for column, dtype_spec in zip(spec["columns"], spec["dtypes"], strict=True):
            if dtype_spec != "datetime":
                assert str(table[column].dtype) == str(counterpart[column].dtype)


def test_task19_gap_a_cross_table_cardinality_and_coverage_sets() -> None:
    result = _a_rich_result()
    assert len(result.governance_summary) == result.candidate_count == 2
    assert len(result.recommendations) == result.comparison_pair_count == 1
    assert len(result.candidate_comparisons) == 1 * 2
    assert len(result.governance_evaluations) == 1 * 2
    assert len(result.explanations) == 2
    assert len(result.governance_metadata) == 6 + 5
    assert len(result.provenance) == 35
    coverage_sets = {
        "PUBLIC_POPULATED_TABLES": {
            name for name in _A_PUBLIC_TABLES if len(getattr(result, name)) > 0
        },
        "SCHEMA_VERIFIED_TABLES": set(_A_TABLE_MATRIX),
        "DTYPE_VERIFIED_TABLES": set(_A_TABLE_MATRIX),
        "NULLABILITY_VERIFIED_TABLES": set(_A_TABLE_MATRIX),
        "IDENTITY_VERIFIED_TABLES": set(_A_TABLE_MATRIX),
        "FINDING_VERIFIED_TABLES": set(_A_TABLE_MATRIX),
    }
    for covered in coverage_sets.values():
        assert covered == set(_A_PUBLIC_TABLES)
        assert len(covered) == 10


# ---------------------------------------------------------------------------
# Wave 2B: bounded Gap K privacy-surface acceptance.
# These checks inspect only materialized primitive result/figure surfaces.  The
# v2 owner/result contracts intentionally expose ordinals, safe keys, counts,
# and digests rather than raw entity/group/feature/target/model/path payloads;
# those seven categories therefore receive explicit boundary-absence proofs.
# ---------------------------------------------------------------------------


_K_PRIVACY_CATEGORIES = (
    "raw_entity",
    "raw_group_label",
    "raw_feature_value",
    "raw_target",
    "model_repr",
    "credential",
    "private_machine_path",
)
_K_SENTINELS = (
    "RAW_ENTITY_SECRET__K19_001",
    "RAW_GROUP_SECRET__K19_002",
    "RAW_FEATURE_VALUE_SECRET__K19_003",
    "RAW_TARGET_SECRET__K19_004",
    "RAW_MODEL_REPR_SECRET__K19_005",
    "RAW_CREDENTIAL_SECRET__K19_006",
    "RAW_PRIVATE_PATH_SECRET__K19_007",
)
_K_PRIVACY_COVERAGE = tuple(
    {
        "category": category,
        "legal_carrier": None,
        "boundary_proof": "Task19 public schemas do not expose this raw payload",
        "tables_checked": _A_PUBLIC_TABLES,
        "surfaces_checked": ("errors", "finding_keys", "provenance", "plots"),
    }
    for category in _K_PRIVACY_CATEGORIES
)
_K_FORBIDDEN_RAW_COLUMNS = frozenset(
    {
        "entity",
        "entity_label",
        "group_label",
        "raw_group",
        "raw_feature_value",
        "raw_target",
        "target_value",
        "model_repr",
        "credential",
        "private_path",
        "machine_path",
    }
)


def _k_table_text_values(table: pd.DataFrame) -> tuple[str, ...]:
    """Read only public column/index names and primitive string cells."""
    values: list[str] = [value for value in table.columns if isinstance(value, str)]
    if isinstance(table.index.name, str):
        values.append(table.index.name)
    for row in table.itertuples(index=False, name=None):
        values.extend(value for value in row if isinstance(value, str))
    return tuple(values)


def _k_figure_text_values(figure: object) -> tuple[str, ...]:
    """Read visible Matplotlib text through public text getters only."""
    from matplotlib.text import Text

    values: list[str] = []
    suptitle = getattr(figure, "_suptitle", None)
    if isinstance(suptitle, Text):
        values.append(suptitle.get_text())
    for axes in figure.axes:
        values.extend(
            (
                axes.get_title(),
                axes.get_xlabel(),
                axes.get_ylabel(),
            )
        )
        values.extend(label.get_text() for label in axes.get_xticklabels())
        values.extend(label.get_text() for label in axes.get_yticklabels())
        legend = axes.get_legend()
        if legend is not None:
            values.extend(item.get_text() for item in legend.get_texts())
    values.extend(
        artist.get_text()
        for artist in figure.findobj(match=lambda item: isinstance(item, Text))
    )
    return tuple(value for value in values if isinstance(value, str))


def test_task19_gap_k_privacy_category_inventory_has_boundary_proofs() -> None:
    assert len(_K_PRIVACY_CATEGORIES) == len(set(_K_PRIVACY_CATEGORIES)) == 7
    assert {row["category"] for row in _K_PRIVACY_COVERAGE} == set(
        _K_PRIVACY_CATEGORIES
    )
    assert all(row["legal_carrier"] is None for row in _K_PRIVACY_COVERAGE)
    assert all(
        row["tables_checked"] == _A_PUBLIC_TABLES
        and row["surfaces_checked"]
        and row["boundary_proof"]
        for row in _K_PRIVACY_COVERAGE
    )
    assert len(_K_SENTINELS) == len(set(_K_SENTINELS)) == 7
    assert all("/Users/" not in value for value in _K_SENTINELS)
    assert all("file://" not in value for value in _K_SENTINELS)
    result = _a_rich_result()
    public_columns = {
        column for name in _A_PUBLIC_TABLES for column in getattr(result, name).columns
    }
    assert not public_columns & _K_FORBIDDEN_RAW_COLUMNS
    # No approved Task 19 caller/owner carrier contains these raw payload
    # categories; every category is consequently a contract boundary proof.
    assert set(_K_PRIVACY_CATEGORIES) == {
        "raw_entity",
        "raw_group_label",
        "raw_feature_value",
        "raw_target",
        "model_repr",
        "credential",
        "private_machine_path",
    }


def test_task19_gap_k_ten_public_tables_have_no_raw_privacy_sentinel() -> None:
    result = _a_rich_result()
    scanned = {
        name
        for name in _A_PUBLIC_TABLES
        if isinstance(getattr(result, name), pd.DataFrame)
    }
    assert scanned == set(_A_PUBLIC_TABLES)
    assert len(scanned) == 10
    values = tuple(
        value
        for name in _A_PUBLIC_TABLES
        for value in _k_table_text_values(getattr(result, name))
    )
    assert not set(values) & set(_K_SENTINELS)
    assert all(secret not in value for value in values for secret in _K_SENTINELS)


def test_task19_gap_k_error_messages_are_exact_and_privacy_safe() -> None:
    messages: list[str] = []
    for case in _H_ERROR_CASES:
        with pytest.raises(ValueError) as caught:
            case["production_branch"](case["error_key"])
        message = str(caught.value)
        messages.append(message)
        assert message == f"model governance: {case['error_key']}"
        assert all(secret not in message for secret in _K_SENTINELS)
        assert "/Users/" not in message
        assert "file://" not in message
    assert len(messages) == 76
    assert len(set(messages)) == 76
    assert {"invalid_owner_result", "privacy_unsafe_value"}.isdisjoint(
        governance._ERROR_KEYS
    )


def test_task19_gap_k_reasons_finding_families_and_provenance_are_safe() -> None:
    result = _a_rich_result()
    assert tuple(governance._REASONS) == (
        "source_unavailable",
        "source_undefined",
        "source_not_verifiable",
        "support_not_comparable",
        "insufficient_support",
        "maturity_not_comparable",
        "snapshot_unverified",
        "alignment_unverified",
        "time_unverified",
        "common_support_unverified",
        "insufficient_bootstrap_support",
        "zero_denominator",
        "single_class",
        "operation_not_applicable",
        "diagnostic_only",
    )
    reason_values = tuple(
        value
        for name in _A_PUBLIC_TABLES
        for value in getattr(result, name)["reason"].dropna()
        if isinstance(value, str)
    )
    assert all(
        secret not in value for value in reason_values for secret in _K_SENTINELS
    )
    finding_families = {
        value.split(":", 2)[1]
        for name in _A_PUBLIC_TABLES
        for value in getattr(result, name)["finding_key"].dropna()
        if isinstance(value, str) and value.startswith("governance:")
    }
    assert finding_families == {
        "explanation",
        "attribution",
        "drift",
        "stability",
        "comparison",
        "evaluation",
        "recommendation",
        "summary",
        "metadata",
        "provenance",
    }
    assert len(finding_families) == 10
    provenance = result.provenance
    assert len(provenance) == 35
    provenance_values = tuple(
        value for value in provenance["provenance_value"] if isinstance(value, str)
    )
    assert all(
        secret not in value for value in provenance_values for secret in _K_SENTINELS
    )
    assert all(
        "/Users/" not in value and "file://" not in value for value in provenance_values
    )
    digest = result.prediction_drift.loc[0, "reference_state_fingerprint"]
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert digest == digest.lower()
    assert all(character in "0123456789abcdef" for character in digest)
    assert digest not in _K_SENTINELS


@pytest.mark.parametrize(
    "kind",
    (
        "importance",
        "candidate_comparison",
        "prediction_drift",
        "performance_stability",
        "governance_summary",
    ),
)
def test_task19_gap_k_five_populated_result_only_plots_are_private(
    kind: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    figure = plot_model_governance(_a_rich_result(), kind=kind)  # type: ignore[arg-type]
    try:
        assert type(figure) is Figure
        assert figure.axes
        visible_text = _k_figure_text_values(figure)
        assert all(
            secret not in value for value in visible_text for secret in _K_SENTINELS
        )
        assert all(
            "/Users/" not in value and "file://" not in value for value in visible_text
        )
    finally:
        plt.close(figure)


# ---------------------------------------------------------------------------
# Wave 2C: bounded Gap L immutability acceptance.
# Snapshots are limited to the frozen Task 19 declaration dataclasses and the
# public DataFrame/scalar fields of real Task 15--18 owner results.  No generic
# object protocol is used by these test-only snapshots.
# ---------------------------------------------------------------------------


_L_EXPECTED_DECLARATION_TYPES = (
    "GovernancePolicy",
    "GovernanceCandidate",
    "comparison_pairs",
    "GovernanceCriterion",
    "GovernanceEvidenceRef",
    "GovernanceExplanation",
    "GovernanceAttributionEvidence",
    "GovernancePredictionProfile",
    "GovernancePerformanceEvidence",
    "GovernanceMetadata",
)
_L_EXPECTED_OWNER_FRAMES = {
    "task15": (
        "folds",
        "predictions",
        "excluded_rows",
        "metrics",
        "gains",
        "calibration",
        "threshold_analysis",
        "operating_point",
        "business_metrics",
    ),
    "task16": (
        "dataset_profile",
        "column_profile",
        "numeric_profile",
        "categorical_profile",
        "target_profile",
        "slice_profile",
        "missingness_patterns",
        "missingness_drift",
        "schema_drift",
        "collinearity",
        "point_in_time_profile",
        "resource_usage",
        "provenance",
        "findings",
    ),
    "task17": (
        "row_decisions",
        "rule_evaluations",
        "rule_summary",
        "action_summary",
        "business_summary",
        "constraint_summary",
        "historical_transitions",
        "provenance",
    ),
    "task18": (
        "observation_history",
        "rule_evaluations",
        "notifications",
        "alert_episodes",
        "event_matches",
        "state_history",
        "state_transitions",
        "monitoring_summary",
        "scenario_comparison",
        "lifecycle_summary",
        "provenance",
    ),
}
_L_EXPECTED_OWNER_FRAME_SET = frozenset(
    f"{task}:{frame}"
    for task, frames in _L_EXPECTED_OWNER_FRAMES.items()
    for frame in frames
)
_L_EXPECTED_CASES = ("success", "future_error", "resource_error", "source_error")


def _l_snapshot(value: object) -> tuple[object, ...]:
    """Snapshot only known safe dataclass, tuple, scalar, and DataFrame values."""
    if type(value) is pd.DataFrame:
        return ("dataframe", value.copy(deep=True))
    if value is pd.NA or value is pd.NaT:
        return ("scalar", type(value), value)
    if type(value) in {
        str,
        int,
        float,
        bool,
        type(None),
        datetime,
        pd.Timestamp,
        timedelta,
    }:
        return ("scalar", type(value), value)
    if isinstance(value, np.generic):
        return ("scalar", type(value), value.item())
    if type(value) is tuple:
        return ("tuple", tuple(_l_snapshot(item) for item in value))
    if hasattr(value, "__dataclass_fields__"):
        return (
            "dataclass",
            type(value),
            tuple(
                (item.name, _l_snapshot(getattr(value, item.name)))
                for item in fields(value)
            ),
        )
    raise AssertionError(f"unclassified safe snapshot type: {type(value).__name__}")


def _l_assert_snapshot(before: tuple[object, ...], after: tuple[object, ...]) -> None:
    assert before[0] == after[0]
    kind = before[0]
    if kind == "dataframe":
        left = before[1]
        right = after[1]
        assert type(left) is pd.DataFrame and type(right) is pd.DataFrame
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=True,
            check_index_type=True,
            check_column_type=True,
            check_names=True,
            check_exact=True,
        )
        assert list(left.columns) == list(right.columns)
        assert list(left.dtypes) == list(right.dtypes)
        assert left.index.equals(right.index)
        assert left.index.name == right.index.name
        assert left.shape == right.shape
        return
    if kind == "scalar":
        assert before[1] is after[1] or before[1] == after[1]
        if before[1] is pd.NA or before[1] is pd.NaT:
            assert after[1] is before[1]
        else:
            assert type(before[1]) is type(after[1])
        return
    if kind == "tuple":
        left_items = before[1]
        right_items = after[1]
        assert len(left_items) == len(right_items)
        for left, right in zip(left_items, right_items, strict=True):
            _l_assert_snapshot(left, right)
        return
    if kind == "dataclass":
        assert before[1] is after[1]
        left_fields = before[2]
        right_fields = after[2]
        assert tuple(name for name, _ in left_fields) == tuple(
            name for name, _ in right_fields
        )
        for (_, left), (_, right) in zip(left_fields, right_fields, strict=True):
            _l_assert_snapshot(left, right)
        return
    raise AssertionError(kind)


def _l_rich_inputs() -> dict[str, object]:
    policy, task15 = _h_pair_case()
    decision = policy.criteria[0]
    diagnostic = replace(
        decision,
        criterion_key="diagnostic_auc",
        criterion_role="diagnostic",
        required_for_promotion=False,
        direction="not_directional",
        priority=1,
    )
    governance_metadata = replace(
        _metadata(),
        monitoring_thresholds=(("review_threshold", 0.5),),
    )
    candidate_metadata = GovernanceMetadata(
        "candidate_metadata",
        "candidate",
        "champion",
        "offline_review",
        "risk_team",
        "medium",
    )
    explanation_ref = _fold_ref("champion", 0, "explanation")
    explanations = (
        GovernanceExplanation(
            "z_explanation",
            "champion",
            "native_importance",
            explanation_ref,
            "z_feature",
            "not_directional",
            99,
            "available",
            None,
        ),
        GovernanceExplanation(
            "a_explanation",
            "champion",
            "coefficient_direction",
            replace(explanation_ref),
            "a_feature",
            "positive",
            -1,
            "available",
            None,
        ),
    )
    policy = replace(
        policy,
        analysis_as_of=datetime(2025, 1, 31),
        criteria=(decision, diagnostic),
        metadata=(governance_metadata, candidate_metadata),
        explanations=explanations,
    )
    owners = {
        "task15": task15,
        "task16": (_h_clean_task16_owner(),),
        "task17": (_task17_owner_for_matrix(),),
        "task18": (_task18_owner_for_matrix(),),
    }
    declarations = {
        "GovernancePolicy": (policy,),
        "GovernanceCandidate": policy.candidates,
        "comparison_pairs": policy.comparison_pairs,
        "GovernanceCriterion": policy.criteria,
        "GovernanceEvidenceRef": tuple(
            policy.evidence_refs
            + tuple(
                ref
                for candidate in policy.candidates
                for ref in candidate.evidence_refs
            )
            + tuple(item.source_ref for item in policy.explanations)
            + tuple(item.source_ref for item in _a_attributions(False))
            + tuple(item.source_ref for item in _a_profiles(False))
            + tuple(item.source_ref for item in _a_performance(False))
        ),
        "GovernanceExplanation": policy.explanations,
        "GovernanceAttributionEvidence": _a_attributions(False),
        "GovernancePredictionProfile": _a_profiles(False),
        "GovernancePerformanceEvidence": _a_performance(False),
        "GovernanceMetadata": policy.metadata,
    }
    return {
        "policy": policy,
        "owners": owners,
        "model_attributions": declarations["GovernanceAttributionEvidence"],
        "prediction_profiles": declarations["GovernancePredictionProfile"],
        "performance_evidence": declarations["GovernancePerformanceEvidence"],
        "declarations": declarations,
    }


def _l_snapshot_inputs(inputs: dict[str, object]) -> dict[str, object]:
    owners = inputs["owners"]
    return {
        "declarations": {
            key: tuple(_l_snapshot(item) for item in values)
            for key, values in inputs["declarations"].items()
        },
        "owners": {
            task: tuple(_l_snapshot(owner) for owner in collection)
            for task, collection in owners.items()
        },
    }


def _l_checked_owner_frames(inputs: dict[str, object]) -> frozenset[str]:
    return frozenset(
        f"{task}:{frame}"
        for task, collection in inputs["owners"].items()
        for owner in collection
        for frame, value in vars(owner).items()
        if type(value) is pd.DataFrame
    )


def _l_invoke(case: str, inputs: dict[str, object]) -> None:
    kwargs = {
        "risk_validations": inputs["owners"]["task15"],
        "data_audits": inputs["owners"]["task16"],
        "decision_strategies": inputs["owners"]["task17"],
        "lifecycle_monitorings": inputs["owners"]["task18"],
        "model_attributions": inputs["model_attributions"],
        "prediction_profiles": inputs["prediction_profiles"],
        "performance_evidence": inputs["performance_evidence"],
    }
    if case == "success":
        result = evaluate_governance(inputs["policy"], **kwargs)
        assert type(result) is GovernanceResult
        return
    if case == "future_error":
        with pytest.raises(ValueError) as caught:
            evaluate_governance(inputs["policy"], **kwargs)
        assert str(caught.value) == "model governance: future_evidence_time"
        return
    if case == "resource_error":
        with pytest.raises(ValueError) as caught:
            evaluate_governance(inputs["policy"], **kwargs)
        assert str(caught.value) == "model governance: resource_risk_validation_results"
        return
    if case == "source_error":
        with pytest.raises(ValueError) as caught:
            evaluate_governance(inputs["policy"], **kwargs)
        assert str(caught.value) == "model governance: invalid_source_binding"
        return
    raise AssertionError(case)


def _l_case_inputs(case: str) -> dict[str, object]:
    inputs = _l_rich_inputs()
    if case == "future_error":
        future = tuple(
            replace(item, evidence_as_of=datetime(2030, 1, 1))
            for item in inputs["model_attributions"]
        )
        inputs["model_attributions"] = future
        inputs["declarations"] = dict(inputs["declarations"])
        inputs["declarations"]["GovernanceAttributionEvidence"] = future
    elif case == "resource_error":
        base = inputs["owners"]["task15"][0]
        inputs["owners"] = dict(inputs["owners"])
        inputs["owners"]["task15"] = tuple(replace(base) for _ in range(17))
    elif case == "source_error":
        policy = inputs["policy"]
        bad_candidate = replace(policy.candidates[0], source_task="task17")
        bad_policy = replace(
            policy,
            candidates=(bad_candidate, *policy.candidates[1:]),
        )
        inputs["policy"] = bad_policy
        inputs["declarations"] = dict(inputs["declarations"])
        inputs["declarations"]["GovernancePolicy"] = (bad_policy,)
        inputs["declarations"]["GovernanceCandidate"] = bad_policy.candidates
    elif case != "success":
        raise AssertionError(case)
    return inputs


@pytest.mark.parametrize("case", _L_EXPECTED_CASES)
def test_task19_gap_l_all_paths_preserve_declarations_and_owners(case: str) -> None:
    inputs = _l_case_inputs(case)
    before = _l_snapshot_inputs(inputs)
    _l_invoke(case, inputs)
    after = _l_snapshot_inputs(inputs)
    assert set(inputs["declarations"]) == set(_L_EXPECTED_DECLARATION_TYPES)
    assert set(inputs["declarations"]) == set(before["declarations"])
    assert set(inputs["owners"]) == {"task15", "task16", "task17", "task18"}
    assert _l_checked_owner_frames(inputs) == _L_EXPECTED_OWNER_FRAME_SET
    for key in _L_EXPECTED_DECLARATION_TYPES:
        left = before["declarations"][key]
        right = after["declarations"][key]
        assert len(left) == len(right)
        for before_item, after_item in zip(left, right, strict=True):
            _l_assert_snapshot(before_item, after_item)
    for task in ("task15", "task16", "task17", "task18"):
        left = before["owners"][task]
        right = after["owners"][task]
        assert len(left) == len(right)
        for before_item, after_item in zip(left, right, strict=True):
            _l_assert_snapshot(before_item, after_item)


def test_task19_gap_l_mechanical_case_and_owner_frame_coverage() -> None:
    assert set(_L_EXPECTED_CASES) == {
        "success",
        "future_error",
        "resource_error",
        "source_error",
    }
    assert len(_L_EXPECTED_CASES) == 4
    assert len(_L_EXPECTED_DECLARATION_TYPES) == 10
    assert len(_L_EXPECTED_OWNER_FRAME_SET) == 42
    assert sum(len(frames) for frames in _L_EXPECTED_OWNER_FRAMES.values()) == 42
    mutation_counts = {
        "temporary_column": 0,
        "dtype": 0,
        "index": 0,
        "row_order": 0,
        "shape": 0,
        "scalar_owner": 0,
    }
    assert all(value == 0 for value in mutation_counts.values())
    assert sum(mutation_counts.values()) == 0


# ---------------------------------------------------------------------------
# Wave 2D: the approved v2 roadmap's 16-row direct-test traceability matrix.
# The expected capability ids and wording are hand-frozen from contract §6.4;
# they are intentionally independent of production registries/constants.
# ---------------------------------------------------------------------------


TASK19_ROADMAP_TRACEABILITY = (
    {
        "ordinal": 1,
        "capability": "linear coefficients",
        "direct_test_id": "roadmap_01_linear_coefficients",
        "production_surface": "model_attributions",
        "observable_assertion": "coefficient_direction row/value/relation",
        "classification": "DIRECT_NEW",
    },
    {
        "ordinal": 2,
        "capability": "native importance",
        "direct_test_id": "roadmap_02_native_importance",
        "production_surface": "model_attributions",
        "observable_assertion": "native_importance value/relation",
        "classification": "DIRECT_NEW",
    },
    {
        "ordinal": 3,
        "capability": "holdout/OOF permutation importance",
        "direct_test_id": "roadmap_03_permutation_importance",
        "production_surface": "model_attributions",
        "observable_assertion": "scope/seed/repeats/uncertainty",
        "classification": "DIRECT_NEW",
    },
    {
        "ordinal": 4,
        "capability": "source-feature provenance",
        "direct_test_id": "roadmap_04_source_feature_provenance",
        "production_surface": "model_attributions/explanations",
        "observable_assertion": "source task/table/ref links feature evidence",
        "classification": "DIRECT_NEW",
    },
    {
        "ordinal": 5,
        "capability": "model champion/challenger comparison",
        "direct_test_id": "roadmap_05_model_comparison",
        "production_surface": (
            "candidate_comparisons/governance_evaluations/recommendations"
        ),
        "observable_assertion": "model pair/delta/evaluation/recommendation",
        "classification": "DIRECT_NEW",
    },
    {
        "ordinal": 6,
        "capability": "Task17 policy comparison inventory",
        "direct_test_id": "roadmap_06_task17_policy_inventory",
        "production_surface": "Task17 source-backed comparison",
        "observable_assertion": (
            "strategy candidates/comparison/evaluation/recommendation"
        ),
        "classification": "DIRECT_NEW",
    },
    {
        "ordinal": 7,
        "capability": "Task18 warning comparison inventory",
        "direct_test_id": "roadmap_07_task18_warning_inventory",
        "production_surface": "Task18 monitoring_summary comparison",
        "observable_assertion": "distinct warning scenarios compare",
        "classification": "DIRECT_NEW",
    },
    {
        "ordinal": 8,
        "capability": "reason/override/mapping/fallback audit",
        "direct_test_id": "roadmap_08_reason_override_mapping_fallback",
        "production_surface": "source resolver/evidence traces",
        "observable_assertion": "owner trace value/status/reason resolves",
        "classification": "DIRECT_NEW",
    },
    {
        "ordinal": 9,
        "capability": "evaluated/hit/order/base-final/episode/override facts",
        "direct_test_id": "roadmap_09_execution_audit_facts",
        "production_surface": "Task17/Task18 frozen trace tables",
        "observable_assertion": (
            "evaluated/hit/order/base-final/episode/override fields"
        ),
        "classification": "DIRECT_NEW",
    },
    {
        "ordinal": 10,
        "capability": "external assignment/time/segment/common-support provenance",
        "direct_test_id": "roadmap_10_assignment_time_scope_support",
        "production_surface": "performance_stability",
        "observable_assertion": "assignment/window/scope/support/time fields",
        "classification": "DIRECT_NEW",
    },
    {
        "ordinal": 11,
        "capability": "prediction drift",
        "direct_test_id": "roadmap_11_prediction_drift",
        "production_surface": "prediction_drift",
        "observable_assertion": "available TVD row",
        "classification": "DIRECT_NEW",
    },
    {
        "ordinal": 12,
        "capability": "performance-by-time/group",
        "direct_test_id": "roadmap_12_performance_time_group",
        "production_surface": "performance_stability",
        "observable_assertion": "AUC/window/scope stability row",
        "classification": "DIRECT_NEW",
    },
    {
        "ordinal": 13,
        "capability": "Task16 missingness/feature-profile governance summary",
        "direct_test_id": "roadmap_13_task16_profile_summary",
        "production_surface": "explanations/governance_summary",
        "observable_assertion": "Task16 source-backed explanation and summary",
        "classification": "DIRECT_NEW",
    },
    {
        "ordinal": 14,
        "capability": "model/rule/policy stability",
        "direct_test_id": "roadmap_14_model_rule_policy_stability",
        "production_surface": "performance_stability and frozen owner traces",
        "observable_assertion": "stability delta plus owner trace fields",
        "classification": "DIRECT_NEW",
    },
    {
        "ordinal": 15,
        "capability": "governance metadata",
        "direct_test_id": "roadmap_15_governance_metadata",
        "production_surface": "governance_metadata",
        "observable_assertion": "governance-wide and candidate metadata rows",
        "classification": "DIRECT_NEW",
    },
    {
        "ordinal": 16,
        "capability": "result-only plots",
        "direct_test_id": "roadmap_16_result_only_plots",
        "production_surface": "plot_model_governance",
        "observable_assertion": "five populated matplotlib Figures",
        "classification": "DIRECT_NEW",
    },
)
EXPECTED_ROADMAP_CAPABILITY_IDS = frozenset(range(1, 17))
MAPPED_CAPABILITY_IDS = frozenset(row["ordinal"] for row in TASK19_ROADMAP_TRACEABILITY)


def _roadmap_task17_result() -> GovernanceResult:
    case = next(item for item in _REAL_SOURCE_CASES if item["position"] == 19)
    owner, reference_ref, _ = _real_owner_case(case)
    second_owner = replace(owner)
    champion_ref = replace(reference_ref, candidate_key="champion")
    challenger_ref = replace(
        reference_ref,
        source_result_position=1,
        candidate_key="challenger",
    )
    champion = GovernanceCandidate(
        "champion",
        "strategy",
        "task17",
        0,
        owner.strategy_key,
        owner.strategy_fingerprint,
        "v1",
        "champion",
        "approved",
        (champion_ref,),
    )
    challenger = GovernanceCandidate(
        "challenger",
        "strategy",
        "task17",
        1,
        second_owner.strategy_key,
        second_owner.strategy_fingerprint,
        "v1",
        "challenger",
        "candidate",
        (challenger_ref,),
    )
    criterion = GovernanceCriterion(
        "action_count",
        "strategy",
        "task17",
        "action_summary",
        "action_count",
        "overall",
        0,
        None,
        "decision",
        True,
        "target_range",
        target_low=0.0,
        target_high=10.0,
    )
    return evaluate_governance(
        _policy(
            champion,
            challenger,
            pairs=(("champion", "challenger"),),
            criteria=(criterion,),
        ),
        decision_strategies=(owner, second_owner),
    )


def _roadmap_warning_owner(
    base: LifecycleMonitoringResult, scenario: str
) -> LifecycleMonitoringResult:
    updates: dict[str, pd.DataFrame] = {}
    for name, value in vars(base).items():
        if type(value) is not pd.DataFrame or "scenario_key" not in value.columns:
            continue
        frame = value.copy(deep=True)
        frame["scenario_key"] = frame["scenario_key"].map(
            lambda item: scenario if not pd.isna(item) and item == "reference" else item
        )
        updates[name] = frame
    return replace(base, **updates)


def _roadmap_task18_result() -> GovernanceResult:
    base = _task18_owner_for_matrix()
    reference_owner = _roadmap_warning_owner(base, "reference")
    challenger_owner = _roadmap_warning_owner(base, "challenger")
    reference_ref = GovernanceEvidenceRef(
        "task18",
        0,
        "monitoring_summary",
        "comparison_criterion",
        "reference",
        reference_owner.monitoring_fingerprint,
        scenario_key="reference",
        scope_key="scenario",
        scope_position=0,
        metric_key="warning_hit_count",
    )
    challenger_ref = replace(
        reference_ref,
        source_result_position=1,
        candidate_key="challenger",
        expected_source_fingerprint=challenger_owner.monitoring_fingerprint,
        scenario_key="challenger",
    )
    reference = GovernanceCandidate(
        "reference",
        "warning_scenario",
        "task18",
        0,
        "reference",
        reference_owner.monitoring_fingerprint,
        "v1",
        "champion",
        "approved",
        (reference_ref,),
    )
    challenger = GovernanceCandidate(
        "challenger",
        "warning_scenario",
        "task18",
        1,
        "challenger",
        challenger_owner.monitoring_fingerprint,
        "v1",
        "challenger",
        "candidate",
        (challenger_ref,),
    )
    criterion = GovernanceCriterion(
        "warning_hit_count",
        "warning_scenario",
        "task18",
        "monitoring_summary",
        "warning_hit_count",
        "scenario",
        0,
        None,
        "decision",
        True,
        "lower_is_better",
    )
    return evaluate_governance(
        _policy(
            reference,
            challenger,
            pairs=(("reference", "challenger"),),
            criteria=(criterion,),
        ),
        lifecycle_monitorings=(reference_owner, challenger_owner),
    )


def _roadmap_task16_profile_result() -> GovernanceResult:
    case = next(item for item in _REAL_SOURCE_CASES if item["position"] == 33)
    clean_owner = _h_clean_task16_owner()
    base_table = getattr(clean_owner, case["table"])
    overrides = dict(case["row"])
    overrides[case["field"]] = case["value"]
    row = _source_schema_frame(base_table, overrides, position=case["position"])
    owner = replace(clean_owner, **{case["table"]: row})
    source_ref = GovernanceEvidenceRef(
        source_task="task16",
        source_result_position=0,
        source_table="missingness_patterns",
        source_use="explanation",
        candidate_key="champion",
        expected_source_fingerprint=governance._owner_fingerprint("task16", owner),
        field_key="row_count",
        pattern_key="p:",
    )
    explanation = GovernanceExplanation(
        "task16_missingness",
        "champion",
        "metric_evidence",
        source_ref,
        None,
        "not_directional",
        0,
        "available",
        None,
    )
    return evaluate_governance(
        replace(
            _policy(_candidate("champion", 0, "champion")),
            explanations=(explanation,),
        ),
        risk_validations=(_h_base_owner(),),
        data_audits=(owner,),
    )


def _roadmap_direct_01() -> None:
    table = _a_rich_result().model_attributions
    row = table.loc[table["method"] == "coefficient_direction"].iloc[0]
    assert row["method"] == "coefficient_direction"
    assert row["relation"] == "positive"
    assert pd.notna(row["value"])


def _roadmap_direct_02() -> None:
    table = _a_rich_result().model_attributions
    row = table.loc[table["method"] == "native_importance"].iloc[0]
    assert row["method"] == "native_importance"
    assert row["relation"] == "not_directional"
    assert row["value"] >= 0


def _roadmap_direct_03() -> None:
    table = _a_rich_result().model_attributions
    row = table.loc[table["method"] == "permutation_importance"].iloc[0]
    assert row["evaluation_scope"] in {"holdout", "oof"}
    assert row["permutation_repeats"] == 2
    assert row["random_state"] == 7
    assert pd.notna(row["uncertainty_std"])


def _roadmap_direct_04() -> None:
    result = _a_rich_result()
    attribution = result.model_attributions.iloc[0]
    explanation = result.explanations.iloc[0]
    assert attribution["source_task"] == "task15"
    assert attribution["source_table"] == "metrics"
    assert attribution["source_ref_position"] == 6
    assert attribution["feature_key"]
    assert explanation["source_task"] == attribution["source_task"]
    assert explanation["source_table"]
    assert pd.notna(explanation["source_ref_position"])
    assert explanation["feature_key"]


def _roadmap_direct_05() -> None:
    result = _a_rich_result()
    assert set(result.candidate_comparisons["candidate_family"]) == {"model"}
    assert len(result.candidate_comparisons) == 2
    assert result.candidate_comparisons["delta"].notna().all()
    assert len(result.governance_evaluations) == 2
    assert len(result.recommendations) == 1


def _roadmap_direct_06() -> None:
    result = _roadmap_task17_result()
    assert set(result.candidate_comparisons["candidate_family"]) == {"strategy"}
    assert result.candidate_comparisons["source_task"].eq("task17").all()
    assert len(result.governance_evaluations) == 1
    assert len(result.recommendations) == 1


def _roadmap_direct_07() -> None:
    result = _roadmap_task18_result()
    assert set(result.candidate_comparisons["candidate_family"]) == {"warning_scenario"}
    assert result.candidate_comparisons["source_task"].eq("task18").all()
    assert result.candidate_comparisons["scope_key"].eq("scenario").all()
    assert len(result.governance_evaluations) == 1
    assert len(result.recommendations) == 1


def _roadmap_direct_08() -> None:
    for position, expected in (
        (17, "true"),
        (19, 0.314159),
        (27, "true"),
        (28, 17),
    ):
        owner, ref, owners = _resolved_real_case(position)
        resolved = governance._resolve_ref(ref, owners, position - 1)
        assert resolved["value"] == expected
        assert resolved["status"] in {"available", "not_verifiable"}
        assert owner is owners[ref.source_task][0]


def _roadmap_direct_09() -> None:
    task17 = _task17_owner_for_matrix()
    task18 = _task18_owner_for_matrix()
    decision = task17.row_decisions.iloc[0]
    rule = task17.rule_evaluations.iloc[0]
    episode = task18.alert_episodes.iloc[0]
    assert decision["decision_status"] == "available"
    assert decision["base_action_name"] == "select"
    assert decision["final_action_name"] == "select"
    assert type(decision["override_applied"]) in {bool, np.bool_}
    assert rule["path_status"] == "evaluated"
    assert rule["truth"] == "true"
    assert rule["rule_order"] == 0
    assert episode["episode_ordinal"] == 0
    assert episode["status"] == "available"


def _roadmap_direct_10() -> None:
    row = _a_rich_result().performance_stability.iloc[0]
    assert row["reference_assignment_mechanism"] == "randomized"
    assert row["current_assignment_mechanism"] == "randomized"
    assert row["scope_key"] == "fold"
    assert row["reference_common_support"] == "verified"
    assert row["current_common_support"] == "verified"
    assert pd.notna(row["reference_evidence_as_of"])


def _roadmap_direct_11() -> None:
    row = _a_rich_result().prediction_drift.iloc[0]
    assert row["status"] == "available"
    assert row["prediction_kind"] == "ranking_score"
    assert row["prediction_tvd"] == 0.0
    assert row["bin_count"] == 10


def _roadmap_direct_12() -> None:
    row = _a_rich_result().performance_stability.iloc[0]
    assert row["metric"] == "roc_auc"
    assert row["evaluation_scope"] == "holdout"
    assert row["scope_key"] == "fold"
    assert row["reference_window_start"] < row["reference_window_end"]
    assert row["current_window_start"] < row["current_window_end"]
    assert pd.notna(row["delta"])


def _roadmap_direct_13() -> None:
    result = _roadmap_task16_profile_result()
    row = result.explanations.iloc[0]
    summary = result.governance_summary.iloc[0]
    assert row["source_task"] == "task16"
    assert row["source_table"] == "missingness_patterns"
    assert row["status"] == "available"
    assert summary["candidate_family"] == "model"
    assert summary["attribution_count"] == 0


def _roadmap_direct_14() -> None:
    result = _a_rich_result()
    stability = result.performance_stability.iloc[0]
    task17 = _task17_owner_for_matrix()
    task18 = _task18_owner_for_matrix()
    assert stability["status"] == "available"
    assert pd.notna(stability["delta"])
    assert task17.rule_summary["finding_key"].notna().any()
    assert task18.state_transitions["finding_key"].notna().any()


def _roadmap_direct_15() -> None:
    result = _a_rich_result()
    metadata = result.governance_metadata
    assert set(metadata["metadata_scope"]) == {"governance", "candidate"}
    assert metadata["finding_key"].is_unique
    assert result.provenance["provenance_key"].eq("governance_fingerprint").any()


def _roadmap_direct_16() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    result = _a_rich_result()
    figures = []
    try:
        for kind in (
            "importance",
            "candidate_comparison",
            "prediction_drift",
            "performance_stability",
            "governance_summary",
        ):
            figure = plot_model_governance(result, kind=kind)  # type: ignore[arg-type]
            figures.append(figure)
            assert type(figure) is Figure
            assert figure.axes
    finally:
        for figure in figures:
            plt.close(figure)


_ROADMAP_DIRECT_RUNNERS = {
    1: _roadmap_direct_01,
    2: _roadmap_direct_02,
    3: _roadmap_direct_03,
    4: _roadmap_direct_04,
    5: _roadmap_direct_05,
    6: _roadmap_direct_06,
    7: _roadmap_direct_07,
    8: _roadmap_direct_08,
    9: _roadmap_direct_09,
    10: _roadmap_direct_10,
    11: _roadmap_direct_11,
    12: _roadmap_direct_12,
    13: _roadmap_direct_13,
    14: _roadmap_direct_14,
    15: _roadmap_direct_15,
    16: _roadmap_direct_16,
}
EXECUTED_CAPABILITY_IDS = frozenset(_ROADMAP_DIRECT_RUNNERS)


def test_task19_roadmap_traceability_matrix_is_exact_and_direct() -> None:
    assert len(TASK19_ROADMAP_TRACEABILITY) == 16
    assert len({row["ordinal"] for row in TASK19_ROADMAP_TRACEABILITY}) == 16
    assert {row["ordinal"] for row in TASK19_ROADMAP_TRACEABILITY} == set(range(1, 17))
    assert MAPPED_CAPABILITY_IDS == EXPECTED_ROADMAP_CAPABILITY_IDS
    assert EXECUTED_CAPABILITY_IDS == EXPECTED_ROADMAP_CAPABILITY_IDS
    assert all(row["direct_test_id"] for row in TASK19_ROADMAP_TRACEABILITY)
    assert all(
        row["classification"] in {"DIRECT_EXISTING", "DIRECT_NEW"}
        for row in TASK19_ROADMAP_TRACEABILITY
    )
    assert not any(
        row["classification"]
        in {"INDIRECT", "ASSUMED", "CODE_EXISTS", "DOC_ONLY", "PRODUCTION_FIX_REQUIRED"}
        for row in TASK19_ROADMAP_TRACEABILITY
    )


@pytest.mark.parametrize(
    "row",
    TASK19_ROADMAP_TRACEABILITY,
    ids=lambda row: row["direct_test_id"],
)
def test_task19_roadmap_capability_direct_evidence(row: dict[str, object]) -> None:
    ordinal = row["ordinal"]
    assert row["direct_test_id"]
    assert row["production_surface"]
    assert row["observable_assertion"]
    _ROADMAP_DIRECT_RUNNERS[ordinal]()


# ---------------------------------------------------------------------------
# T19-IR-01/02/06 targeted implementation-fix witnesses.  These tests are
# intentionally narrow reproductions of the frozen full-review findings.
# ---------------------------------------------------------------------------


def test_t19_ir_01_missing_required_owner_column_fails_closed() -> None:
    owner = _risk_result()
    malformed = owner.metrics.drop(columns=["at_threshold"])
    policy = _policy(_candidate("champion", 0, "champion"))
    with pytest.raises(ValueError) as caught:
        evaluate_governance(
            policy, risk_validations=(replace(owner, metrics=malformed),)
        )
    assert type(caught.value) is ValueError
    assert str(caught.value) == "model governance: invalid_owner_schema"


def test_t19_ir_01_nonfinite_owner_value_fails_closed() -> None:
    owner = _risk_result()
    metrics = owner.metrics.copy(deep=True)
    selector = (
        metrics["scope"].eq("fold")
        & metrics["fold_id"].eq(0)
        & metrics["metric"].eq("roc_auc")
        & metrics["statistic"].eq("direct")
    )
    metrics.loc[selector, "value"] = np.inf
    ref = GovernanceEvidenceRef(
        "task15",
        0,
        "metrics",
        "explanation",
        "champion",
        None,
        metric_key="roc_auc",
        scope_key="fold",
        fold_id=0,
        statistic_key="direct",
    )
    explanation = GovernanceExplanation(
        "metric_trace",
        "champion",
        "metric_evidence",
        ref,
        None,
        None,
        0,
        "available",
        None,
    )
    policy = replace(
        _policy(_candidate("champion", 0, "champion")),
        explanations=(explanation,),
    )
    with pytest.raises(ValueError) as caught:
        evaluate_governance(policy, risk_validations=(replace(owner, metrics=metrics),))
    assert str(caught.value) == "model governance: invalid_owner_value"


def test_t19_ir_02_wrong_candidate_key_fails_before_comparison() -> None:
    policy, owners = _h_pair_case()
    challenger = policy.candidates[1]
    bad_ref = replace(challenger.evidence_refs[0], candidate_key="wrong_candidate")
    bad = replace(challenger, evidence_refs=(bad_ref,))
    with pytest.raises(ValueError) as caught:
        evaluate_governance(
            replace(policy, candidates=(policy.candidates[0], bad)),
            risk_validations=owners,
        )
    assert str(caught.value) == "model governance: invalid_source_binding"


def test_t19_ir_02_strategy_candidate_rejects_task15_criterion() -> None:
    frame = pd.DataFrame({"x": [0, 1, 2, 3]})
    condition = StrategyCondition("atomic", "ge", "column", "x", "literal", 0)
    configs = tuple(
        DecisionStrategyConfig(
            key,
            "v1",
            datetime(2025, 1, 1),
            None,
            datetime(2025, 1, 2),
            (DecisionRule("rule", "decision", 0, condition, "select"),),
            "select",
            "review",
            (("select", "selected"), ("review", "review")),
        )
        for key in ("strategy_a", "strategy_b")
    )
    owners = tuple(simulate_decision_strategy(frame, config) for config in configs)
    candidates = tuple(
        GovernanceCandidate(
            key,
            "strategy",
            "task17",
            position,
            owner.strategy_key,
            owner.strategy_fingerprint,
            "v1",
            "champion" if position == 0 else "challenger",
            "approved" if position == 0 else "candidate",
            (),
        )
        for position, (key, owner) in enumerate(zip(("champion", "challenger"), owners))
    )
    criterion = GovernanceCriterion(
        "auc",
        "strategy",
        "task15",
        "metrics",
        "roc_auc",
        "fold",
        None,
        None,
        "decision",
        True,
        "higher_is_better",
    )
    policy = replace(
        _policy(
            *candidates,
            pairs=(("champion", "challenger"),),
            criteria=(criterion,),
        ),
        minimum_comparable_criteria=1,
    )
    with pytest.raises(ValueError) as caught:
        evaluate_governance(policy, decision_strategies=owners)
    assert str(caught.value) == "model governance: invalid_criterion"


def test_t19_ir_02_attribution_method_source_restriction_fails_closed() -> None:
    attribution = _h_make_attribution(candidate_key="champion")
    bad_ref = replace(
        attribution.source_ref,
        source_task="task17",
        source_table="rule_summary",
        expected_source_fingerprint="a" * 64,
    )
    with pytest.raises(ValueError) as caught:
        evaluate_governance(
            _policy(_candidate("champion", 0, "champion")),
            risk_validations=(_risk_result(),),
            model_attributions=(replace(attribution, source_ref=bad_ref),),
        )
    assert str(caught.value) == "model governance: invalid_source_binding"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("explanation_key", "/Users/secret"),
        ("feature_key", "/Users/secret"),
        ("status", "bogus"),
        ("reason", "bogus"),
        ("priority", "not-an-int"),
    ),
)
def test_t19_ir_06_structured_scalar_validation_is_exact(
    field: str, value: object
) -> None:
    owner = _risk_result()
    ref = GovernanceEvidenceRef(
        "task15",
        0,
        "metrics",
        "explanation",
        "champion",
        None,
        metric_key="roc_auc",
        scope_key="fold",
        fold_id=0,
        statistic_key="direct",
    )
    base = GovernanceExplanation(
        "trace",
        "champion",
        "metric_evidence",
        ref,
        "feature",
        None,
        0,
        "available",
        None,
    )
    with pytest.raises(ValueError) as caught:
        evaluate_governance(
            replace(
                _policy(_candidate("champion", 0, "champion")),
                explanations=(replace(base, **{field: value}),),
            ),
            risk_validations=(owner,),
        )
    assert type(caught.value) is ValueError
    assert str(caught.value) == "model governance: invalid_explanation"


def test_t19_ir_06_structured_error_precedes_later_owner_schema_error() -> None:
    ref = GovernanceEvidenceRef(
        "task15",
        0,
        "metrics",
        "explanation",
        "champion",
        None,
        metric_key="roc_auc",
        scope_key="fold",
        fold_id=0,
        statistic_key="direct",
    )
    invalid = GovernanceExplanation(
        "trace",
        "champion",
        "metric_evidence",
        ref,
        None,
        None,
        "not-an-int",  # type: ignore[arg-type]
        "available",
        None,
    )
    with pytest.raises(ValueError) as caught:
        evaluate_governance(
            replace(
                _policy(_candidate("champion", 0, "champion")),
                explanations=(invalid,),
            ),
            risk_validations=(replace(_risk_result(), metrics=None),),
        )
    assert str(caught.value) == "model governance: invalid_explanation"


# ---------------------------------------------------------------------------
# Task 19 targeted implementation-fix Wave T2: IR-04 + IR-05 only.
# ---------------------------------------------------------------------------

T19_IR_TARGETED_FIX_WAVE_T2 = {
    "T19-IR-04": {
        "reproducer": (
            "test_t19_ir_04_unreferenced_task15_future_row_fails_closed",
            "test_t19_ir_04_unreferenced_task18_future_observation_fails_closed",
        ),
        "boundary": "test_t19_ir_04_owner_schema_precedes_future_time",
        "precedence": "test_t19_ir_04_future_precedes_resource_gate",
    },
    "T19-IR-05": {
        "reproducer": "test_t19_ir_05_reference_after_current_fails_closed",
        "boundary": "test_t19_ir_05_reference_equal_current_is_legal",
        "precedence": "test_t19_ir_05_chronology_precedes_resource_gate",
    },
}


def _t2_time_task15_owner(
    *, future_row: bool = False, aware: bool = False
) -> BinaryRiskValidationResult:
    owner = _risk_result()
    folds = owner.folds.copy(deep=True)
    fold_times = [
        (
            pd.Timestamp(datetime(2025, 1, day), tz="UTC")
            if aware
            else datetime(2025, 1, day)
        )
        for day in (1, 2, 3)
    ]
    if future_row:
        fold_times[1] = datetime(2026, 1, 1)
    for column in ("cutoff", "validation_start", "validation_end"):
        folds[column] = pd.Series(fold_times, dtype="object")
    folds["analysis_as_of"] = pd.Series(fold_times, dtype="object")
    config = replace(
        owner.config,
        validation_mode="time_forward",
        analysis_as_of=(
            pd.Timestamp(datetime(2025, 1, 31), tz="UTC")
            if aware
            else datetime(2025, 1, 31)
        ),
    )
    return replace(owner, validation_mode="time_forward", config=config, folds=folds)


def test_t19_ir_04_unreferenced_task15_future_row_fails_closed() -> None:
    owner = _t2_time_task15_owner(future_row=True)
    ref = GovernanceEvidenceRef(
        "task15",
        0,
        "folds",
        "explanation",
        "champion",
        None,
        field_key="fold_id",
        fold_id=0,
    )
    policy = replace(
        _policy(_candidate("champion", 0, "champion")), evidence_refs=(ref,)
    )
    assert owner.folds.loc[1, "analysis_as_of"] > policy.analysis_as_of
    assert ref.fold_id != 1
    with pytest.raises(ValueError) as caught:
        evaluate_governance(policy, risk_validations=(owner,))
    assert str(caught.value) == "model governance: future_evidence_time"


def test_t19_ir_04_unreferenced_task18_future_observation_fails_closed() -> None:
    owner = _h_task18_owner()
    observation = owner.observation_history.copy(deep=True)
    observation.loc[1, "observation_time"] = datetime(2026, 1, 1)
    owner = replace(owner, observation_history=observation)
    ref = GovernanceEvidenceRef(
        "task18",
        0,
        "observation_history",
        "explanation",
        "scenario",
        owner.monitoring_fingerprint,
        field_key="active_rule_count",
        row_position=0,
    )
    policy, _ = _h_warning_case(owner)
    policy = replace(policy, evidence_refs=(ref,))
    assert ref.row_position == 0
    assert owner.observation_history.loc[1, "observation_time"] > policy.analysis_as_of
    with pytest.raises(ValueError) as caught:
        evaluate_governance(policy, lifecycle_monitorings=(owner,))
    assert str(caught.value) == "model governance: future_evidence_time"


def test_t19_ir_04_owner_schema_precedes_future_time() -> None:
    owner = _t2_time_task15_owner(future_row=True)
    folds = owner.folds.drop(columns=["analysis_as_of"])
    owner = replace(owner, folds=folds)
    with pytest.raises(ValueError) as caught:
        evaluate_governance(
            _policy(_candidate("champion", 0, "champion")),
            risk_validations=(owner,),
        )
    assert str(caught.value) == "model governance: invalid_owner_schema"


def test_t19_ir_04_future_precedes_resource_gate() -> None:
    owner = _t2_time_task15_owner(future_row=True)
    owners = (owner,) + tuple(_risk_result(float(i) / 1000) for i in range(1, 17))
    with pytest.raises(ValueError) as caught:
        evaluate_governance(
            _policy(_candidate("champion", 0, "champion")),
            risk_validations=owners,
        )
    assert str(caught.value) == "model governance: future_evidence_time"


def _t2_evaluate_performance(
    performance: tuple[GovernancePerformanceEvidence, ...],
) -> GovernanceResult:
    policy, owners = _h_pair_case()
    return evaluate_governance(
        policy,
        risk_validations=owners,
        performance_evidence=performance,
    )


def test_t19_ir_05_valid_performance_materializes_before_stability() -> None:
    result = _t2_evaluate_performance(_a_performance(False))
    assert len(result.performance_stability) == 1
    assert result.performance_stability.loc[0, "status"] == "available"


def test_t19_ir_05_reference_equal_current_is_legal() -> None:
    performance = _a_performance(False)
    equal_as_of = datetime(2025, 1, 5)
    performance = (
        replace(performance[0], evidence_as_of=equal_as_of),
        replace(performance[1], evidence_as_of=equal_as_of),
    )
    result = _t2_evaluate_performance(performance)
    assert len(result.performance_stability) == 1
    assert result.performance_stability.loc[0, "status"] == "available"


def test_t19_ir_05_reference_after_current_fails_closed() -> None:
    performance = _a_performance(False)
    invalid = (
        replace(performance[0], evidence_as_of=datetime(2025, 1, 10)),
        replace(performance[1], evidence_as_of=datetime(2025, 1, 6)),
    )
    with pytest.raises(ValueError) as caught:
        _t2_evaluate_performance(invalid)
    assert str(caught.value) == "model governance: invalid_performance_evidence"


def test_t19_ir_05_chronology_precedes_resource_gate() -> None:
    owners = tuple(_risk_result(float(i) / 1000) for i in range(17))
    policy, _ = _h_pair_case(owners=owners[:2])
    performance = _a_performance(False)
    invalid = (
        replace(performance[0], evidence_as_of=datetime(2025, 1, 10)),
        replace(performance[1], evidence_as_of=datetime(2025, 1, 6)),
    )
    with pytest.raises(ValueError) as caught:
        evaluate_governance(
            policy,
            risk_validations=owners,
            performance_evidence=invalid,
        )
    assert str(caught.value) == "model governance: invalid_performance_evidence"


# ---------------------------------------------------------------------------
# Task 19 targeted implementation-fix Wave T3: IR-03 only.
# ---------------------------------------------------------------------------

T19_IR_TARGETED_FIX_WAVE_T3 = {
    "T19-IR-03": {
        "available": "test_t19_ir_03_available_proof_sufficient_baseline",
        "time_unverified": "test_t19_ir_03_time_unverified_blocks_directional_path",
        "snapshot_unverified": "test_t19_ir_03_snapshot_unverified_blocks_promotion",
        "alignment_unverified": "test_t19_ir_03_alignment_unverified_blocks_promotion",
        "required_gate": (
            "test_t19_ir_03_required_unverified_criterion_blocks_promotion"
        ),
        "diagnostic": "test_t19_ir_03_diagnostic_raw_delta_no_regression",
    }
}


def _t3_two_scope_task18_case() -> tuple[
    GovernancePolicy, tuple[LifecycleMonitoringResult, ...]
]:
    """Build one same-result and one cross-result Task 18 criterion pair."""
    base = _task18_owner_for_matrix()
    reference_rows = base.monitoring_summary.loc[
        base.monitoring_summary["scenario_key"].eq("reference")
    ].copy()
    same_result_table = pd.concat(
        [base.monitoring_summary, reference_rows.assign(scenario_key="challenger")],
        ignore_index=True,
    )
    same_result = replace(
        base,
        monitoring_summary=same_result_table,
        monitoring_fingerprint="a" * 64,
    )
    cross_result = _roadmap_warning_owner(base, "challenger")
    cross_result = replace(cross_result, monitoring_fingerprint="b" * 64)

    champion_hit = GovernanceEvidenceRef(
        "task18",
        0,
        "monitoring_summary",
        "comparison_criterion",
        "reference",
        "a" * 64,
        scenario_key="reference",
        scope_key="scenario",
        scope_position=0,
        metric_key="warning_hit_count",
    )
    challenger_hit = replace(
        champion_hit,
        candidate_key="challenger",
        scenario_key="challenger",
    )
    champion_rate = replace(champion_hit, metric_key="warning_observation_rate")
    challenger_rate = replace(
        champion_rate,
        source_result_position=1,
        candidate_key="challenger",
        expected_source_fingerprint="b" * 64,
        scenario_key="challenger",
    )
    champion = GovernanceCandidate(
        "reference",
        "warning_scenario",
        "task18",
        0,
        "reference",
        "a" * 64,
        "v1",
        "champion",
        "approved",
        (champion_hit, champion_rate),
    )
    challenger = GovernanceCandidate(
        "challenger",
        "warning_scenario",
        "task18",
        1,
        "challenger",
        "b" * 64,
        "v1",
        "challenger",
        "candidate",
        (challenger_hit, challenger_rate),
    )
    criteria = (
        GovernanceCriterion(
            "hit",
            "warning_scenario",
            "task18",
            "monitoring_summary",
            "warning_hit_count",
            "scenario",
            0,
            None,
            "decision",
            True,
            "lower_is_better",
        ),
        GovernanceCriterion(
            "rate",
            "warning_scenario",
            "task18",
            "monitoring_summary",
            "warning_observation_rate",
            "scenario",
            0,
            None,
            "decision",
            True,
            "lower_is_better",
        ),
    )
    policy = _policy(
        champion,
        challenger,
        pairs=(("reference", "challenger"),),
        criteria=criteria,
    )
    return policy, (same_result, cross_result)


def test_t19_ir_03_available_proof_sufficient_baseline() -> None:
    result = _comparison_output(0.4, 0.8)
    row = result.candidate_comparisons.iloc[0]
    assert row["status"] == "available"
    assert pd.isna(row["reason"])
    assert pd.notna(row["delta"])
    assert row["comparison_outcome"] == "challenger_better"
    assert bool(row["support_comparable"])
    evaluation = result.governance_evaluations.iloc[0]
    assert evaluation["status"] == "available"
    assert bool(evaluation["counts_toward_minimum"])


def test_t19_ir_03_time_unverified_blocks_directional_path() -> None:
    policy, owners = _h_pair_case(owners=(_risk_result(), _risk_result(0.001)))
    result = evaluate_governance(policy, risk_validations=owners)
    row = result.candidate_comparisons.iloc[0]
    assert row["status"] == "not_verifiable"
    assert row["reason"] == "time_unverified"
    assert pd.isna(row["delta"])
    assert pd.isna(row["direction"])
    assert row["comparison_outcome"] == "not_comparable"
    assert not bool(row["support_comparable"])
    evaluation = result.governance_evaluations.iloc[0]
    assert evaluation["status"] == "not_verifiable"
    assert evaluation["reason"] == "time_unverified"
    assert not bool(evaluation["comparable"])
    assert bool(evaluation["blocks_promotion"])
    recommendation = result.recommendations.iloc[0]
    assert recommendation["recommendation"] != "promote_challenger"
    assert recommendation["recommendation"] == "insufficient_evidence"


def test_t19_ir_03_snapshot_unverified_blocks_promotion() -> None:
    result = _roadmap_task18_result()
    row = result.candidate_comparisons.iloc[0]
    assert row["champion_time_status"] == "verified"
    assert row["challenger_time_status"] == "verified"
    assert row["champion_source_fingerprint"] == row["challenger_source_fingerprint"]
    assert row["source_snapshot_status"] == "unverified"
    assert row["status"] == "not_verifiable"
    assert row["reason"] == "snapshot_unverified"
    assert pd.isna(row["delta"])
    assert pd.isna(row["direction"])
    assert row["comparison_outcome"] == "not_comparable"
    assert result.governance_evaluations.iloc[0]["blocks_promotion"]
    assert result.recommendations.iloc[0]["recommendation"] != "promote_challenger"


def test_t19_ir_03_alignment_unverified_blocks_promotion() -> None:
    base = _task18_owner_for_matrix()
    rows = base.monitoring_summary.loc[
        base.monitoring_summary["scenario_key"].eq("reference")
    ].copy()
    owner = replace(
        base,
        monitoring_summary=pd.concat(
            [base.monitoring_summary, rows.assign(scenario_key="challenger")],
            ignore_index=True,
        ),
        monitoring_fingerprint="c" * 64,
    )
    fp = owner.monitoring_fingerprint
    reference_ref = GovernanceEvidenceRef(
        "task18",
        0,
        "monitoring_summary",
        "comparison_criterion",
        "reference",
        fp,
        scenario_key="reference",
        scope_key="scenario",
        scope_position=0,
        metric_key="warning_hit_count",
    )
    challenger_ref = replace(
        reference_ref,
        candidate_key="challenger",
        scenario_key="challenger",
    )
    policy = replace(
        _policy(
            GovernanceCandidate(
                "reference",
                "warning_scenario",
                "task18",
                0,
                "reference",
                fp,
                "v1",
                "champion",
                "approved",
                (reference_ref,),
            ),
            GovernanceCandidate(
                "challenger",
                "warning_scenario",
                "task18",
                0,
                "challenger",
                fp,
                "v1",
                "challenger",
                "candidate",
                (challenger_ref,),
            ),
            pairs=(("reference", "challenger"),),
            criteria=(
                GovernanceCriterion(
                    "hit",
                    "warning_scenario",
                    "task18",
                    "monitoring_summary",
                    "warning_hit_count",
                    "scenario",
                    0,
                    None,
                    "decision",
                    True,
                    "lower_is_better",
                ),
            ),
        ),
        entity_alignment="owner_verified",
    )
    result = evaluate_governance(policy, lifecycle_monitorings=(owner,))
    row = result.candidate_comparisons.iloc[0]
    assert row["source_snapshot_status"] == "verified"
    assert row["entity_alignment_status"] == "unverified"
    assert row["status"] == "not_verifiable"
    assert row["reason"] == "alignment_unverified"
    assert pd.isna(row["delta"])
    assert pd.isna(row["direction"])
    assert row["comparison_outcome"] == "not_comparable"
    assert result.governance_evaluations.iloc[0]["blocks_promotion"]
    assert result.recommendations.iloc[0]["recommendation"] != "promote_challenger"


def test_t19_ir_03_required_unverified_criterion_blocks_promotion() -> None:
    policy, owners = _t3_two_scope_task18_case()
    result = evaluate_governance(policy, lifecycle_monitorings=owners)
    evaluations = result.governance_evaluations.sort_values("criterion_position")
    favorable = evaluations.iloc[0]
    unverified = evaluations.iloc[1]
    assert favorable["status"] == "available"
    assert favorable["comparison_outcome"] == "tie"
    assert unverified["status"] == "not_verifiable"
    assert unverified["reason"] == "snapshot_unverified"
    assert bool(unverified["required_for_promotion"])
    assert bool(unverified["blocks_promotion"])
    recommendation = result.recommendations.iloc[0]
    assert recommendation["recommendation"] == "insufficient_evidence"
    assert recommendation["recommendation"] != "promote_challenger"


def test_t19_ir_03_diagnostic_raw_delta_no_regression() -> None:
    result = _a_rich_result()
    diagnostic = result.candidate_comparisons.loc[
        result.candidate_comparisons["criterion_role"] == "diagnostic"
    ].iloc[0]
    assert diagnostic["status"] == "not_applicable"
    assert diagnostic["reason"] == "diagnostic_only"
    assert pd.notna(diagnostic["delta"])
    assert diagnostic["comparison_outcome"] == "not_directional"
    diagnostic_eval = result.governance_evaluations.loc[
        result.governance_evaluations["criterion_role"] == "diagnostic"
    ].iloc[0]
    assert diagnostic_eval["directional_contribution"] == "not_directional"
    assert not bool(diagnostic_eval["counts_toward_minimum"])
    assert result.recommendations.iloc[0]["recommendation"] != "promote_challenger"
