"""Task 20 Wave I1 typed workflow contract tests."""

from __future__ import annotations

import inspect
from dataclasses import MISSING, fields

import numpy as np
import pandas as pd
import pytest

import sharper.v02_workflow as workflow
from sharper.data_audit import DataAuditResult
from sharper.decision_strategy import DecisionStrategyConfig, DecisionStrategyResult
from sharper.lifecycle_monitoring import (
    LifecycleMonitoringConfig,
    LifecycleMonitoringResult,
)
from sharper.model_governance import (
    GovernanceCandidate,
    GovernancePolicy,
    GovernanceResult,
)
from sharper.risk_validation import (
    BinaryRiskValidationConfig,
    BinaryRiskValidationResult,
    ExternalRiskPredictions,
)


def _empty_instance(cls):
    instance = object.__new__(cls)
    for field in fields(cls):
        object.__setattr__(instance, field.name, None)
    return instance


def _owner_result(cls, warnings=(), limitations=()):
    result = _empty_instance(cls)
    object.__setattr__(result, "warnings", tuple(warnings))
    object.__setattr__(result, "limitations", tuple(limitations))
    return result


def _score_request(positive_label=None) -> workflow.V02ScoreValidationRequest:
    external = ExternalRiskPredictions(
        row_positions=(0,),
        fold_ids=(0,),
        fold_fit_row_positions=((0, (0,)),),
        ranking_scores=(0.5,),
        ranking_direction="higher_risk",
        event_probabilities=None,
        probability_positive_label=None,
        probability_provenance=None,
    )
    return workflow.V02ScoreValidationRequest(
        target="target",
        config=BinaryRiskValidationConfig(validation_mode="stratified_holdout"),
        positive_label=positive_label,
        external_predictions=external,
    )


def _governance_request(candidate=None) -> workflow.V02GovernanceRequest:
    policy = _empty_instance(GovernancePolicy)
    object.__setattr__(policy, "candidates", () if candidate is None else (candidate,))
    return workflow.V02GovernanceRequest(policy=policy)


def _install_owner_spies(
    monkeypatch: pytest.MonkeyPatch, calls: list[str], results=None
):
    results = results or {}

    def audit(data, *, reference=None, roles=None, config=None):
        calls.append("audit_data_quality")
        return results.get("audit", _owner_result(DataAuditResult))

    def score(
        data,
        target,
        *,
        positive_label=None,
        config=None,
        estimator=None,
        external_predictions=None,
        features=None,
        exclude_columns=(),
    ):
        calls.append("validate_binary_risk")
        return results.get("score", _owner_result(BinaryRiskValidationResult))

    def preloan(data, config, *, risk_validation=None, data_audit=None):
        calls.append("simulate_decision_strategy")
        return results.get("preloan", _owner_result(DecisionStrategyResult))

    def postloan(data, config, *, risk_validation=None, data_audit=None):
        calls.append("monitor_lifecycle")
        return results.get("postloan", _owner_result(LifecycleMonitoringResult))

    def governance(
        policy,
        *,
        risk_validations=(),
        data_audits=(),
        decision_strategies=(),
        lifecycle_monitorings=(),
        model_attributions=(),
        prediction_profiles=(),
        performance_evidence=(),
    ):
        calls.append("evaluate_governance")
        return results.get("governance", _owner_result(GovernanceResult))

    monkeypatch.setattr(workflow, "audit_data_quality", audit)
    monkeypatch.setattr(workflow, "validate_binary_risk", score)
    monkeypatch.setattr(workflow, "simulate_decision_strategy", preloan)
    monkeypatch.setattr(workflow, "monitor_lifecycle", postloan)
    monkeypatch.setattr(workflow, "evaluate_governance", governance)


def test_v02_public_symbols_signatures_and_fields() -> None:
    expected = {
        workflow.V02ScoreValidationRequest: (
            "target",
            "config",
            "positive_label",
            "estimator",
            "external_predictions",
            "features",
            "exclude_columns",
        ),
        workflow.V02AuditRequest: ("reference", "roles", "config"),
        workflow.V02PreLoanRequest: ("config",),
        workflow.V02PostLoanRequest: ("config",),
        workflow.V02GovernanceRequest: (
            "policy",
            "model_attributions",
            "prediction_profiles",
            "performance_evidence",
        ),
        workflow.V02WorkflowRequest: (
            "data",
            "score_validation",
            "audit",
            "preloan",
            "postloan",
            "governance",
        ),
        workflow.V02WorkflowResult: (
            "contract_version",
            "enabled_paths",
            "path_status",
            "call_trace",
            "score_validation",
            "data_audit",
            "preloan",
            "postloan",
            "governance",
            "warnings",
            "limitations",
        ),
    }
    for cls, names in expected.items():
        assert cls.__dataclass_params__.frozen is True
        assert tuple(field.name for field in fields(cls)) == names
        assert cls.__doc__

    score_fields = fields(workflow.V02ScoreValidationRequest)
    assert score_fields[0].default is MISSING
    assert score_fields[1].default is MISSING
    assert [field.default for field in score_fields[2:]] == [None, None, None, None, ()]
    assert str(inspect.signature(workflow.run_v02_workflow)) == (
        "(request: 'V02WorkflowRequest') -> 'V02WorkflowResult'"
    )


@pytest.mark.parametrize(
    ("carrier", "call_name"),
    [
        ("audit", "audit_data_quality"),
        ("preloan", "simulate_decision_strategy"),
        ("postloan", "monitor_lifecycle"),
    ],
)
def test_v02_paths_enable_disable_independently(
    monkeypatch: pytest.MonkeyPatch, carrier: str, call_name: str
) -> None:
    calls: list[str] = []
    _install_owner_spies(monkeypatch, calls)
    values = {
        "audit": workflow.V02AuditRequest(),
        "preloan": workflow.V02PreLoanRequest(_empty_instance(DecisionStrategyConfig)),
        "postloan": workflow.V02PostLoanRequest(
            _empty_instance(LifecycleMonitoringConfig)
        ),
    }
    result = workflow.run_v02_workflow(
        workflow.V02WorkflowRequest(
            data=pd.DataFrame({"x": [1]}), **{carrier: values[carrier]}
        )
    )
    assert result.enabled_paths == (carrier,)
    assert result.call_trace == (call_name,)
    assert calls == [call_name]
    assert result.path_status.loc[0, "status"] == "not_requested"
    assert (
        result.path_status.loc[workflow._PATH_ORDER.index(carrier), "status"]
        == "completed"
    )


def test_v02_enabled_owner_calls_once_in_frozen_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _install_owner_spies(monkeypatch, calls)
    request = workflow.V02WorkflowRequest(
        data=pd.DataFrame({"x": [1]}),
        score_validation=_score_request(),
        audit=workflow.V02AuditRequest(),
        preloan=workflow.V02PreLoanRequest(_empty_instance(DecisionStrategyConfig)),
        postloan=workflow.V02PostLoanRequest(
            _empty_instance(LifecycleMonitoringConfig)
        ),
        governance=_governance_request(),
    )
    result = workflow.run_v02_workflow(request)
    expected = [
        "audit_data_quality",
        "validate_binary_risk",
        "simulate_decision_strategy",
        "monitor_lifecycle",
        "evaluate_governance",
    ]
    assert calls == expected
    assert result.call_trace == tuple(expected)
    assert result.enabled_paths == tuple(workflow._PATH_ORDER)


def test_v02_owner_result_carriers_and_governance_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = GovernanceCandidate(
        candidate_key="model",
        candidate_family="model",
        source_task="task15",
        source_result_position=0,
        source_candidate_key=None,
        expected_source_fingerprint=None,
        version=None,
        declared_role="champion",
        declared_state="approved",
        evidence_refs=(),
    )
    calls: list[str] = []
    score_result = _owner_result(BinaryRiskValidationResult)
    received: dict[str, object] = {}

    def score(*args, **kwargs):
        calls.append("validate_binary_risk")
        return score_result

    def governance(policy, **kwargs):
        calls.append("evaluate_governance")
        received.update(kwargs)
        return _owner_result(GovernanceResult)

    monkeypatch.setattr(workflow, "validate_binary_risk", score)
    monkeypatch.setattr(workflow, "evaluate_governance", governance)
    result = workflow.run_v02_workflow(
        workflow.V02WorkflowRequest(
            data=pd.DataFrame({"x": [1]}),
            score_validation=_score_request(),
            governance=_governance_request(candidate),
        )
    )
    assert result.score_validation is score_result
    assert received["risk_validations"] == (score_result,)
    assert received["data_audits"] == ()
    assert calls == ["validate_binary_risk", "evaluate_governance"]


def test_v02_result_has_no_raw_frame_or_runtime_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"x": [1]})
    bad_result = _owner_result(DataAuditResult)
    object.__setattr__(bad_result, "dataset_profile", frame)
    monkeypatch.setattr(
        workflow, "audit_data_quality", lambda *args, **kwargs: bad_result
    )
    with pytest.raises(ValueError, match=r"^sharper task20: result_contract$"):
        workflow.run_v02_workflow(
            workflow.V02WorkflowRequest(data=frame, audit=workflow.V02AuditRequest())
        )


def test_v02_repeatability_and_ordering(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_owner_spies(monkeypatch, [])
    request = workflow.V02WorkflowRequest(
        data=pd.DataFrame({"x": [1]}),
        audit=workflow.V02AuditRequest(),
        preloan=workflow.V02PreLoanRequest(_empty_instance(DecisionStrategyConfig)),
    )
    first = workflow.run_v02_workflow(request)
    second = workflow.run_v02_workflow(request)
    assert first.enabled_paths == second.enabled_paths
    assert first.call_trace == second.call_trace
    pd.testing.assert_frame_equal(first.path_status, second.path_status)
    assert first.warnings == second.warnings == ()
    assert first.limitations == second.limitations == ()


def test_v02_privacy_and_input_immutability(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({"x": [1, 2]})
    reference = frame
    before = frame.copy(deep=True)
    calls: list[tuple[object, object]] = []

    def audit(data, *, reference=None, roles=None, config=None):
        calls.append((data, reference))
        return _owner_result(DataAuditResult)

    monkeypatch.setattr(workflow, "audit_data_quality", audit)
    result = workflow.run_v02_workflow(
        workflow.V02WorkflowRequest(
            data=frame,
            audit=workflow.V02AuditRequest(reference=reference),
        )
    )
    pd.testing.assert_frame_equal(frame, before)
    assert calls == [(frame, reference)]
    assert result.data_audit is not None
    assert result.data_audit is not frame


@pytest.mark.parametrize(
    ("case", "error"),
    [
        (object(), "invalid_request_type"),
        (workflow.V02WorkflowRequest(data=object()), "request_raw_carrier"),
        (
            workflow.V02WorkflowRequest(data=pd.DataFrame()),
            "request_requires_primary_path",
        ),
    ],
)
def test_v02_workflow_basic_error_precedence(case, error: str) -> None:
    with pytest.raises(ValueError, match=rf"^sharper task20: {error}$"):
        workflow.run_v02_workflow(case)


def test_v02_request_path_input_conflict() -> None:
    request = workflow.V02WorkflowRequest(data=pd.DataFrame({"x": [1]}))
    object.__setattr__(request, "reference", pd.DataFrame({"x": [1]}))
    with pytest.raises(
        ValueError, match=r"^sharper task20: request_path_input_conflict$"
    ):
        workflow.run_v02_workflow(request)


def test_v02_owner_call_contract_and_multi_invalid_first_error() -> None:
    request = workflow.V02WorkflowRequest(
        data=pd.DataFrame({"x": [1]}),
        score_validation=_score_request(positive_label=object()),
    )
    with pytest.raises(ValueError, match=r"^sharper task20: owner_call_contract$"):
        workflow.run_v02_workflow(request)


def test_v02_enabled_failure_stops_later_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def audit(*args, **kwargs):
        calls.append("audit_data_quality")
        raise ValueError("data audit input is invalid: sentinel")

    def preloan(*args, **kwargs):
        calls.append("simulate_decision_strategy")
        return _owner_result(DecisionStrategyResult)

    monkeypatch.setattr(workflow, "audit_data_quality", audit)
    monkeypatch.setattr(workflow, "simulate_decision_strategy", preloan)
    with pytest.raises(ValueError, match="data audit input is invalid"):
        workflow.run_v02_workflow(
            workflow.V02WorkflowRequest(
                data=pd.DataFrame({"x": [1]}),
                audit=workflow.V02AuditRequest(),
                preloan=workflow.V02PreLoanRequest(
                    _empty_instance(DecisionStrategyConfig)
                ),
            )
        )
    assert calls == ["audit_data_quality"]


@pytest.mark.parametrize(
    "label", ["yes", True, 1, np.str_("yes"), np.bool_(True), np.int64(1)]
)
def test_v02_positive_label_closed_domain(
    monkeypatch: pytest.MonkeyPatch, label
) -> None:
    captured: list[object] = []

    def score(*args, **kwargs):
        captured.append(kwargs["positive_label"])
        return _owner_result(BinaryRiskValidationResult)

    monkeypatch.setattr(workflow, "validate_binary_risk", score)
    workflow.run_v02_workflow(
        workflow.V02WorkflowRequest(
            data=pd.DataFrame({"x": [1]}),
            score_validation=_score_request(label),
        )
    )
    assert type(captured[0]) in (str, bool, int)
    assert captured[0] == (label.item() if isinstance(label, np.generic) else label)


def test_v02_positive_label_subclass_and_arbitrary_fallback_rejected() -> None:
    class Label(str):
        pass

    for label in (Label("yes"), 1.0, ("yes",)):
        with pytest.raises(ValueError, match=r"^sharper task20: owner_call_contract$"):
            workflow.run_v02_workflow(
                workflow.V02WorkflowRequest(
                    data=pd.DataFrame({"x": [1]}),
                    score_validation=_score_request(label),
                )
            )


def test_v02_warning_limitation_source_merge_and_dedupe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = {
        "audit": _owner_result(
            DataAuditResult,
            warnings=("large_input", "large_input"),
            limitations=("budget_limited_evidence",),
        ),
        "score": _owner_result(
            BinaryRiskValidationResult,
            warnings=("duplicate_index",),
            limitations=("probability_metrics_unavailable",),
        ),
        "preloan": _owner_result(
            DecisionStrategyResult,
            warnings=("not-forwarded",),
            limitations=("simulated_actions_not_executed",),
        ),
        "governance": _owner_result(
            GovernanceResult,
            warnings=("governance:summary:0",),
            limitations=("diagnostic_only",),
        ),
    }
    _install_owner_spies(monkeypatch, [], results)
    result = workflow.run_v02_workflow(
        workflow.V02WorkflowRequest(
            data=pd.DataFrame({"x": [1]}),
            score_validation=_score_request(),
            audit=workflow.V02AuditRequest(),
            preloan=workflow.V02PreLoanRequest(_empty_instance(DecisionStrategyConfig)),
            governance=_governance_request(),
        )
    )
    assert result.warnings == (
        "task20.warning.task16.large_input",
        "task20.warning.task15.duplicate_index",
        "task20.warning.task19.governance:summary:0",
    )
    assert result.limitations == (
        "task20.limitation.task16.budget_limited_evidence",
        "task20.limitation.task15.probability_metrics_unavailable",
        "task20.limitation.task17.simulated_actions_not_executed",
        "task20.limitation.task19.diagnostic_only",
    )


def test_v02_governance_dependency_missing_precedes_owner_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = GovernanceCandidate(
        candidate_key="model",
        candidate_family="model",
        source_task="task15",
        source_result_position=0,
        source_candidate_key=None,
        expected_source_fingerprint=None,
        version=None,
        declared_role="champion",
        declared_state="approved",
        evidence_refs=(),
    )
    calls: list[str] = []
    _install_owner_spies(monkeypatch, calls)
    with pytest.raises(
        ValueError, match=r"^sharper task20: governance_dependency_missing$"
    ):
        workflow.run_v02_workflow(
            workflow.V02WorkflowRequest(
                data=pd.DataFrame({"x": [1]}),
                governance=_governance_request(candidate),
            )
        )
    assert calls == []


def test_v02_path_status_schema_and_status_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_owner_spies(monkeypatch, [])
    result = workflow.run_v02_workflow(
        workflow.V02WorkflowRequest(
            data=pd.DataFrame({"x": [1]}), audit=workflow.V02AuditRequest()
        )
    )
    assert tuple(result.path_status.columns) == (
        "path_key",
        "enabled",
        "status",
        "reason",
    )
    assert [str(dtype) for dtype in result.path_status.dtypes] == [
        "string",
        "boolean",
        "string",
        "string",
    ]
    assert tuple(result.path_status["path_key"]) == workflow._PATH_ORDER
    assert set(result.path_status["status"]) == {"completed", "not_requested"}
    assert set(result.path_status["reason"]) == {"completed", "path_not_requested"}
