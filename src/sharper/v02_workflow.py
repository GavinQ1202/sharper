"""Typed opt-in orchestration for the Sharper v0.2 owner APIs.

This module owns only Task 20 request/result carriers, validation precedence, and
the fixed handoff between the public Task 15--19 functions.  It does not expose
the v0.2 symbols from :mod:`sharper` until the final release-surface wave.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, timedelta
from typing import Literal, NoReturn

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from sharper.data_audit import (
    DataAuditConfig,
    DataAuditResult,
    DataAuditRoles,
    audit_data_quality,
)
from sharper.decision_strategy import (
    DecisionStrategyConfig,
    DecisionStrategyResult,
    simulate_decision_strategy,
)
from sharper.lifecycle_monitoring import (
    LifecycleMonitoringConfig,
    LifecycleMonitoringResult,
    monitor_lifecycle,
)
from sharper.model_governance import (
    GovernanceAttributionEvidence,
    GovernanceCandidate,
    GovernancePerformanceEvidence,
    GovernancePolicy,
    GovernancePredictionProfile,
    GovernanceResult,
    evaluate_governance,
)
from sharper.risk_validation import (
    BinaryRiskValidationConfig,
    BinaryRiskValidationResult,
    ExternalRiskPredictions,
    validate_binary_risk,
)

_PATH_ORDER = (
    "score_validation",
    "audit",
    "preloan",
    "postloan",
    "governance",
)
_OWNER_RESULT_ORDER = (
    ("task16", "audit"),
    ("task15", "score_validation"),
    ("task17", "preloan"),
    ("task18", "postloan"),
    ("task19", "governance"),
)

_TASK15_WARNING_SOURCES = frozenset(
    {
        "duplicate_index",
        "duplicate_rows",
        "missing_target_rows_excluded",
        "external_fit_not_verifiable",
        "duplicate_thresholds_removed",
        "duplicate_gain_fractions_removed",
        "large_input",
    }
)
_TASK15_LIMITATION_SOURCES = frozenset(
    {
        "random_or_group_validation_not_time_safe",
        "entity_isolation_not_checked",
        "time_validation_not_general_feature_audit",
        "external_fit_not_verifiable",
        "ranking_probability_order_may_differ",
        "probability_metrics_unavailable",
        "calibration_diagnostic_only",
        "partial_validation_maturity",
        "single_class_validation_fold",
        "observed_association_not_causal",
    }
)
_TASK16_WARNING_SOURCES = frozenset(
    {
        "large_input",
        "duplicate_scan_skipped",
        "unique_inspection_skipped",
        "category_levels_truncated",
        "missing_patterns_truncated",
        "collinearity_columns_truncated",
        "insufficient_drift_rows",
        "point_in_time_not_verifiable",
    }
)
_TASK16_LIMITATION_SOURCES = frozenset(
    {
        "in_memory_single_process",
        "structural_identifier_evidence_only",
        "association_not_causation",
        "target_proxy_false_positive_possible",
        "caller_declared_time_provenance",
        "no_automatic_leakage_repair",
        "budget_limited_evidence",
    }
)
_TASK17_WARNING_SOURCES = frozenset(
    {"strategy", "source", "mapping", "outcome", "constraint", "resource"}
)
_TASK17_LIMITATION_SOURCES = frozenset(
    {
        "simulated_actions_not_executed",
        "historical_comparison_not_causal",
        "model_expectation_not_observed",
        "outcome_support_limited",
        "custom_score_provenance_caller_declared",
    }
)
_TASK18_WARNING_SOURCES = frozenset(
    {"input", "time", "source", "scenario", "episode", "event", "lifecycle", "resource"}
)
_TASK18_LIMITATION_SOURCES = frozenset(
    {
        "offline_monitoring_not_executed",
        "historical_comparison_not_causal",
        "caller_defined_states_and_alert_levels",
        "external_score_semantics_caller_declared",
        "right_censored_event_horizons",
        "peer_baseline_is_descriptive",
        "entity_linkage_depends_on_prepared_input",
    }
)
_TASK19_LIMITATION_SOURCES = frozenset(
    {
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
    }
)
_TASK19_WARNING_PATTERNS = (
    re.compile(r"governance:explanation:\d+"),
    re.compile(r"governance:attribution:\d+:\d+"),
    re.compile(r"governance:drift:\d+:\d+:\d+"),
    re.compile(r"governance:stability:\d+:\d+:\d+"),
    re.compile(r"governance:comparison:\d+:\d+"),
    re.compile(r"governance:evaluation:\d+:\d+"),
    re.compile(r"governance:recommendation:\d+"),
    re.compile(r"governance:summary:\d+"),
    re.compile(r"governance:metadata:governance:\d+:\d+"),
    re.compile(r"governance:metadata:candidate:\d+:\d+:\d+"),
    re.compile(r"governance:provenance:\d+"),
)


@dataclass(frozen=True)
class V02ScoreValidationRequest:
    """Declare the optional Task 15 score-validation handoff."""

    target: str
    config: BinaryRiskValidationConfig
    positive_label: str | int | bool | np.generic | None = None
    estimator: ClassifierMixin | None = None
    external_predictions: ExternalRiskPredictions | None = None
    features: tuple[str, ...] | None = None
    exclude_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class V02AuditRequest:
    """Declare the optional Task 16 audit and audit-only reference frame."""

    reference: pd.DataFrame | None = None
    roles: DataAuditRoles | None = None
    config: DataAuditConfig | None = None


@dataclass(frozen=True)
class V02PreLoanRequest:
    """Declare the optional Task 17 pre-loan strategy simulation."""

    config: DecisionStrategyConfig


@dataclass(frozen=True)
class V02PostLoanRequest:
    """Declare the optional Task 18 post-loan monitoring run."""

    config: LifecycleMonitoringConfig


@dataclass(frozen=True)
class V02GovernanceRequest:
    """Declare the optional final Task 19 governance evaluation."""

    policy: GovernancePolicy
    model_attributions: tuple[GovernanceAttributionEvidence, ...] = ()
    prediction_profiles: tuple[GovernancePredictionProfile, ...] = ()
    performance_evidence: tuple[GovernancePerformanceEvidence, ...] = ()


@dataclass(frozen=True)
class V02WorkflowRequest:
    """Carry the primary frame and explicitly enabled v0.2 paths."""

    data: pd.DataFrame
    score_validation: V02ScoreValidationRequest | None = None
    audit: V02AuditRequest | None = None
    preloan: V02PreLoanRequest | None = None
    postloan: V02PostLoanRequest | None = None
    governance: V02GovernanceRequest | None = None


@dataclass(frozen=True)
class V02WorkflowResult:
    """Contain the complete Task 20 orchestration result."""

    contract_version: Literal["task20-integration-v1"]
    enabled_paths: tuple[str, ...]
    path_status: pd.DataFrame
    call_trace: tuple[str, ...]
    score_validation: BinaryRiskValidationResult | None
    data_audit: DataAuditResult | None
    preloan: DecisionStrategyResult | None
    postloan: LifecycleMonitoringResult | None
    governance: GovernanceResult | None
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]


def _task20_error(key: str) -> NoReturn:
    raise ValueError(f"sharper task20: {key}")


def _contains_open_carrier(value: object, seen: set[int] | None = None) -> bool:
    """Find callables and file-like handles without inspecting raw frame cells."""
    if value is None or isinstance(
        value, (str, bytes, int, float, bool, date, timedelta)
    ):
        return False
    if isinstance(value, np.generic) or isinstance(value, pd.DataFrame):
        return False
    if callable(value) or hasattr(value, "read") or hasattr(value, "write"):
        return True
    if seen is None:
        seen = set()
    identifier = id(value)
    if identifier in seen:
        return False
    seen.add(identifier)
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_open_carrier(item, seen) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_open_carrier(item, seen)
            for pair in value.items()
            for item in pair
        )
    if is_dataclass(value):
        for field in fields(value):
            if _contains_open_carrier(getattr(value, field.name), seen):
                return True
    return False


def _validate_raw_carriers(request: V02WorkflowRequest) -> None:
    if type(request.data) is not pd.DataFrame:
        _task20_error("request_raw_carrier")
    for name, expected in (
        ("score_validation", V02ScoreValidationRequest),
        ("audit", V02AuditRequest),
        ("preloan", V02PreLoanRequest),
        ("postloan", V02PostLoanRequest),
        ("governance", V02GovernanceRequest),
    ):
        value = getattr(request, name)
        if value is not None and type(value) is not expected:
            _task20_error("request_raw_carrier")
    if request.audit is not None:
        reference = request.audit.reference
        if reference is not None and type(reference) is not pd.DataFrame:
            _task20_error("request_raw_carrier")
    if _contains_open_carrier(request):
        _task20_error("request_raw_carrier")


def _validate_path_conflicts(request: V02WorkflowRequest) -> None:
    expected = {field.name for field in fields(V02WorkflowRequest)}
    if set(vars(request)) - expected:
        _task20_error("request_path_input_conflict")


def _normalize_positive_label(
    value: str | int | bool | np.generic | None,
) -> str | int | bool | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) in (str, bool, int):
        return value
    _task20_error("owner_call_contract")


def _validate_score_request(
    request: V02ScoreValidationRequest,
) -> str | int | bool | None:
    if (
        type(request.target) is not str
        or type(request.config) is not BinaryRiskValidationConfig
    ):
        _task20_error("owner_call_contract")
    if request.features is not None and (
        type(request.features) is not tuple
        or any(type(column) is not str for column in request.features)
    ):
        _task20_error("owner_call_contract")
    if type(request.exclude_columns) is not tuple or any(
        type(column) is not str for column in request.exclude_columns
    ):
        _task20_error("owner_call_contract")
    estimator = request.estimator
    external = request.external_predictions
    if (estimator is None) == (external is None):
        _task20_error("owner_call_contract")
    if estimator is not None and not isinstance(estimator, ClassifierMixin):
        _task20_error("owner_call_contract")
    if external is not None and type(external) is not ExternalRiskPredictions:
        _task20_error("owner_call_contract")
    return _normalize_positive_label(request.positive_label)


def _validate_request_carriers(request: V02WorkflowRequest) -> None:
    if request.audit is not None and (
        request.audit.roles is not None
        and type(request.audit.roles) is not DataAuditRoles
    ):
        _task20_error("owner_call_contract")
    if request.audit is not None and (
        request.audit.config is not None
        and type(request.audit.config) is not DataAuditConfig
    ):
        _task20_error("owner_call_contract")
    if (
        request.preloan is not None
        and type(request.preloan.config) is not DecisionStrategyConfig
    ):
        _task20_error("owner_call_contract")
    if (
        request.postloan is not None
        and type(request.postloan.config) is not LifecycleMonitoringConfig
    ):
        _task20_error("owner_call_contract")
    if request.governance is not None:
        if type(request.governance.policy) is not GovernancePolicy:
            _task20_error("owner_call_contract")
        for value, expected in (
            (request.governance.model_attributions, GovernanceAttributionEvidence),
            (request.governance.prediction_profiles, GovernancePredictionProfile),
            (request.governance.performance_evidence, GovernancePerformanceEvidence),
        ):
            if type(value) is not tuple or any(
                type(item) is not expected for item in value
            ):
                _task20_error("owner_call_contract")


def _governance_dependency_missing(request: V02WorkflowRequest) -> bool:
    governance = request.governance
    if governance is None or type(governance.policy) is not GovernancePolicy:
        return False
    candidates = governance.policy.candidates
    if type(candidates) is not tuple or any(
        type(candidate) is not GovernanceCandidate for candidate in candidates
    ):
        return False
    enabled = {
        "model": request.score_validation is not None,
        "strategy": request.preloan is not None,
        "warning_scenario": request.postloan is not None,
    }
    return any(
        candidate.candidate_family in enabled
        and not enabled[candidate.candidate_family]
        for candidate in candidates
    )


def _validate_owner_result(
    result: object,
    expected: type[object],
    input_frames: tuple[pd.DataFrame, ...],
) -> None:
    if type(result) is not expected:
        _task20_error("result_contract")
    for field in fields(expected):
        if not hasattr(result, field.name):
            _task20_error("result_contract")
    warnings = getattr(result, "warnings")
    limitations = getattr(result, "limitations")
    if type(warnings) is not tuple or type(limitations) is not tuple:
        _task20_error("result_contract")
    if any(type(value) is not str for value in (*warnings, *limitations)):
        _task20_error("result_contract")
    for frame in input_frames:
        if _graph_contains_identity(result, frame):
            _task20_error("result_contract")
    if _contains_open_carrier(result):
        _task20_error("result_contract")


def _graph_contains_identity(
    value: object, target: object, seen: set[int] | None = None
) -> bool:
    if value is target:
        return True
    if isinstance(value, pd.DataFrame):
        return False
    if seen is None:
        seen = set()
    identifier = id(value)
    if identifier in seen:
        return False
    seen.add(identifier)
    if is_dataclass(value):
        return any(
            _graph_contains_identity(getattr(value, field.name), target, seen)
            for field in fields(value)
            if hasattr(value, field.name)
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_graph_contains_identity(item, target, seen) for item in value)
    if isinstance(value, dict):
        return any(
            _graph_contains_identity(item, target, seen)
            for pair in value.items()
            for item in pair
        )
    return False


def _path_status(request: V02WorkflowRequest) -> pd.DataFrame:
    enabled = [getattr(request, path) is not None for path in _PATH_ORDER]
    return pd.DataFrame(
        {
            "path_key": pd.Series(_PATH_ORDER, dtype="string"),
            "enabled": pd.Series(enabled, dtype="boolean"),
            "status": pd.Series(
                ["completed" if value else "not_requested" for value in enabled],
                dtype="string",
            ),
            "reason": pd.Series(
                ["completed" if value else "path_not_requested" for value in enabled],
                dtype="string",
            ),
        }
    )


def _task19_warning_source(value: str) -> bool:
    return any(pattern.fullmatch(value) for pattern in _TASK19_WARNING_PATTERNS)


def _valid_source(owner: str, value: str, limitation: bool) -> bool:
    if owner == "task15":
        return value in (
            _TASK15_LIMITATION_SOURCES if limitation else _TASK15_WARNING_SOURCES
        )
    if owner == "task16":
        return value in (
            _TASK16_LIMITATION_SOURCES if limitation else _TASK16_WARNING_SOURCES
        )
    if owner == "task17":
        return value in (
            _TASK17_LIMITATION_SOURCES if limitation else _TASK17_WARNING_SOURCES
        )
    if owner == "task18":
        return value in (
            _TASK18_LIMITATION_SOURCES if limitation else _TASK18_WARNING_SOURCES
        )
    if owner == "task19":
        return (
            value in _TASK19_LIMITATION_SOURCES
            if limitation
            else _task19_warning_source(value)
        )
    return False


def _collect_tokens(
    results: dict[str, object | None],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    warnings: list[str] = []
    limitations: list[str] = []
    warning_seen: set[str] = set()
    limitation_seen: set[str] = set()
    for owner, path in _OWNER_RESULT_ORDER:
        result = results[path]
        if result is None:
            continue
        for source in getattr(result, "warnings"):
            if _valid_source(owner, source, False):
                token = f"task20.warning.{owner}.{source}"
                if token not in warning_seen:
                    warning_seen.add(token)
                    warnings.append(token)
        for source in getattr(result, "limitations"):
            if _valid_source(owner, source, True):
                token = f"task20.limitation.{owner}.{source}"
                if token not in limitation_seen:
                    limitation_seen.add(token)
                    limitations.append(token)
    return tuple(warnings), tuple(limitations)


def run_v02_workflow(request: V02WorkflowRequest) -> V02WorkflowResult:
    """Run the enabled Task 15--19 public owner paths exactly once.

    Parameters
    ----------
    request
        Frozen Task 20 carrier containing the primary DataFrame and explicit
        optional path declarations. The DataFrame and nested carriers are read
        only; the audit reference is the sole secondary frame.

    Returns
    -------
    V02WorkflowResult
        A complete typed result with fixed path rows, call trace, owner results,
        and closed warning/limitation tokens.

    Raises
    ------
    ValueError
        With the frozen ``sharper task20: <error_key>`` prefix for integration
        validation failures. Owner errors retain their original type, message,
        and cause.

    Examples
    --------
    >>> # result = run_v02_workflow(
    ... #     V02WorkflowRequest(data=frame, audit=V02AuditRequest())
    ... # )
    """
    if type(request) is not V02WorkflowRequest:
        _task20_error("invalid_request_type")

    _validate_raw_carriers(request)
    _validate_path_conflicts(request)
    if not any(getattr(request, path) is not None for path in _PATH_ORDER):
        _task20_error("request_requires_primary_path")

    normalized_positive_label: str | int | bool | None = None
    if request.score_validation is not None:
        normalized_positive_label = _validate_score_request(request.score_validation)

    if _governance_dependency_missing(request):
        _task20_error("governance_dependency_missing")

    _validate_request_carriers(request)

    audit_result: DataAuditResult | None = None
    score_result: BinaryRiskValidationResult | None = None
    preloan_result: DecisionStrategyResult | None = None
    postloan_result: LifecycleMonitoringResult | None = None
    governance_result: GovernanceResult | None = None
    call_trace: list[str] = []

    if request.audit is not None:
        call_trace.append("audit_data_quality")
        audit_result = audit_data_quality(
            request.data,
            reference=request.audit.reference,
            roles=request.audit.roles,
            config=request.audit.config,
        )

    if request.score_validation is not None:
        call_trace.append("validate_binary_risk")
        score_result = validate_binary_risk(
            request.data,
            request.score_validation.target,
            positive_label=normalized_positive_label,
            config=request.score_validation.config,
            estimator=request.score_validation.estimator,
            external_predictions=request.score_validation.external_predictions,
            features=request.score_validation.features,
            exclude_columns=request.score_validation.exclude_columns,
        )

    if request.preloan is not None:
        call_trace.append("simulate_decision_strategy")
        preloan_result = simulate_decision_strategy(
            request.data,
            request.preloan.config,
            risk_validation=score_result,
            data_audit=audit_result,
        )

    if request.postloan is not None:
        call_trace.append("monitor_lifecycle")
        postloan_result = monitor_lifecycle(
            request.data,
            request.postloan.config,
            risk_validation=score_result,
            data_audit=audit_result,
        )

    if request.governance is not None:
        call_trace.append("evaluate_governance")
        governance_result = evaluate_governance(
            request.governance.policy,
            risk_validations=() if score_result is None else (score_result,),
            data_audits=() if audit_result is None else (audit_result,),
            decision_strategies=() if preloan_result is None else (preloan_result,),
            lifecycle_monitorings=() if postloan_result is None else (postloan_result,),
            model_attributions=request.governance.model_attributions,
            prediction_profiles=request.governance.prediction_profiles,
            performance_evidence=request.governance.performance_evidence,
        )

    expected_results = (
        (
            request.score_validation is not None,
            score_result,
            BinaryRiskValidationResult,
        ),
        (request.audit is not None, audit_result, DataAuditResult),
        (request.preloan is not None, preloan_result, DecisionStrategyResult),
        (request.postloan is not None, postloan_result, LifecycleMonitoringResult),
        (request.governance is not None, governance_result, GovernanceResult),
    )
    input_frames = (request.data,)
    if request.audit is not None and request.audit.reference is not None:
        input_frames += (request.audit.reference,)
    for enabled, result, expected in expected_results:
        if enabled:
            _validate_owner_result(result, expected, input_frames)
        elif result is not None:
            _task20_error("result_contract")

    owner_results: dict[str, object | None] = {
        "score_validation": score_result,
        "audit": audit_result,
        "preloan": preloan_result,
        "postloan": postloan_result,
        "governance": governance_result,
    }
    warnings, limitations = _collect_tokens(owner_results)
    result = V02WorkflowResult(
        contract_version="task20-integration-v1",
        enabled_paths=tuple(
            path for path in _PATH_ORDER if getattr(request, path) is not None
        ),
        path_status=_path_status(request),
        call_trace=tuple(call_trace),
        score_validation=score_result,
        data_audit=audit_result,
        preloan=preloan_result,
        postloan=postloan_result,
        governance=governance_result,
        warnings=warnings,
        limitations=limitations,
    )
    if _graph_contains_identity(result, request.data):
        _task20_error("result_contract")
    if (
        request.audit is not None
        and request.audit.reference is not None
        and _graph_contains_identity(result, request.audit.reference)
    ):
        _task20_error("result_contract")
    return result
