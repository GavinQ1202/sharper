"""Offline explainability and champion/challenger governance evidence.

This module is deliberately a consumer of the frozen Task 15--18 result
contracts.  It never executes an estimator, mutates an owner result, performs
an external action, or promotes a candidate.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd

from sharper.data_audit import (
    _TABLE_SCHEMAS as _TASK16_TABLE_SCHEMAS,
)
from sharper.data_audit import (
    DataAuditResult,
)
from sharper.decision_strategy import (
    _ACTION_SUMMARY_COLUMNS as _TASK17_ACTION_SUMMARY_COLUMNS,
)
from sharper.decision_strategy import (
    _BUSINESS_SUMMARY_COLUMNS as _TASK17_BUSINESS_SUMMARY_COLUMNS,
)
from sharper.decision_strategy import (
    _CONSTRAINT_COLUMNS as _TASK17_CONSTRAINT_COLUMNS,
)
from sharper.decision_strategy import (
    _PROVENANCE_COLUMNS as _TASK17_PROVENANCE_COLUMNS,
)
from sharper.decision_strategy import (
    _ROW_COLUMNS as _TASK17_ROW_COLUMNS,
)
from sharper.decision_strategy import (
    _RULE_EVALUATION_COLUMNS as _TASK17_RULE_EVALUATION_COLUMNS,
)
from sharper.decision_strategy import (
    _RULE_SUMMARY_COLUMNS as _TASK17_RULE_SUMMARY_COLUMNS,
)
from sharper.decision_strategy import (
    _TRANSITION_COLUMNS as _TASK17_TRANSITION_COLUMNS,
)
from sharper.decision_strategy import (
    DecisionStrategyResult,
)
from sharper.evaluation import (
    _RISK_CALIBRATION_COLUMNS as _TASK15_CALIBRATION_COLUMNS,
)
from sharper.evaluation import (
    _RISK_GAINS_COLUMNS as _TASK15_GAINS_COLUMNS,
)
from sharper.evaluation import (
    _RISK_METRIC_COLUMNS as _TASK15_METRIC_COLUMNS,
)
from sharper.evaluation import (
    _RISK_THRESHOLD_COLUMNS as _TASK15_THRESHOLD_COLUMNS,
)
from sharper.lifecycle_monitoring import (
    _TABLE_SCHEMAS as _TASK18_TABLE_SCHEMAS,
)
from sharper.lifecycle_monitoring import (
    LifecycleMonitoringResult,
)
from sharper.risk_validation import (
    _BUSINESS_COLUMNS as _TASK15_BUSINESS_COLUMNS,
)
from sharper.risk_validation import (
    _EXCLUDED_COLUMNS as _TASK15_EXCLUDED_COLUMNS,
)
from sharper.risk_validation import (
    _FOLD_COLUMNS as _TASK15_FOLD_COLUMNS,
)
from sharper.risk_validation import (
    _OPERATING_COLUMNS as _TASK15_OPERATING_COLUMNS,
)
from sharper.risk_validation import (
    _PREDICTION_COLUMNS as _TASK15_PREDICTION_COLUMNS,
)
from sharper.risk_validation import (
    BinaryRiskValidationResult,
)


@dataclass(frozen=True)
class GovernanceEvidenceRef:
    source_task: Literal["task15", "task16", "task17", "task18"]
    source_result_position: int
    source_table: str
    source_use: Literal[
        "comparison_criterion",
        "diagnostic",
        "explanation",
        "attribution_context",
        "drift_context",
        "stability_context",
    ]
    candidate_key: str | None
    expected_source_fingerprint: str | None
    field_key: str | None = None
    metric_key: str | None = None
    side_key: str | None = None
    column_key: str | None = None
    scope_key: str | None = None
    scope_column: str | None = None
    scope_position: int | None = None
    time_position: int | None = None
    fold_id: int | None = None
    row_position: int | None = None
    entity_position: int | None = None
    from_row_position: int | None = None
    to_row_position: int | None = None
    scenario_key: str | None = None
    reference_scenario_key: str | None = None
    comparator_scenario_key: str | None = None
    rule_key: str | None = None
    phase_key: str | None = None
    action_key: str | None = None
    action_role: str | None = None
    constraint_key: str | None = None
    slice_role: str | None = None
    row_kind: str | None = None
    state_key: str | None = None
    from_state_key: str | None = None
    to_state_key: str | None = None
    statistic_key: str | None = None
    category_key: str | None = None
    category_position: int | None = None
    pattern_key: str | None = None
    episode_ordinal: int | None = None
    notification_ordinal: int | None = None
    event_ordinal: int | None = None
    numeric_value: float | None = None
    finding_key: str | None = None
    provenance_key: str | None = None


@dataclass(frozen=True)
class GovernanceCandidate:
    candidate_key: str
    candidate_family: Literal["model", "strategy", "warning_scenario"]
    source_task: Literal["task15", "task17", "task18"]
    source_result_position: int
    source_candidate_key: str | None
    expected_source_fingerprint: str | None
    version: str | None
    declared_role: Literal["champion", "challenger"]
    declared_state: Literal[
        "candidate", "under_review", "approved", "rejected", "retired"
    ]
    evidence_refs: tuple[GovernanceEvidenceRef, ...]


@dataclass(frozen=True)
class GovernanceCriterion:
    criterion_key: str
    candidate_family: Literal["model", "strategy", "warning_scenario"]
    source_task: Literal["task15", "task16", "task17", "task18"]
    source_table: str
    metric_key: str
    scope_key: str
    scope_position: int | None
    rule_key: str | None
    criterion_role: Literal["decision", "diagnostic"]
    required_for_promotion: bool
    direction: Literal[
        "higher_is_better", "lower_is_better", "target_range", "not_directional"
    ]
    target_low: float | None = None
    target_high: float | None = None
    minimum_support: int = 1
    required_support_unit: str | None = None
    priority: int = 0


@dataclass(frozen=True)
class GovernanceExplanation:
    explanation_key: str
    candidate_key: str
    method: Literal[
        "rule_trace",
        "reason_trace",
        "source_lineage",
        "metric_evidence",
        "scenario_delta_trace",
        "state_transition_trace",
        "coefficient_direction",
        "native_importance",
        "permutation_importance",
    ]
    source_ref: GovernanceEvidenceRef
    feature_key: str | None
    relation: Literal["positive", "negative", "neutral", "not_directional"] | None
    priority: int
    status: Literal[
        "available", "unavailable", "undefined", "not_applicable", "not_verifiable"
    ]
    reason: str | None


@dataclass(frozen=True)
class GovernanceAttributionEvidence:
    candidate_key: str
    method: Literal[
        "coefficient_direction", "native_importance", "permutation_importance"
    ]
    feature_key: str
    metric_key: str | None
    value: float
    relation: Literal["positive", "negative", "neutral", "not_directional"]
    evaluation_scope: Literal["not_applicable", "holdout", "oof"]
    support_n: int
    uncertainty_std: float | None
    permutation_repeats: int | None
    random_state: int | None
    evidence_as_of: datetime
    source_ref: GovernanceEvidenceRef


@dataclass(frozen=True)
class GovernancePredictionProfile:
    candidate_key: str
    snapshot_key: str
    snapshot_role: Literal["reference", "current"]
    analysis_as_of: datetime
    prediction_kind: Literal["ranking_score", "event_probability"]
    scope_key: str
    scope_position: int | None
    reference_state_fingerprint: str
    bin_boundaries: tuple[float, ...]
    bin_counts: tuple[int, ...]
    support_n: int
    missing_n: int
    bootstrap_repeats: int
    random_state: int
    source_ref: GovernanceEvidenceRef


@dataclass(frozen=True)
class GovernancePerformanceEvidence:
    candidate_key: str
    snapshot_key: str
    snapshot_role: Literal["reference", "current"]
    window_start: datetime
    window_end: datetime
    evidence_as_of: datetime
    evaluation_scope: Literal["holdout", "oof"]
    scope_key: str
    scope_position: int | None
    target_values: tuple[bool, ...]
    ranking_scores: tuple[float, ...] | None
    event_probabilities: tuple[float, ...] | None
    assignment_mechanism: Literal["randomized", "non_randomized", "unknown"]
    common_support: Literal["verified", "unverified"]
    bootstrap_repeats: int
    random_state: int
    source_ref: GovernanceEvidenceRef


@dataclass(frozen=True)
class GovernanceMetadata:
    metadata_key: str
    metadata_scope: Literal["governance", "candidate"]
    candidate_key: str | None
    purpose_key: str
    owner_key: str
    materiality: Literal["low", "medium", "high"]
    assumption_keys: tuple[str, ...] = ()
    limitation_keys: tuple[str, ...] = ()
    monitoring_thresholds: tuple[tuple[str, float], ...] = ()
    issue_status: Literal["none", "open", "monitoring", "resolved"] = "none"
    remediation_status: Literal[
        "not_required", "planned", "in_progress", "complete"
    ] = "not_required"


@dataclass(frozen=True)
class GovernancePolicy:
    governance_key: str
    governance_version: str
    analysis_as_of: datetime
    candidates: tuple[GovernanceCandidate, ...]
    comparison_pairs: tuple[tuple[str, str], ...]
    criteria: tuple[GovernanceCriterion, ...]
    metadata: tuple[GovernanceMetadata, ...]
    minimum_comparable_criteria: int = 1
    human_review_mode: Literal["promotion_only", "all_recommendations"] = (
        "promotion_only"
    )
    evidence_refs: tuple[GovernanceEvidenceRef, ...] = ()
    explanations: tuple[GovernanceExplanation, ...] = ()
    entity_alignment: Literal["not_requested", "owner_verified"] = "not_requested"


@dataclass(frozen=True)
class GovernanceResult:
    governance_key: str
    governance_version: str
    governance_fingerprint: str
    analysis_as_of: datetime
    candidate_count: int
    comparison_pair_count: int
    criterion_count: int
    explanation_count: int
    source_snapshot_status: Literal["verified", "unverified", "not_applicable"]
    entity_alignment_status: Literal["verified", "unverified", "not_applicable"]
    evidence_time_status: Literal["verified", "unverified", "not_applicable"]
    explanations: pd.DataFrame
    model_attributions: pd.DataFrame
    prediction_drift: pd.DataFrame
    performance_stability: pd.DataFrame
    candidate_comparisons: pd.DataFrame
    governance_evaluations: pd.DataFrame
    recommendations: pd.DataFrame
    governance_summary: pd.DataFrame
    governance_metadata: pd.DataFrame
    provenance: pd.DataFrame
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]


_SOURCE_TABLES = (
    ("task15", "metrics"),
    ("task15", "gains"),
    ("task15", "threshold_analysis"),
    ("task15", "business_metrics"),
    ("task15", "folds"),
    ("task16", "dataset_profile"),
    ("task16", "column_profile"),
    ("task16", "numeric_profile"),
    ("task16", "categorical_profile"),
    ("task16", "missingness_drift"),
    ("task16", "point_in_time_profile"),
    ("task16", "slice_profile"),
    ("task16", "resource_usage"),
    ("task16", "findings"),
    ("task16", "provenance"),
    ("task17", "row_decisions"),
    ("task17", "rule_evaluations"),
    ("task17", "rule_summary"),
    ("task17", "action_summary"),
    ("task17", "business_summary"),
    ("task17", "constraint_summary"),
    ("task17", "historical_transitions"),
    ("task17", "provenance"),
    ("task18", "monitoring_summary"),
    ("task18", "scenario_comparison"),
    ("task18", "lifecycle_summary"),
    ("task18", "rule_evaluations"),
    ("task18", "alert_episodes"),
    ("task18", "state_history"),
    ("task18", "state_transitions"),
    ("task18", "provenance"),
    ("task16", "target_profile"),
    ("task16", "missingness_patterns"),
    ("task16", "schema_drift"),
    ("task16", "collinearity"),
    ("task18", "observation_history"),
    ("task18", "notifications"),
    ("task18", "event_matches"),
)
_SOURCE_REGISTRY = tuple((i + 1, *entry) for i, entry in enumerate(_SOURCE_TABLES))


# These are the public result-table schemas owned by Tasks 15--18.  The
# governance resolver must validate the complete frozen column set before it
# looks at a locator or value; validating only the columns used by one ref
# would allow a malformed owner result to masquerade as valid evidence.
_OWNER_REQUIRED_COLUMNS: dict[tuple[str, str], tuple[str, ...]] = {
    ("task15", "folds"): tuple(_TASK15_FOLD_COLUMNS),
    ("task15", "predictions"): tuple(_TASK15_PREDICTION_COLUMNS),
    ("task15", "excluded_rows"): tuple(_TASK15_EXCLUDED_COLUMNS),
    ("task15", "metrics"): tuple(_TASK15_METRIC_COLUMNS),
    ("task15", "gains"): tuple(_TASK15_GAINS_COLUMNS),
    ("task15", "calibration"): tuple(_TASK15_CALIBRATION_COLUMNS),
    ("task15", "threshold_analysis"): tuple(_TASK15_THRESHOLD_COLUMNS),
    ("task15", "operating_point"): tuple(_TASK15_OPERATING_COLUMNS),
    ("task15", "business_metrics"): tuple(_TASK15_BUSINESS_COLUMNS),
    **{
        ("task16", name): tuple(column for column, _ in schema)
        for name, schema in _TASK16_TABLE_SCHEMAS.items()
    },
    ("task17", "row_decisions"): tuple(_TASK17_ROW_COLUMNS),
    ("task17", "rule_evaluations"): tuple(_TASK17_RULE_EVALUATION_COLUMNS),
    ("task17", "rule_summary"): tuple(_TASK17_RULE_SUMMARY_COLUMNS),
    ("task17", "action_summary"): tuple(_TASK17_ACTION_SUMMARY_COLUMNS),
    ("task17", "business_summary"): tuple(_TASK17_BUSINESS_SUMMARY_COLUMNS),
    ("task17", "constraint_summary"): tuple(_TASK17_CONSTRAINT_COLUMNS),
    ("task17", "historical_transitions"): tuple(_TASK17_TRANSITION_COLUMNS),
    ("task17", "provenance"): tuple(_TASK17_PROVENANCE_COLUMNS),
    **{
        ("task18", name): tuple(column for column, _ in schema)
        for name, schema in _TASK18_TABLE_SCHEMAS.items()
    },
}


_COMPARISON_SOURCE_FAMILIES = {
    "model": frozenset({1, 2, 3, 4}),
    "strategy": frozenset({18, 19, 20}),
    "warning_scenario": frozenset({24}),
}
_EXPLANATION_METHODS = frozenset(
    {
        "rule_trace",
        "reason_trace",
        "source_lineage",
        "metric_evidence",
        "scenario_delta_trace",
        "state_transition_trace",
        "coefficient_direction",
        "native_importance",
        "permutation_importance",
    }
)
_ATTRIBUTION_METHODS = frozenset(
    {"coefficient_direction", "native_importance", "permutation_importance"}
)
_EXPLANATION_RELATIONS = frozenset(
    {"positive", "negative", "neutral", "not_directional"}
)
_DECLARATION_STATUSES = frozenset(
    {"available", "unavailable", "undefined", "not_applicable", "not_verifiable"}
)


def _source_uses(position: int) -> tuple[str, ...]:
    if position == 1:
        return (
            "comparison_criterion",
            "explanation",
            "attribution_context",
            "drift_context",
            "stability_context",
        )
    if position in (2, 3, 4, 18, 19, 20, 24):
        return ("comparison_criterion", "explanation")
    if position in (5, 15, 23, 31):
        return (
            "diagnostic",
            "explanation",
            "attribution_context",
            "drift_context",
            "stability_context",
        )
    if position in (16, 17, 27, 28, 29, 30, 36, 37, 38):
        return ("explanation",)
    return ("diagnostic", "explanation")


_DIRECTION_GROUPS = (
    (
        "task15",
        "metrics",
        "higher_is_better",
        ("roc_auc", "average_precision", "normalized_gini", "ks_statistic"),
    ),
    (
        "task15",
        "metrics",
        "lower_is_better",
        ("brier_score", "log_loss", "expected_calibration_error"),
    ),
    ("task15", "gains", "higher_is_better", ("event_rate", "capture", "lift")),
    (
        "task15",
        "threshold_analysis",
        "higher_is_better",
        (
            "sensitivity",
            "specificity",
            "precision",
            "negative_predictive_value",
            "f1",
            "accuracy",
        ),
    ),
    ("task15", "threshold_analysis", "target_range", ("predicted_positive_rate",)),
    ("task15", "business_metrics", "higher_is_better", ("event_rate",)),
    (
        "task15",
        "business_metrics",
        "lower_is_better",
        ("observed_loss_sum", "expected_loss_sum"),
    ),
    (
        "task15",
        "business_metrics",
        "target_range",
        ("predicted_positive_rate", "exposure_sum"),
    ),
    (
        "task17",
        "rule_summary",
        "higher_is_better",
        ("captured_event_count", "target_capture_rate"),
    ),
    (
        "task17",
        "rule_summary",
        "lower_is_better",
        (
            "unknown_count",
            "unknown_rate",
            "not_evaluated_count",
            "overlap_count",
            "overlap_rate",
            "conflict_count",
        ),
    ),
    (
        "task17",
        "rule_summary",
        "target_range",
        (
            "hit_count",
            "hit_rate",
            "applied_count",
            "sole_hit_count",
            "incremental_action_count",
            "leave_one_out_changed_action_count",
        ),
    ),
    (
        "task17",
        "action_summary",
        "higher_is_better",
        ("assumed_action_value_sum", "assumption_based_payoff_sum"),
    ),
    (
        "task17",
        "action_summary",
        "lower_is_better",
        (
            "event_count",
            "event_rate",
            "expected_loss_sum",
            "assumption_based_observed_event_loss_sum",
            "assumed_action_cost_sum",
        ),
    ),
    (
        "task17",
        "action_summary",
        "target_range",
        ("action_count", "action_rate", "exposure_sum"),
    ),
    (
        "task17",
        "business_summary",
        "higher_is_better",
        (
            "decided_rate",
            "assumed_action_value_sum",
            "assumption_based_payoff_sum",
            "historical_mapped_rate",
        ),
    ),
    (
        "task17",
        "business_summary",
        "lower_is_better",
        (
            "observed_event_count",
            "observed_event_rate",
            "unknown_action_rate",
            "expected_loss_sum",
            "expected_loss_rate",
            "assumption_based_observed_event_loss_sum",
            "actual_observed_loss_sum",
            "actual_observed_loss_rate",
            "assumed_action_cost_sum",
            "selected_event_rate",
        ),
    ),
    (
        "task17",
        "business_summary",
        "target_range",
        ("selected_rate", "rejected_rate", "review_capacity_rate", "exposure_sum"),
    ),
    (
        "task18",
        "monitoring_summary",
        "higher_is_better",
        (
            "resolved_episode_count",
            "captured_event_count",
            "event_recall",
            "notification_precision",
            "lead_time_mean",
            "lead_time_median",
            "warning_to_event_rate",
        ),
    ),
    (
        "task18",
        "monitoring_summary",
        "lower_is_better",
        (
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
            "episode_duration_mean",
            "episode_duration_median",
            "false_alert_share",
            "false_positive_rate",
            "expected_loss_sum",
            "expected_loss_rate",
            "observed_loss_sum",
            "observed_loss_rate",
        ),
    ),
    ("task18", "monitoring_summary", "target_range", ("exposure_sum",)),
)
_DIRECTION_REGISTRY = tuple(
    (task, table, metric, direction)
    for task, table, direction, metrics in _DIRECTION_GROUPS
    for metric in metrics
)

_ERROR_KEYS = (
    "invalid_policy_type",
    "invalid_governance_key",
    "invalid_governance_version",
    "invalid_analysis_as_of",
    "datetime_awareness_mismatch",
    "invalid_candidate_container",
    "invalid_pair_container",
    "invalid_criterion_container",
    "invalid_metadata_container",
    "invalid_evidence_ref_container",
    "invalid_explanation_container",
    "invalid_attribution_container",
    "invalid_prediction_profile_container",
    "invalid_performance_evidence_container",
    "invalid_owner_result_container",
    "invalid_human_review_mode",
    "invalid_entity_alignment",
    "invalid_candidate",
    "duplicate_candidate",
    "invalid_champion",
    "invalid_pair",
    "invalid_pair_coverage",
    "duplicate_pair",
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
    "authoritative_time_missing",
    "authoritative_time_mismatch",
    "future_evidence_time",
    "invalid_canonical_value",
    "resource_candidates",
    "resource_comparison_pairs",
    "resource_criteria",
    "resource_explanations",
    "resource_model_attribution_rows",
    "resource_attribution_permutation_repeats",
    "resource_prediction_profile_rows",
    "resource_prediction_profile_bin_count",
    "resource_performance_evidence_rows",
    "resource_performance_vector_values",
    "resource_drift_bootstrap_draws",
    "resource_performance_bootstrap_draws",
    "resource_prediction_drift_rows",
    "resource_performance_stability_rows",
    "resource_governance_metadata_rows",
    "resource_source_evidence_rows",
    "resource_candidate_comparison_rows",
    "resource_governance_evaluation_rows",
    "resource_recommendation_rows",
    "resource_governance_summary_rows",
    "resource_provenance_rows",
    "resource_risk_validation_results",
    "resource_data_audit_results",
    "resource_decision_strategy_results",
    "resource_lifecycle_monitoring_results",
    "resource_evidence_refs",
)
_REASONS = (
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

_RESOURCE_GATES = (
    ("risk_validation_results", 16, "resource_risk_validation_results"),
    ("data_audit_results", 16, "resource_data_audit_results"),
    ("decision_strategy_results", 16, "resource_decision_strategy_results"),
    ("lifecycle_monitoring_results", 16, "resource_lifecycle_monitoring_results"),
    ("candidates", 16, "resource_candidates"),
    ("comparison_pairs", 15, "resource_comparison_pairs"),
    ("criteria", 64, "resource_criteria"),
    ("explanations", 4096, "resource_explanations"),
    ("model_attribution_rows", 4096, "resource_model_attribution_rows"),
    (
        "attribution_permutation_repeats",
        100,
        "resource_attribution_permutation_repeats",
    ),
    ("prediction_profile_rows", 4096, "resource_prediction_profile_rows"),
    ("performance_evidence_rows", 4096, "resource_performance_evidence_rows"),
    ("governance_metadata_rows", 256, "resource_governance_metadata_rows"),
    ("evidence_refs", 8192, "resource_evidence_refs"),
    (
        "performance_vector_values",
        200000,
        "resource_performance_vector_values",
    ),
    ("drift_bootstrap_draws", 2000000, "resource_drift_bootstrap_draws"),
    (
        "performance_bootstrap_draws",
        2000000,
        "resource_performance_bootstrap_draws",
    ),
)

_FIXED_INVARIANTS = (
    ("prediction_profile_bin_count", "resource_prediction_profile_bin_count"),
    ("source_evidence_rows", "resource_source_evidence_rows"),
    ("prediction_drift_rows", "resource_prediction_drift_rows"),
    ("performance_stability_rows", "resource_performance_stability_rows"),
    ("candidate_comparison_rows", "resource_candidate_comparison_rows"),
    ("governance_evaluation_rows", "resource_governance_evaluation_rows"),
    ("recommendation_rows", "resource_recommendation_rows"),
    ("governance_summary_rows", "resource_governance_summary_rows"),
    ("provenance_rows", "resource_provenance_rows"),
)


def _resource_preflight(projected: tuple[int, ...]) -> None:
    """Apply the closed caller-variable resource registry in ordinal order."""
    if type(projected) is not tuple or len(projected) != len(_RESOURCE_GATES):
        _fail("invalid_canonical_value")
    for actual, (_, maximum, error) in zip(projected, _RESOURCE_GATES, strict=True):
        if not _exact_int(actual):
            _fail("invalid_canonical_value")
        if actual > maximum:
            _fail(error)


def _proof_status(
    left_task: str | None,
    left_position: int | None,
    right_task: str | None,
    right_position: int | None,
    *,
    alignment_required: bool,
) -> tuple[str, str]:
    """Classify snapshot and entity proof without inferring from public values."""
    if right_task is None or right_position is None:
        return "not_applicable", "not_applicable"
    same_owner = left_task == right_task and left_position == right_position
    snapshot = "verified" if same_owner else "unverified"
    alignment = (
        "verified"
        if same_owner and alignment_required
        else "unverified"
        if alignment_required
        else "not_applicable"
    )
    return snapshot, alignment


def _comparison_proof_failure(
    criterion: GovernanceCriterion,
    champion_ref: GovernanceEvidenceRef,
    challenger_ref: GovernanceEvidenceRef,
    champion_time_status: str,
    challenger_time_status: str,
    *,
    alignment_required: bool,
) -> tuple[str, str, str | None]:
    """Return the first missing proof dimension for a comparison.

    This is the single comparison-eligibility proof gate.  Owner status and
    source binding are validated by the caller before this helper runs; a
    legal but unverified evidence pair therefore remains a result row rather
    than becoming a hard input error.  Aggregate owner metrics deliberately
    do not claim a raw snapshot/entity relationship, while Task 18 scenario
    pairs may prove the snapshot within one owner result but do not prove
    entity alignment merely from that pair.
    """
    snapshot, alignment = _proof_status(
        champion_ref.source_task,
        champion_ref.source_result_position,
        challenger_ref.source_task,
        challenger_ref.source_result_position,
        alignment_required=alignment_required,
    )

    # Task 15 decision metrics and Task 17/18 overall rows are aggregate
    # evidence.  Their contract explicitly fixes both raw-row proof
    # dimensions to not_applicable.
    aggregate = champion_ref.source_task == challenger_ref.source_task and (
        champion_ref.source_task == "task15"
        or (
            champion_ref.source_task in {"task17", "task18"}
            and criterion.scope_key == "overall"
        )
    )
    if aggregate:
        snapshot = "not_applicable"
        alignment = "not_applicable"

    # A Task 18 R/C pair within one owner result is a valid AM-04 snapshot
    # pair, but it carries no owner-verified entity proof.  Keep this
    # independent from the scenario-pair snapshot classification so a caller
    # requesting entity alignment receives the exact alignment reason.
    if (
        alignment_required
        and champion_ref.source_task == challenger_ref.source_task == "task18"
        and criterion.scope_key != "overall"
        and champion_ref.source_result_position == challenger_ref.source_result_position
        and champion_ref.entity_position is None
        and challenger_ref.entity_position is None
    ):
        alignment = "unverified"

    # Subordinate locator identities are part of alignment proof when the
    # caller explicitly requests owner-verified alignment.  Equal owner
    # positions alone cannot upgrade a differing entity/time locator.
    if alignment_required and alignment == "verified":
        if (
            champion_ref.entity_position != challenger_ref.entity_position
            or champion_ref.time_position != challenger_ref.time_position
        ):
            alignment = "unverified"

    if champion_time_status != "verified" or challenger_time_status != "verified":
        return snapshot, alignment, "time_unverified"

    if snapshot == "unverified":
        return snapshot, alignment, "snapshot_unverified"
    if alignment == "unverified":
        return snapshot, alignment, "alignment_unverified"
    return snapshot, alignment, None


def _task19_source_status_reason(
    status: object, reason: object
) -> tuple[str, str] | None:
    """Map a legal owner status/reason pair to Task 19 semantics."""
    reason_key = reason if type(reason) is str else None
    if status == "unavailable":
        return "unavailable", "source_unavailable"
    if status == "undefined":
        return "undefined", reason_key if reason_key in {
            "source_undefined",
            "insufficient_support",
            "insufficient_bootstrap_support",
            "zero_denominator",
            "single_class",
        } else "source_undefined"
    if status == "not_verifiable":
        return "not_verifiable", reason_key if reason_key in {
            "source_not_verifiable",
            "support_not_comparable",
            "maturity_not_comparable",
            "snapshot_unverified",
            "alignment_unverified",
            "time_unverified",
            "common_support_unverified",
        } else "source_not_verifiable"
    if status == "not_applicable":
        return "not_applicable", "operation_not_applicable"
    return None


def _validate_fixed_invariants(
    *,
    prediction_profiles: int,
    source_evidence_rows: int,
    evidence_refs: int,
    prediction_drift_rows: int,
    performance_evidence: int,
    performance_stability_rows: int,
    pairs: int,
    criteria: int,
    candidate_comparison_rows: int,
    governance_evaluation_rows: int,
    recommendation_rows: int,
    candidates: int,
    governance_summary_rows: int,
    provenance_rows: int,
) -> None:
    """Defensively assert the nine frozen cardinality invariants."""
    checks = (
        prediction_profiles % 2 == 0,
        source_evidence_rows == evidence_refs,
        prediction_drift_rows == prediction_profiles // 2,
        performance_stability_rows == performance_evidence // 2,
        candidate_comparison_rows == pairs * criteria,
        governance_evaluation_rows == pairs * criteria,
        recommendation_rows == pairs,
        governance_summary_rows == candidates,
        provenance_rows == 35,
    )
    for valid, (_, error) in zip(checks, _FIXED_INVARIANTS, strict=True):
        if not valid:
            _fail(error)


_SCHEMAS: dict[str, tuple[tuple[str, str], ...]] = {
    "explanations": tuple((x, "Int64") for x in ("explanation_position",))
    + tuple((x, "string") for x in ("explanation_key",))
    + (("candidate_position", "Int64"),)
    + tuple((x, "string") for x in ("candidate_family", "method"))
    + (("source_ref_position", "Int64"),)
    + tuple((x, "string") for x in ("source_task",))
    + (("source_result_position", "Int64"),)
    + (("source_table", "string"), ("source_registry_position", "Int64"))
    + tuple((x, "string") for x in ("source_fingerprint", "feature_key", "relation"))
    + (("priority", "Int64"),)
    + tuple(
        (x, "string")
        for x in (
            "evidence_time_status",
            "source_status",
            "source_reason",
            "status",
            "reason",
            "finding_key",
        )
    ),
    "model_attributions": (
        ("candidate_position", "Int64"),
        ("attribution_position", "Int64"),
    )
    + tuple(
        (x, "string")
        for x in ("candidate_family", "method", "feature_key", "metric_key")
    )
    + (
        ("value", "Float64"),
        ("relation", "string"),
        ("evaluation_scope", "string"),
        ("support_n", "Int64"),
        ("uncertainty_std", "Float64"),
        ("permutation_repeats", "Int64"),
        ("random_state", "Int64"),
        ("evidence_as_of", "datetime"),
        ("evidence_time_status", "string"),
        ("source_task", "string"),
        ("source_result_position", "Int64"),
        ("source_table", "string"),
        ("source_ref_position", "Int64"),
        ("source_fingerprint", "string"),
        ("source_status", "string"),
        ("source_reason", "string"),
        ("status", "string"),
        ("reason", "string"),
        ("finding_key", "string"),
    ),
    "prediction_drift": (
        ("candidate_position", "Int64"),
        ("reference_profile_position", "Int64"),
        ("current_profile_position", "Int64"),
    )
    + tuple((x, "string") for x in ("prediction_kind", "scope_key"))
    + (("scope_position", "Int64"),)
    + tuple((x, "string") for x in ("reference_snapshot_key", "current_snapshot_key"))
    + (("reference_analysis_as_of", "datetime"), ("current_analysis_as_of", "datetime"))
    + tuple((x, "string") for x in ("reference_time_status", "current_time_status"))
    + tuple(
        (x, "Int64")
        for x in (
            "reference_support_n",
            "current_support_n",
            "reference_missing_n",
            "current_missing_n",
            "bin_count",
        )
    )
    + tuple(
        (x, "string")
        for x in (
            "reference_state_fingerprint",
            "reference_source_fingerprint",
            "current_source_fingerprint",
            "metric",
        )
    )
    + (
        ("prediction_tvd", "Float64"),
        ("direction", "string"),
        ("uncertainty_std", "Float64"),
        ("bootstrap_repeats", "Int64"),
        ("random_state", "Int64"),
    )
    + tuple((x, "string") for x in ("status", "reason", "finding_key")),
    "performance_stability": (
        ("candidate_position", "Int64"),
        ("reference_evidence_position", "Int64"),
        ("current_evidence_position", "Int64"),
    )
    + tuple((x, "string") for x in ("evaluation_scope", "scope_key"))
    + (("scope_position", "Int64"),)
    + tuple((x, "string") for x in ("reference_snapshot_key", "current_snapshot_key"))
    + tuple(
        (x, "datetime")
        for x in (
            "reference_window_start",
            "reference_window_end",
            "current_window_start",
            "current_window_end",
            "reference_evidence_as_of",
            "current_evidence_as_of",
        )
    )
    + tuple(
        (x, "string")
        for x in ("reference_time_status", "current_time_status", "metric")
    )
    + tuple((x, "Float64") for x in ("reference_value", "current_value", "delta"))
    + (("direction", "string"),)
    + tuple(
        (x, "Float64") for x in ("reference_uncertainty_std", "current_uncertainty_std")
    )
    + tuple(
        (x, "Int64")
        for x in (
            "reference_bootstrap_repeats",
            "current_bootstrap_repeats",
            "reference_random_state",
            "current_random_state",
            "reference_support_n",
            "current_support_n",
        )
    )
    + tuple(
        (x, "string")
        for x in (
            "reference_assignment_mechanism",
            "current_assignment_mechanism",
            "reference_common_support",
            "current_common_support",
            "reference_source_fingerprint",
            "current_source_fingerprint",
            "status",
            "reason",
            "finding_key",
        )
    ),
}

_SCHEMAS.update(
    {
        "candidate_comparisons": tuple(
            (x, "Int64")
            for x in (
                "pair_position",
                "champion_candidate_position",
                "challenger_candidate_position",
            )
        )
        + (
            ("candidate_family", "string"),
            ("criterion_position", "Int64"),
            ("criterion_role", "string"),
            ("source_task", "string"),
            ("source_table", "string"),
            ("metric_key", "string"),
            ("scope_key", "string"),
            ("scope_position", "Int64"),
            ("rule_key", "string"),
        )
        + tuple(
            (x, "Int64")
            for x in (
                "champion_source_result_position",
                "challenger_source_result_position",
                "champion_source_ref_position",
                "challenger_source_ref_position",
            )
        )
        + tuple(
            (x, "string")
            for x in (
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
            )
        )
        + tuple((x, "Float64") for x in ("champion_value", "challenger_value", "delta"))
        + (
            ("champion_support_n", "Int64"),
            ("challenger_support_n", "Int64"),
            ("support_unit", "string"),
            ("direction", "string"),
            ("target_low", "Float64"),
            ("target_high", "Float64"),
            ("comparison_outcome", "string"),
            ("support_comparable", "boolean"),
            ("status", "string"),
            ("reason", "string"),
            ("finding_key", "string"),
        ),
        "governance_evaluations": tuple(
            (x, "Int64")
            for x in (
                "pair_position",
                "champion_candidate_position",
                "challenger_candidate_position",
                "criterion_position",
            )
        )
        + (
            ("criterion_role", "string"),
            ("required_for_promotion", "boolean"),
            ("priority", "Int64"),
            ("comparison_outcome", "string"),
            ("comparable", "boolean"),
            ("counts_toward_minimum", "boolean"),
            ("blocks_promotion", "boolean"),
            ("directional_contribution", "string"),
            ("evidence_time_status", "string"),
            ("status", "string"),
            ("reason", "string"),
            ("finding_key", "string"),
        ),
        "recommendations": tuple(
            (x, "Int64")
            for x in (
                "pair_position",
                "champion_candidate_position",
                "challenger_candidate_position",
            )
        )
        + (
            ("candidate_family", "string"),
            ("recommendation", "string"),
            ("recommendation_basis", "string"),
            ("hard_veto", "boolean"),
            ("human_review_mode", "string"),
            ("human_review_required", "boolean"),
        )
        + tuple(
            (x, "Int64")
            for x in (
                "minimum_comparable_criteria",
                "criteria_available_n",
                "criteria_unavailable_n",
                "criteria_better_n",
                "criteria_worse_n",
                "criteria_tied_n",
                "required_incomplete_n",
            )
        )
        + (
            ("support_comparable", "boolean"),
            ("status", "string"),
            ("reason", "string"),
            ("finding_key", "string"),
        ),
        "governance_summary": (("candidate_position", "Int64"),)
        + tuple(
            (x, "string")
            for x in (
                "candidate_family",
                "declared_role",
                "declared_state",
                "source_task",
            )
        )
        + (("source_result_position", "Int64"), ("source_candidate_position", "Int64"))
        + tuple(
            (x, "string")
            for x in (
                "source_snapshot_status",
                "entity_alignment_status",
                "evidence_time_status",
            )
        )
        + tuple(
            (x, "Int64")
            for x in (
                "criterion_count",
                "available_criterion_count",
                "unavailable_criterion_count",
                "not_verifiable_criterion_count",
                "attribution_count",
                "prediction_drift_count",
                "performance_stability_count",
                "recommendation_count",
                "human_review_required_count",
            )
        )
        + (("status", "string"), ("reason", "string"), ("finding_key", "string")),
        "governance_metadata": (
            ("metadata_position", "Int64"),
            ("metadata_scope", "string"),
            ("candidate_position", "Int64"),
            ("field_position", "Int64"),
            ("field_key", "string"),
            ("item_position", "Int64"),
            ("text_value", "string"),
            ("numeric_value", "Float64"),
            ("evidence_time_status", "string"),
            ("status", "string"),
            ("reason", "string"),
            ("finding_key", "string"),
        ),
        "provenance": (
            ("provenance_position", "Int64"),
            ("provenance_key", "string"),
            ("provenance_value", "string"),
            ("status", "string"),
            ("reason", "string"),
            ("finding_key", "string"),
        ),
    }
)


def _fail(key: str) -> None:
    raise ValueError(f"model governance: {key}")


def _is_safe_key(value: object) -> bool:
    if type(value) is not str or not value:
        return False
    return all(c.isascii() and (c.isalnum() or c in "_.:-") for c in value)


def _exact_int(value: object, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _exact_float(value: object) -> bool:
    return type(value) is float and math.isfinite(value)


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


_OWNER_STATUS_VOCABULARY = frozenset(
    {
        "available",
        "unavailable",
        "undefined",
        "not_applicable",
        "not_verifiable",
        "computed",
        "empty_bin",
        "source_not_requested",
        "evaluated",
        "not_evaluated",
        "inactive",
        "active",
        "pending",
        "clear",
        "resolved",
        "emitted",
        "suppressed",
        "not_emitted",
        "mature",
        "immature",
        "verified",
        "unverified",
    }
)

_OWNER_REASON_VOCABULARY = frozenset(
    {
        "computed",
        "empty_bin",
        "source_not_requested",
        "single_class",
        "zero_denominator",
        "insufficient_support",
        "insufficient_bootstrap_support",
        "not_threshold_segment",
        "exposure_absent",
        "observed_loss_absent",
        "probability_unavailable",
        "label_not_evaluable",
        "observed_loss_not_resegmentable",
        "observed_loss_not_mature",
        "exposure_unavailable",
        "action_assumption_not_declared",
        "strategy_inactive",
        "unknown_condition",
        "rule_inactive",
        "default_action_applied",
        "rule_conflict",
        "constraint_satisfied",
        "constraint_failed",
        "transition_allowed",
        "entry_observation",
        "episode_resolved",
        "event_captured",
        "default_state_applied",
        "episode_active",
        "maturity_not_comparable",
        "source_unavailable",
        "source_undefined",
        "source_not_verifiable",
        "support_not_comparable",
        "snapshot_unverified",
        "alignment_unverified",
        "time_unverified",
        "common_support_unverified",
        "not_provided",
    }
)


def _is_datetime(value: object) -> bool:
    return (type(value) is datetime or type(value) is pd.Timestamp) and not pd.isna(
        value
    )


def _aware(value: datetime | pd.Timestamp) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def _normal_time(value: datetime | pd.Timestamp) -> datetime | pd.Timestamp:
    if not _aware(value):
        return value
    if type(value) is pd.Timestamp:
        return value.tz_convert("UTC")
    return value.astimezone(timezone.utc)


def _canonical_datetime(value: datetime | pd.Timestamp) -> dict[str, str]:
    normalized = _normal_time(value)
    suffix = "Z" if _aware(value) else ""
    if type(normalized) is pd.Timestamp:
        stamp = normalized.strftime("%Y-%m-%dT%H:%M:%S")
        fraction = f"{normalized.microsecond * 1000 + normalized.nanosecond:09d}"
        return {"t": "timestamp", "v": f"{stamp}.{fraction}{suffix}"}
    return {"t": "datetime", "v": normalized.strftime("%Y-%m-%dT%H:%M:%S.%f") + suffix}


_CANONICAL_DATACLASSES = (
    GovernanceEvidenceRef,
    GovernanceCandidate,
    GovernanceCriterion,
    GovernanceExplanation,
    GovernanceAttributionEvidence,
    GovernancePredictionProfile,
    GovernancePerformanceEvidence,
    GovernanceMetadata,
    GovernancePolicy,
)


def _canonical_node(value: object, *, private_dict: bool = False) -> dict[str, Any]:
    if value is None:
        return {"t": "none"}
    if type(value) is bool:
        return {"t": "bool", "v": value}
    if type(value) is int:
        return {"t": "int", "v": f"{value:d}"}
    if type(value) is float:
        if not math.isfinite(value):
            _fail("invalid_canonical_value")
        return {"t": "float", "v": value.hex()}
    if type(value) is str:
        return {"t": "str", "v": value}
    if _is_datetime(value):
        return _canonical_datetime(value)  # type: ignore[arg-type]
    if type(value) is tuple:
        return {"t": "tuple", "v": [_canonical_node(x) for x in value]}
    if type(value) is dict and private_dict:
        if any(type(k) is not str for k in value):
            _fail("invalid_canonical_value")
        return {
            "t": "map",
            "v": [[k, _canonical_node(value[k])] for k in sorted(value)],
        }
    if type(value) in _CANONICAL_DATACLASSES:
        return {
            "t": "dataclass",
            "n": type(value).__name__,
            "v": [
                [f.name, _canonical_node(getattr(value, f.name))] for f in fields(value)
            ],
        }
    _fail("invalid_canonical_value")


def _canonical_json(value: object, *, private_dict: bool = False) -> str:
    return json.dumps(
        _canonical_node(value, private_dict=private_dict),
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def _fingerprint(value: object, *, private_dict: bool = False) -> str:
    return hashlib.sha256(
        _canonical_json(value, private_dict=private_dict).encode("utf-8")
    ).hexdigest()


def _frame(name: str, rows: list[dict[str, object]], *, aware: bool) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    for column, dtype in _SCHEMAS[name]:
        values = [row.get(column, pd.NA) for row in rows]
        if dtype == "datetime":
            actual = "datetime64[ns, UTC]" if aware else "datetime64[ns]"
            data[column] = pd.Series(values, dtype=actual)
        else:
            data[column] = pd.Series(values, dtype=dtype)
    return pd.DataFrame(data, columns=[x for x, _ in _SCHEMAS[name]])


def _owner_fingerprint(task: str, owner: object) -> str | None:
    if task == "task15":
        return None
    return getattr(
        owner,
        {
            "task16": "config_fingerprint",
            "task17": "strategy_fingerprint",
            "task18": "monitoring_fingerprint",
        }[task],
    )


def _owner_collections(
    risk_validations: tuple[BinaryRiskValidationResult, ...],
    data_audits: tuple[DataAuditResult, ...],
    decision_strategies: tuple[DecisionStrategyResult, ...],
    lifecycle_monitorings: tuple[LifecycleMonitoringResult, ...],
) -> dict[str, tuple[object, ...]]:
    return {
        "task15": risk_validations,
        "task16": data_audits,
        "task17": decision_strategies,
        "task18": lifecycle_monitorings,
    }


def _validate_container(
    value: object,
    expected: type,
    key: str,
    *,
    duplicate_key: str = "duplicate_owner_source",
) -> None:
    if type(value) is not tuple or any(type(x) is not expected for x in value):
        _fail(key)
    if len({id(x) for x in value}) != len(value):
        _fail(duplicate_key)


def _validate_ref_shape(ref: GovernanceEvidenceRef) -> None:
    if type(ref) is not GovernanceEvidenceRef:
        _fail("invalid_evidence_ref")
    if ref.source_task not in (
        "task15",
        "task16",
        "task17",
        "task18",
    ) or not _exact_int(ref.source_result_position):
        _fail("invalid_evidence_ref")
    if type(ref.source_table) is not str or type(ref.source_use) is not str:
        _fail("invalid_evidence_ref")
    for f in fields(ref):
        value = getattr(ref, f.name)
        if (
            value is not None
            and f.name.endswith(("position", "ordinal", "fold_id"))
            and not _exact_int(value)
        ):
            _fail("invalid_source_locator")
        if (
            value is not None
            and f.name in ("numeric_value",)
            and not _exact_float(value)
        ):
            _fail("invalid_source_locator")


def _owner_task(owner: object) -> str | None:
    if type(owner) is BinaryRiskValidationResult:
        return "task15"
    if type(owner) is DataAuditResult:
        return "task16"
    if type(owner) is DecisionStrategyResult:
        return "task17"
    if type(owner) is LifecycleMonitoringResult:
        return "task18"
    return None


def _validate_owner_table(owner: object, table_name: str, table: object) -> None:
    """Validate an owner table's frozen schema and cell status/reason domains."""
    if type(table) is not pd.DataFrame:
        _fail("invalid_owner_schema")
    task = _owner_task(owner)
    expected_columns = _OWNER_REQUIRED_COLUMNS.get((task, table_name))
    if expected_columns is None or list(table.columns) != list(expected_columns):
        _fail("invalid_owner_schema")
    numeric_columns = {
        "value",
        "metric_value",
        "support_n",
        "n_rows",
        "n_evaluable_rows",
        "numerator",
        "denominator",
    }
    for column in table.columns:
        if column in numeric_columns and table[column].dtype == "object":
            _fail("invalid_owner_dtype")
    for column in table.columns:
        if column == "status" or column.endswith("_status"):
            values = table[column].dropna().tolist()
            if any(
                type(value) is not str or value not in _OWNER_STATUS_VOCABULARY
                for value in values
            ):
                _fail("invalid_owner_status")
        if column == "reason" or column.endswith("_reason"):
            values = table[column].dropna().tolist()
            if any(
                type(value) is not str or value not in _OWNER_REASON_VOCABULARY
                for value in values
            ):
                _fail("invalid_owner_reason")


def _validate_structured_ref(
    ref: GovernanceEvidenceRef,
    *,
    candidate_key: str,
    source_use: str,
    error_key: str,
) -> None:
    """Validate a structured carrier's candidate and exact source-use tag."""
    if ref.candidate_key != candidate_key:
        _fail("invalid_source_binding")
    if ref.source_use != source_use:
        _fail("invalid_evidence_ref")


def _validate_explanation_declaration(
    item: GovernanceExplanation,
    candidate_by_key: dict[str, GovernanceCandidate],
) -> None:
    if (
        type(item) is not GovernanceExplanation
        or not _is_safe_key(item.explanation_key)
        or item.candidate_key not in candidate_by_key
        or item.method not in _EXPLANATION_METHODS
        or type(item.feature_key) not in (str, type(None))
        or (item.feature_key is not None and not _is_safe_key(item.feature_key))
        or item.relation not in _EXPLANATION_RELATIONS | {None}
        or type(item.priority) is not int
        or item.status not in _DECLARATION_STATUSES
        or type(item.reason) not in (str, type(None))
        or (item.reason is not None and item.reason not in _REASONS)
        or (item.status == "available" and item.reason is not None)
        or (item.status != "available" and item.reason is None)
    ):
        _fail("invalid_explanation")
    candidate = candidate_by_key[item.candidate_key]
    if item.method in _ATTRIBUTION_METHODS and candidate.candidate_family != "model":
        _fail("invalid_explanation")
    _validate_structured_ref(
        item.source_ref,
        candidate_key=item.candidate_key,
        source_use="explanation",
        error_key="invalid_explanation",
    )
    source_position = next(
        (
            position
            for position, task, table in _SOURCE_REGISTRY
            if task == item.source_ref.source_task
            and table == item.source_ref.source_table
        ),
        None,
    )
    allowed_positions = {
        "rule_trace": {17, 18, 27},
        "reason_trace": set(range(1, 39)),
        "source_lineage": {5, 15, 23, 31},
        "metric_evidence": {1, 2, 3, 4, 10, 11, 12, 18, 19, 20, 24, 26, 32, 33, 34, 35},
        "scenario_delta_trace": {25},
        "state_transition_trace": {30},
        "coefficient_direction": {1, 5},
        "native_importance": {1, 5},
        "permutation_importance": {1, 5},
    }
    if source_position not in allowed_positions[item.method]:
        _fail("invalid_explanation")


def _validate_attribution_declaration(
    item: GovernanceAttributionEvidence,
    candidate_by_key: dict[str, GovernanceCandidate],
) -> None:
    if item.candidate_key not in candidate_by_key:
        _fail("invalid_attribution")
    if (
        type(item) is not GovernanceAttributionEvidence
        or item.method not in _ATTRIBUTION_METHODS
        or not _is_safe_key(item.feature_key)
        or (item.metric_key is not None and not _is_safe_key(item.metric_key))
        or not _exact_float(item.value)
        or item.relation not in _EXPLANATION_RELATIONS
        or item.evaluation_scope not in ("not_applicable", "holdout", "oof")
        or not _exact_int(item.support_n)
        or (
            item.uncertainty_std is not None
            and (not _exact_float(item.uncertainty_std) or item.uncertainty_std < 0)
        )
        or (
            item.permutation_repeats is not None
            and not _exact_int(item.permutation_repeats, minimum=1)
        )
        or (item.random_state is not None and not _exact_int(item.random_state))
        or not _is_datetime(item.evidence_as_of)
    ):
        _fail("invalid_attribution")
    if candidate_by_key[item.candidate_key].candidate_family != "model":
        _fail("invalid_source_binding")
    if item.method == "coefficient_direction":
        expected = (
            "positive"
            if item.value > 0
            else "negative"
            if item.value < 0
            else "neutral"
        )
        if (
            item.relation != expected
            or item.metric_key is not None
            or item.evaluation_scope != "not_applicable"
            or item.uncertainty_std is not None
            or item.permutation_repeats is not None
            or item.random_state is not None
        ):
            _fail("invalid_attribution")
    elif item.method == "native_importance":
        if (
            item.value < 0
            or item.relation != "not_directional"
            or item.metric_key is not None
            or item.evaluation_scope != "not_applicable"
            or item.uncertainty_std is not None
            or item.permutation_repeats is not None
            or item.random_state is not None
        ):
            _fail("invalid_attribution")
    elif not (
        item.uncertainty_std is not None
        and item.permutation_repeats is not None
        and item.random_state is not None
        and item.evaluation_scope in ("holdout", "oof")
        and type(item.metric_key) is str
    ):
        _fail("invalid_attribution")
    _validate_structured_ref(
        item.source_ref,
        candidate_key=item.candidate_key,
        source_use="attribution_context",
        error_key="invalid_attribution",
    )
    if item.source_ref.source_task != "task15" or item.source_ref.source_table not in (
        "metrics",
        "folds",
    ):
        _fail("invalid_source_binding")
    if (
        item.source_ref.source_result_position
        != candidate_by_key[item.candidate_key].source_result_position
    ):
        _fail("invalid_source_binding")


def _validate_prediction_declaration(
    item: GovernancePredictionProfile,
    candidate_by_key: dict[str, GovernanceCandidate],
) -> None:
    if item.candidate_key not in candidate_by_key:
        _fail("invalid_prediction_profile")
    if (
        type(item) is not GovernancePredictionProfile
        or not _is_safe_key(item.snapshot_key)
        or item.snapshot_role not in ("reference", "current")
        or not _is_datetime(item.analysis_as_of)
        or item.prediction_kind not in ("ranking_score", "event_probability")
        or not _is_safe_key(item.scope_key)
        or (item.scope_position is not None and not _exact_int(item.scope_position))
        or not _valid_sha256(item.reference_state_fingerprint)
        or type(item.bin_boundaries) is not tuple
        or type(item.bin_counts) is not tuple
        or not _exact_int(item.support_n)
        or not _exact_int(item.missing_n)
        or not 2 <= item.bootstrap_repeats <= 1000
        or not _exact_int(item.random_state)
    ):
        _fail("invalid_prediction_profile")
    if candidate_by_key[item.candidate_key].candidate_family != "model":
        _fail("invalid_source_binding")
    _validate_structured_ref(
        item.source_ref,
        candidate_key=item.candidate_key,
        source_use="drift_context",
        error_key="invalid_prediction_profile",
    )
    if item.source_ref.source_task != "task15" or item.source_ref.source_table not in (
        "metrics",
        "folds",
    ):
        _fail("invalid_source_binding")
    if (
        item.source_ref.source_result_position
        != candidate_by_key[item.candidate_key].source_result_position
    ):
        _fail("invalid_source_binding")


def _validate_performance_declaration(
    item: GovernancePerformanceEvidence,
    candidate_by_key: dict[str, GovernanceCandidate],
) -> None:
    if item.candidate_key not in candidate_by_key:
        _fail("invalid_performance_evidence")
    if (
        type(item) is not GovernancePerformanceEvidence
        or not _is_safe_key(item.snapshot_key)
        or item.snapshot_role not in ("reference", "current")
        or not all(
            _is_datetime(x)
            for x in (item.window_start, item.window_end, item.evidence_as_of)
        )
        or item.evaluation_scope not in ("holdout", "oof")
        or not _is_safe_key(item.scope_key)
        or (item.scope_position is not None and not _exact_int(item.scope_position))
        or type(item.target_values) is not tuple
        or any(type(x) is not bool for x in item.target_values)
        or item.assignment_mechanism not in ("randomized", "non_randomized", "unknown")
        or item.common_support not in ("verified", "unverified")
        or not 2 <= item.bootstrap_repeats <= 1000
        or not _exact_int(item.random_state)
    ):
        _fail("invalid_performance_evidence")
    if candidate_by_key[item.candidate_key].candidate_family != "model":
        _fail("invalid_source_binding")
    _validate_structured_ref(
        item.source_ref,
        candidate_key=item.candidate_key,
        source_use="stability_context",
        error_key="invalid_performance_evidence",
    )
    if item.source_ref.source_task != "task15" or item.source_ref.source_table not in (
        "metrics",
        "folds",
    ):
        _fail("invalid_source_binding")
    if (
        item.source_ref.source_result_position
        != candidate_by_key[item.candidate_key].source_result_position
    ):
        _fail("invalid_source_binding")


def _candidate_owner(
    candidate: GovernanceCandidate, owners: dict[str, tuple[object, ...]]
) -> object:
    expected = {"model": "task15", "strategy": "task17", "warning_scenario": "task18"}
    if (
        candidate.candidate_family not in expected
        or candidate.source_task != expected[candidate.candidate_family]
    ):
        _fail("invalid_source_binding")
    if not _exact_int(
        candidate.source_result_position
    ) or candidate.source_result_position >= len(owners[candidate.source_task]):
        _fail("invalid_source_binding")
    owner = owners[candidate.source_task][candidate.source_result_position]
    fingerprint = _owner_fingerprint(candidate.source_task, owner)
    if candidate.expected_source_fingerprint is not None and not _valid_sha256(
        candidate.expected_source_fingerprint
    ):
        _fail("invalid_source_fingerprint")
    if fingerprint is not None and not _valid_sha256(fingerprint):
        _fail("invalid_source_fingerprint")
    if candidate.expected_source_fingerprint != fingerprint:
        _fail("source_fingerprint_mismatch")
    if (
        candidate.candidate_family == "model"
        and candidate.source_candidate_key is not None
    ):
        _fail("invalid_source_binding")
    if (
        candidate.candidate_family == "strategy"
        and candidate.source_candidate_key != owner.strategy_key
    ):
        _fail("invalid_source_binding")
    if candidate.candidate_family == "warning_scenario":
        if (
            type(candidate.source_candidate_key) is not str
            or "scenario_key" not in owner.monitoring_summary.columns
        ):
            _fail("invalid_source_binding")
        if (
            int(
                (
                    owner.monitoring_summary["scenario_key"]
                    == candidate.source_candidate_key
                ).sum()
            )
            == 0
        ):
            _fail("source_not_found")
    return owner


def _ref_owner(
    ref: GovernanceEvidenceRef, owners: dict[str, tuple[object, ...]]
) -> tuple[object, int, str | None]:
    _validate_ref_shape(ref)
    registry = {(task, table): position for position, task, table in _SOURCE_REGISTRY}
    position = registry.get((ref.source_task, ref.source_table))
    if position is None:
        _fail("unsupported_source")
    if ref.source_use not in _source_uses(position):
        _fail("unsupported_source")
    collection = owners[ref.source_task]
    if ref.source_result_position >= len(collection):
        _fail("invalid_source_binding")
    owner = collection[ref.source_result_position]
    fingerprint = _owner_fingerprint(ref.source_task, owner)
    if ref.expected_source_fingerprint is not None and not _valid_sha256(
        ref.expected_source_fingerprint
    ):
        _fail("invalid_source_fingerprint")
    if fingerprint is not None and not _valid_sha256(fingerprint):
        _fail("invalid_source_fingerprint")
    if ref.expected_source_fingerprint != fingerprint:
        _fail("source_fingerprint_mismatch")
    table = getattr(owner, ref.source_table, None)
    _validate_owner_table(owner, ref.source_table, table)
    return owner, position, fingerprint


_LOCATORS = {
    "side_key": ("side",),
    "column_key": ("column", "left_column"),
    "scope_key": ("scope", "scope_type", "segment_kind", "scope_key"),
    "scope_column": ("scope_column", "right_column"),
    "scope_position": ("scope_position", "scope_ordinal", "slice_ordinal"),
    "time_position": ("time_slice_ordinal",),
    "fold_id": ("fold_id",),
    "row_position": ("row_position",),
    "entity_position": ("entity_position",),
    "from_row_position": ("from_row_position",),
    "to_row_position": ("to_row_position",),
    "scenario_key": ("scenario_key",),
    "reference_scenario_key": ("reference_scenario_key",),
    "comparator_scenario_key": ("comparator_scenario_key",),
    "rule_key": ("rule_key",),
    "phase_key": ("phase",),
    "action_key": ("action_key", "simulated_action_name"),
    "action_role": ("action_role",),
    "constraint_key": ("constraint_key",),
    "slice_role": ("slice_role",),
    "row_kind": ("row_kind",),
    "state_key": ("state_key",),
    "from_state_key": ("from_state_key",),
    "to_state_key": ("to_state_key",),
    "statistic_key": ("statistic",),
    "category_key": ("resource", "threshold_kind", "historical_action_name"),
    "category_position": ("class_position", "category_position"),
    "pattern_key": ("pattern_key",),
    "episode_ordinal": ("episode_ordinal",),
    "notification_ordinal": ("notification_ordinal",),
    "event_ordinal": ("event_ordinal",),
    "finding_key": ("finding_key",),
    "provenance_key": ("provenance_key",),
}

# Wide owner tables encode the metric identity in the selected value/status
# column rather than in a dedicated ``metric`` column.  These are the only
# approved wide metric families in the Task 15 source contract.
_WIDE_METRICS = {
    ("task15", "gains"): frozenset({"event_rate", "capture", "lift"}),
    ("task15", "threshold_analysis"): frozenset(
        {
            "sensitivity",
            "specificity",
            "precision",
            "negative_predictive_value",
            "f1",
            "accuracy",
            "predicted_positive_rate",
        }
    ),
}

# The compound locator is part of the source contract, not an optional hint.
# Keeping this table private prevents a generic "first matching row" fallback.
_REQUIRED_REF_FIELDS = {
    1: ("scope_key", "fold_id", "metric_key", "statistic_key"),
    2: ("scope_key", "fold_id", "metric_key", "numeric_value"),
    3: ("scope_key", "fold_id", "metric_key", "category_key", "numeric_value"),
    4: ("scope_key", "metric_key", "numeric_value"),
    5: ("fold_id",),
    6: ("side_key",),
    7: ("side_key", "column_key"),
    8: ("side_key", "column_key"),
    9: ("side_key", "column_key"),
    10: ("column_key",),
    11: ("side_key", "scope_key", "column_key"),
    12: ("side_key", "slice_role", "row_kind", "scope_position"),
    13: ("side_key", "category_key"),
    14: ("finding_key",),
    15: ("provenance_key",),
    16: ("row_position",),
    17: ("row_position", "rule_key"),
    18: (
        "scope_key",
        "scope_position",
        "time_position",
        "phase_key",
        "rule_key",
        "metric_key",
    ),
    19: (
        "scope_key",
        "scope_position",
        "time_position",
        "action_key",
        "metric_key",
    ),
    20: (
        "scope_key",
        "scope_position",
        "time_position",
        "action_key",
        "action_role",
        "metric_key",
    ),
    21: ("constraint_key", "metric_key"),
    22: ("category_key", "action_key"),
    23: ("provenance_key",),
    24: ("scenario_key", "scope_key", "scope_position", "metric_key"),
    25: (
        "reference_scenario_key",
        "comparator_scenario_key",
        "scope_key",
        "scope_position",
        "metric_key",
    ),
    26: (
        "scope_key",
        "scope_position",
        "from_state_key",
        "to_state_key",
        "metric_key",
    ),
    27: ("row_position", "scenario_key", "rule_key"),
    28: ("entity_position", "scenario_key", "rule_key", "episode_ordinal"),
    29: ("row_position",),
    30: ("from_row_position", "to_row_position"),
    31: ("provenance_key",),
    32: ("side_key", "category_position"),
    33: ("pattern_key",),
    34: ("column_key",),
    35: ("column_key", "scope_column"),
    36: ("row_position",),
    37: (
        "entity_position",
        "scenario_key",
        "rule_key",
        "episode_ordinal",
        "notification_ordinal",
    ),
    38: ("scenario_key", "entity_position", "event_ordinal"),
}


def _metric_column(table: pd.DataFrame, metric: str) -> str | None:
    """Return the owner column carrying a long-form metric identity."""
    for column in ("metric", "metric_key"):
        if column in table.columns:
            return column
    return None


_FIELD_STATUS_COLUMNS = {
    ("dataset_profile", "n_rows"): ("feature_status", "feature_reason"),
    ("column_profile", "non_missing_count"): ("missing_status", "missing_reason"),
    ("numeric_profile", "mean"): ("location_status", "location_reason"),
    ("categorical_profile", "unique_count"): (
        "cardinality_status",
        "cardinality_reason",
    ),
    ("missingness_drift", "absolute_rate_change"): (
        "absolute_change_status",
        "absolute_change_reason",
    ),
    ("slice_profile", "row_count"): ("size_status", "size_reason"),
    ("target_profile", "count"): ("class_status", "class_reason"),
    ("missingness_patterns", "row_count"): ("count_status", "count_reason"),
    ("row_decisions", "final_action_name"): (
        "decision_status",
        "decision_reason",
    ),
    ("observation_history", "active_rule_count"): (
        "observation_status",
        "observation_reason",
    ),
}


def _status_reason_columns(
    table: pd.DataFrame,
    metric: str | None,
    field_key: str | None,
    table_name: str,
) -> tuple[str | None, str | None]:
    """Resolve the frozen status/reason pair without inventing owner fields."""
    candidates = []
    if metric is not None:
        candidates.append((f"{metric}_status", f"{metric}_reason"))
    if field_key is not None and (table_name, field_key) in _FIELD_STATUS_COLUMNS:
        candidates.append(_FIELD_STATUS_COLUMNS[(table_name, field_key)])
    candidates.extend(
        (
            ("status", "reason"),
            ("decision_status", "decision_reason"),
            ("observation_status", "observation_reason"),
            ("state_status", "state_reason"),
            ("feature_status", "feature_reason"),
            ("value_profile_status", "value_profile_reason"),
            ("count_status", "count_reason"),
        )
    )
    for status, reason in candidates:
        if status in table.columns or reason in table.columns:
            return (
                status if status in table.columns else None,
                reason if reason in table.columns else None,
            )
    return None, None


def _value_column(ref: GovernanceEvidenceRef, table: pd.DataFrame) -> str | None:
    """Select the exact approved value cell for a resolved owner row."""
    metric = ref.metric_key
    if metric is not None:
        if "value" in table.columns:
            return "value"
        if "metric_value" in table.columns:
            return "metric_value"
        if (
            metric in table.columns
            and (
                ref.source_task,
                ref.source_table,
            )
            in _WIDE_METRICS
        ):
            return metric
    elif "value" in table.columns:
        return "value"
    elif "metric_value" in table.columns:
        return "metric_value"
    if ref.field_key is not None and ref.field_key in table.columns:
        return ref.field_key
    return None


def _resolve_ref(
    ref: GovernanceEvidenceRef, owners: dict[str, tuple[object, ...]], ordinal: int
) -> dict[str, object]:
    owner, registry_position, fingerprint = _ref_owner(ref, owners)
    if any(
        getattr(ref, field_name) is None
        for field_name in _REQUIRED_REF_FIELDS[registry_position]
    ):
        _fail("invalid_source_locator")
    table = getattr(owner, ref.source_table)
    mask = pd.Series(True, index=table.index)
    used = 0
    for field_name, columns in _LOCATORS.items():
        value = getattr(ref, field_name)
        if value is None:
            continue
        column = next((c for c in columns if c in table.columns), None)
        if column is None:
            _fail("invalid_source_locator")
        mask &= table[column].eq(value)
        used += 1
    if ref.numeric_value is not None:
        column = (
            "requested_fraction"
            if "requested_fraction" in table.columns
            else "threshold"
            if "threshold" in table.columns
            else "segment_value"
            if "segment_value" in table.columns
            else None
        )
        if column is None:
            _fail("invalid_source_locator")
        mask &= table[column].eq(ref.numeric_value)
        used += 1
    if ref.metric_key is not None:
        metric_column = _metric_column(table, ref.metric_key)
        if metric_column is not None:
            mask &= table[metric_column].eq(ref.metric_key)
            used += 1
        elif ref.metric_key not in _WIDE_METRICS.get(
            (ref.source_task, ref.source_table), frozenset()
        ):
            _fail("invalid_source_locator")
    if used == 0:
        _fail("invalid_source_locator")
    matched = table.loc[mask]
    if len(matched) == 0:
        _fail("source_not_found")
    if len(matched) != 1:
        _fail("source_not_unique")
    row = matched.iloc[0]
    metric = ref.metric_key
    metric_column = _metric_column(table, metric) if metric is not None else None
    if metric is not None and metric_column is not None:
        if row[metric_column] != metric:
            _fail("source_not_found")
    value_column = _value_column(ref, table)
    if value_column is None:
        _fail("invalid_source_locator")
    status_column, reason_column = _status_reason_columns(
        table, metric, ref.field_key, ref.source_table
    )
    status = row[status_column] if status_column is not None else "available"
    reason = row[reason_column] if reason_column is not None else pd.NA
    support_column = next(
        (
            x
            for x in (
                "support_n",
                "support_n_rows",
                "n_rows",
                "n_evaluable_rows",
                "selected_n",
                "valid_n",
                "row_count",
            )
            if x in table.columns
        ),
        None,
    )
    support = row[support_column] if support_column else pd.NA
    unit = (
        row["support_unit"]
        if "support_unit" in table.columns
        else row["unit"]
        if "unit" in table.columns
        else "rows"
    )
    value = row[value_column]
    # The selected value is an owner-domain scalar, not an arbitrary pandas
    # object.  Missing values are allowed only for an owner row whose status
    # explicitly says that the evidence is unavailable/undefined/not
    # applicable/not verifiable; an available non-finite or non-numeric value
    # is an owner-value violation before any Task 19 materialization.
    if status in {"available", "computed", "evaluated", "active", "resolved"}:
        if value is pd.NA or (
            pd.api.types.is_numeric_dtype(table[value_column].dtype)
            and bool(pd.isna(value))
        ):
            _fail("invalid_owner_value")
        if isinstance(value, (float, int, np.floating, np.integer)) and not isinstance(
            value, (bool, np.bool_)
        ):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError, OverflowError):
                _fail("invalid_owner_value")
            if not math.isfinite(numeric_value):
                _fail("invalid_owner_value")
    return {
        "ordinal": ordinal,
        "registry_position": registry_position,
        "fingerprint": fingerprint,
        "status": status,
        "reason": reason,
        "value": value,
        "support": support,
        "unit": unit,
        "row": row,
    }


def _owner_time(
    task: str, owner: object, ref: GovernanceEvidenceRef
) -> datetime | pd.Timestamp | None:
    if task == "task16":
        return None
    if task == "task17" or task == "task18":
        key = "evaluation_time" if task == "task17" else "analysis_as_of"
        table = owner.provenance
        if (
            "provenance_key" not in table.columns
            or "provenance_value" not in table.columns
        ):
            _fail("authoritative_time_missing")
        rows = table.loc[table["provenance_key"].eq(key), "provenance_value"]
        if len(rows) != 1:
            _fail("authoritative_time_missing")
        raw = rows.iloc[0]
        try:
            if type(raw) is str and raw.startswith("{"):
                node = json.loads(raw)
                if node.get("t") in ("datetime", "timestamp"):
                    raw = node.get("v")
                elif type(node.get("__datetime__")) is str:
                    # Task 17/18 owner provenance uses its frozen
                    # canonical-datetime envelope.
                    raw = node["__datetime__"]
                else:
                    _fail("authoritative_time_mismatch")
            return pd.Timestamp(raw).to_pydatetime()
        except (TypeError, ValueError):
            _fail("authoritative_time_mismatch")
    # Task15 absolute time is fold- or observed-loss-owned.
    if (
        task == "task15"
        and getattr(owner, "validation_mode", None) in ("time_holdout", "time_forward")
        and ref.source_table == "folds"
    ):
        if ref.fold_id is None:
            _fail("authoritative_time_missing")
        folds = owner.folds
        rows = folds.loc[folds["fold_id"].eq(ref.fold_id)]
        if (
            len(rows) != 1
            or "analysis_as_of" not in rows.columns
            or pd.isna(rows.iloc[0]["analysis_as_of"])
        ):
            _fail("authoritative_time_missing")
    if (
        ref.source_table == "business_metrics"
        and getattr(owner, "observed_loss_analysis_as_of", None) is not None
    ):
        return owner.observed_loss_analysis_as_of
    folds = owner.folds
    if ref.fold_id is not None and "fold_id" in folds.columns:
        rows = folds.loc[folds["fold_id"].eq(ref.fold_id)]
        for column in ("analysis_as_of", "fold_cutoff"):
            if (
                len(rows) == 1
                and column in rows.columns
                and not pd.isna(rows.iloc[0][column])
            ):
                return rows.iloc[0][column]
    return None


def _whole_input_owner_times(
    task: str, owner: object
) -> tuple[datetime | pd.Timestamp, ...]:
    """Collect every contract-authoritative owner timestamp before resources.

    This is deliberately narrower than a generic datetime scan: Task 15 uses
    its frozen time-fold and observed-loss provenance, Task 17 has only its
    evaluation provenance, and Task 18 uses its owner as-of plus the declared
    datetime columns in its public result schemas.  The caller has already
    completed owner schema/dtype/status/reason/value validation before this
    helper runs, so malformed owner structure retains precedence.
    """

    values: list[datetime | pd.Timestamp] = []

    if task == "task16":
        # Task 16 deliberately has no result-level authoritative absolute time.
        return ()

    if task == "task15":
        mode = getattr(owner, "validation_mode", None)
        folds = getattr(owner, "folds", None)
        if mode in ("time_holdout", "time_forward"):
            if type(folds) is not pd.DataFrame or "analysis_as_of" not in folds:
                _fail("authoritative_time_missing")
            # All frozen fold chronology columns are evidence-bearing.  The
            # analysis_as_of value is required for every fold; the remaining
            # columns may be nullable only where the owner contract permits.
            for column in ("cutoff", "validation_start", "validation_end"):
                if column not in folds:
                    _fail("authoritative_time_missing")
                for raw in folds[column].tolist():
                    if pd.isna(raw):
                        continue
                    if not _is_datetime(raw):
                        _fail("authoritative_time_mismatch")
                    values.append(raw)
            for raw in folds["analysis_as_of"].tolist():
                if pd.isna(raw):
                    _fail("authoritative_time_missing")
                if not _is_datetime(raw):
                    _fail("authoritative_time_mismatch")
                values.append(raw)
        observed_loss = getattr(owner, "observed_loss_analysis_as_of", None)
        if observed_loss is not None:
            if not _is_datetime(observed_loss):
                _fail("authoritative_time_mismatch")
            values.append(observed_loss)
        return tuple(values)

    # Task 17/18 owner provenance is the sole absolute owner clock.  Reuse
    # the existing parser so canonical JSON, missing and malformed values keep
    # the established exact errors.
    provenance_ref = GovernanceEvidenceRef(
        task,
        0,
        "provenance",
        "diagnostic",
        None,
        _owner_fingerprint(task, owner),
        provenance_key="evaluation_time" if task == "task17" else "analysis_as_of",
    )
    owner_as_of = _owner_time(task, owner, provenance_ref)
    if owner_as_of is None:
        _fail("authoritative_time_missing")
    values.append(owner_as_of)

    if task == "task18":
        # Only columns explicitly declared datetime in the frozen owner table
        # schemas are scanned; arbitrary datetime-looking payloads are not.
        for table_name, schema in _TASK18_TABLE_SCHEMAS.items():
            table = getattr(owner, table_name)
            for column, dtype in schema:
                if not dtype.startswith("datetime"):
                    continue
                for raw in table[column].tolist():
                    if pd.isna(raw):
                        continue
                    if not _is_datetime(raw):
                        _fail("authoritative_time_mismatch")
                    values.append(raw)
    return tuple(values)


def _metric(values: tuple[float, ...], targets: tuple[bool, ...], kind: str) -> float:
    y = np.asarray(targets, dtype="int64")
    scores = np.asarray(values, dtype="float64")
    if kind == "brier_score":
        return float(np.mean((scores - y) ** 2))
    positives = scores[y == 1]
    negatives = scores[y == 0]
    if len(positives) == 0 or len(negatives) == 0:
        raise ZeroDivisionError
    return float(
        np.mean(
            (positives[:, None] > negatives).astype(float)
            + 0.5 * (positives[:, None] == negatives)
        )
    )


def _bootstrap_metric(
    evidence: GovernancePerformanceEvidence, kind: str
) -> tuple[float | None, float | None, str | None]:
    values = (
        evidence.ranking_scores if kind == "roc_auc" else evidence.event_probabilities
    )
    assert values is not None
    try:
        value = _metric(values, evidence.target_values, kind)
    except ZeroDivisionError:
        return None, None, "single_class"
    rng = np.random.default_rng(evidence.random_state)
    samples: list[float] = []
    n = len(values)
    for _ in range(evidence.bootstrap_repeats):
        index = rng.integers(0, n, size=n)
        try:
            samples.append(
                _metric(
                    tuple(values[i] for i in index),
                    tuple(evidence.target_values[i] for i in index),
                    kind,
                )
            )
        except ZeroDivisionError:
            continue
    if len(samples) < 2:
        return None, None, "insufficient_bootstrap_support"
    return value, float(np.std(samples, ddof=0)), None


def _validate_performance_chronology(
    performance_evidence: tuple[GovernancePerformanceEvidence, ...],
    ref_by_id: dict[int, dict[str, object]],
) -> None:
    """Validate performance windows and reference/current ordering early.

    This gate is intentionally before resource preflight and all bootstrap or
    stability materialization.  Group identity matches the later stability
    projection, while duplicate/missing snapshot cardinality remains owned by
    that projection's existing validation branch.
    """

    groups: dict[tuple[object, ...], list[GovernancePerformanceEvidence]] = {}
    for item in performance_evidence:
        if not (
            _normal_time(item.window_start)
            < _normal_time(item.window_end)
            <= _normal_time(item.evidence_as_of)
        ):
            _fail("invalid_performance_evidence")
        metric_kind = "roc_auc" if item.ranking_scores is not None else "brier_score"
        source_fp = ref_by_id[id(item.source_ref)]["fingerprint"]
        key = (
            item.candidate_key,
            item.evaluation_scope,
            item.scope_key,
            item.scope_position,
            metric_kind,
            source_fp,
        )
        groups.setdefault(key, []).append(item)
    for values in groups.values():
        references = [item for item in values if item.snapshot_role == "reference"]
        currents = [item for item in values if item.snapshot_role == "current"]
        if len(references) == 1 and len(currents) == 1:
            if _normal_time(references[0].evidence_as_of) > _normal_time(
                currents[0].evidence_as_of
            ):
                _fail("invalid_performance_evidence")
            if _normal_time(references[0].window_end) > _normal_time(
                currents[0].window_start
            ):
                _fail("invalid_performance_evidence")


def evaluate_governance(
    policy: GovernancePolicy,
    *,
    risk_validations: tuple[BinaryRiskValidationResult, ...] = (),
    data_audits: tuple[DataAuditResult, ...] = (),
    decision_strategies: tuple[DecisionStrategyResult, ...] = (),
    lifecycle_monitorings: tuple[LifecycleMonitoringResult, ...] = (),
    model_attributions: tuple[GovernanceAttributionEvidence, ...] = (),
    prediction_profiles: tuple[GovernancePredictionProfile, ...] = (),
    performance_evidence: tuple[GovernancePerformanceEvidence, ...] = (),
) -> GovernanceResult:
    """Validate and materialize bounded offline governance evidence.

    The evaluator consumes only frozen owner results and typed declarations.
    Missing evidence is preserved as status/reason data; invalid declarations
    raise ``ValueError``. Inputs are never modified and no external action is
    performed.
    """
    if type(policy) is not GovernancePolicy:
        _fail("invalid_policy_type")
    if not _is_safe_key(policy.governance_key):
        _fail("invalid_governance_key")
    if not _is_safe_key(policy.governance_version):
        _fail("invalid_governance_version")
    if not _is_datetime(policy.analysis_as_of):
        _fail("invalid_analysis_as_of")
    for value, expected, key in (
        (policy.candidates, GovernanceCandidate, "invalid_candidate_container"),
        (policy.criteria, GovernanceCriterion, "invalid_criterion_container"),
        (policy.metadata, GovernanceMetadata, "invalid_metadata_container"),
        (policy.evidence_refs, GovernanceEvidenceRef, "invalid_evidence_ref_container"),
        (policy.explanations, GovernanceExplanation, "invalid_explanation_container"),
        (
            model_attributions,
            GovernanceAttributionEvidence,
            "invalid_attribution_container",
        ),
        (
            prediction_profiles,
            GovernancePredictionProfile,
            "invalid_prediction_profile_container",
        ),
        (
            performance_evidence,
            GovernancePerformanceEvidence,
            "invalid_performance_evidence_container",
        ),
    ):
        _validate_container(
            value,
            expected,
            key,
            duplicate_key=(
                "duplicate_evidence_ref"
                if key == "invalid_evidence_ref_container"
                else "duplicate_owner_source"
            ),
        )
    if type(policy.comparison_pairs) is not tuple or any(
        type(x) is not tuple or len(x) != 2 or any(type(y) is not str for y in x)
        for x in policy.comparison_pairs
    ):
        _fail("invalid_pair_container")
    _validate_container(
        risk_validations, BinaryRiskValidationResult, "invalid_owner_result_container"
    )
    _validate_container(data_audits, DataAuditResult, "invalid_owner_result_container")
    _validate_container(
        decision_strategies, DecisionStrategyResult, "invalid_owner_result_container"
    )
    _validate_container(
        lifecycle_monitorings,
        LifecycleMonitoringResult,
        "invalid_owner_result_container",
    )
    if policy.human_review_mode not in ("promotion_only", "all_recommendations"):
        _fail("invalid_human_review_mode")
    if policy.entity_alignment not in ("not_requested", "owner_verified"):
        _fail("invalid_entity_alignment")

    candidates = list(policy.candidates)
    for candidate in candidates:
        if (
            not _is_safe_key(candidate.candidate_key)
            or type(candidate.evidence_refs) is not tuple
        ):
            _fail("invalid_candidate")
    keys = [c.candidate_key for c in candidates]
    if len(set(keys)) != len(keys):
        _fail("duplicate_candidate")
    champions = [c for c in candidates if c.declared_role == "champion"]
    if len(champions) != 1 or champions[0].declared_state != "approved":
        _fail("invalid_champion")
    champion = champions[0]
    challengers = [c for c in candidates if c.declared_role == "challenger"]
    if any(
        c.declared_state
        not in ("candidate", "under_review", "approved", "rejected", "retired")
        for c in challengers
    ):
        _fail("invalid_candidate")
    if len(set(policy.comparison_pairs)) != len(policy.comparison_pairs):
        _fail("duplicate_pair")
    expected_pairs = {(champion.candidate_key, c.candidate_key) for c in challengers}
    if set(policy.comparison_pairs) != expected_pairs:
        _fail("invalid_pair_coverage")
    candidate_by_key = {candidate.candidate_key: candidate for candidate in candidates}
    if any(
        candidate_by_key[left].candidate_family
        != candidate_by_key[right].candidate_family
        for left, right in policy.comparison_pairs
    ):
        _fail("invalid_pair")
    if len({c.criterion_key for c in policy.criteria}) != len(policy.criteria):
        _fail("duplicate_criterion")

    # Structured declarations and carrier tags are validated before owner
    # table/schema inspection so malformed caller values cannot be masked by a
    # later owner or resource failure.
    for item in policy.explanations:
        _validate_explanation_declaration(item, candidate_by_key)
    for item in model_attributions:
        _validate_attribution_declaration(item, candidate_by_key)
    for item in prediction_profiles:
        _validate_prediction_declaration(item, candidate_by_key)
    for item in performance_evidence:
        _validate_performance_declaration(item, candidate_by_key)
    for ref in policy.evidence_refs:
        if ref.candidate_key is not None and ref.candidate_key not in candidate_by_key:
            _fail("invalid_source_binding")
        if ref.source_use not in ("diagnostic", "explanation"):
            _fail("invalid_evidence_ref")
    for candidate in candidates:
        for ref in candidate.evidence_refs:
            if ref.candidate_key != candidate.candidate_key:
                _fail("invalid_source_binding")
            if ref.source_use != "comparison_criterion":
                _fail("invalid_evidence_ref")

    owners = _owner_collections(
        risk_validations, data_audits, decision_strategies, lifecycle_monitorings
    )
    for collection in owners.values():
        for owner in collection:
            for table_name, table in vars(owner).items():
                if type(table) is pd.DataFrame:
                    _validate_owner_table(owner, table_name, table)
    candidate_owners: dict[str, object] = {}
    semantic_sources: list[tuple[object, ...]] = []
    for candidate in candidates:
        owner = _candidate_owner(candidate, owners)
        candidate_owners[candidate.candidate_key] = owner
        identity = (
            candidate.source_task,
            candidate.source_result_position,
            candidate.source_candidate_key,
        )
        if identity in semantic_sources:
            _fail("duplicate_candidate")
        semantic_sources.append(identity)

    if type(policy.minimum_comparable_criteria) is not int or not (
        1 <= policy.minimum_comparable_criteria <= max(1, len(policy.criteria))
    ):
        _fail("invalid_criterion")

    seen_metadata_identity: set[tuple[str, str | None]] = set()
    for item in policy.metadata:
        if (
            not _is_safe_key(item.metadata_key)
            or item.metadata_scope not in ("governance", "candidate")
            or (item.metadata_scope == "governance" and item.candidate_key is not None)
            or (item.metadata_scope == "candidate" and item.candidate_key not in keys)
            or not _is_safe_key(item.purpose_key)
            or not _is_safe_key(item.owner_key)
            or item.materiality not in ("low", "medium", "high")
            or type(item.assumption_keys) is not tuple
            or type(item.limitation_keys) is not tuple
            or type(item.monitoring_thresholds) is not tuple
            or any(
                not _is_safe_key(x) for x in item.assumption_keys + item.limitation_keys
            )
            or any(
                type(x) is not tuple
                or len(x) != 2
                or not _is_safe_key(x[0])
                or not _exact_float(x[1])
                for x in item.monitoring_thresholds
            )
            or item.issue_status not in ("none", "open", "monitoring", "resolved")
            or item.remediation_status
            not in ("not_required", "planned", "in_progress", "complete")
        ):
            _fail("invalid_metadata")
        identity = (item.metadata_scope, item.candidate_key)
        if identity in seen_metadata_identity:
            _fail("invalid_metadata")
        seen_metadata_identity.add(identity)

    direction = {(a, b, c): d for a, b, c, d in _DIRECTION_REGISTRY}
    for criterion in policy.criteria:
        if (
            not _is_safe_key(criterion.criterion_key)
            or criterion.candidate_family != champion.candidate_family
        ):
            _fail("invalid_criterion")
        if (
            type(criterion.required_for_promotion) is not bool
            or not _exact_int(criterion.minimum_support, minimum=1)
            or criterion.minimum_support > 200000
        ):
            _fail("invalid_criterion")
        expected_direction = direction.get(
            (criterion.source_task, criterion.source_table, criterion.metric_key)
        )
        if criterion.criterion_role == "diagnostic":
            if (
                criterion.direction != "not_directional"
                or criterion.required_for_promotion
            ):
                _fail("invalid_criterion")
        elif expected_direction is None:
            _fail("unsupported_criterion")
        elif criterion.direction != expected_direction:
            _fail("invalid_criterion")
        if criterion.direction == "target_range":
            if (
                not _exact_float(criterion.target_low)
                or not _exact_float(criterion.target_high)
                or criterion.target_low > criterion.target_high
            ):
                _fail("invalid_criterion")
        elif criterion.target_low is not None or criterion.target_high is not None:
            _fail("invalid_criterion")

        registry_position = next(
            (
                position
                for position, task, table in _SOURCE_REGISTRY
                if task == criterion.source_task and table == criterion.source_table
            ),
            None,
        )
        if (
            registry_position is None
            or registry_position
            not in _COMPARISON_SOURCE_FAMILIES.get(
                champion.candidate_family, frozenset()
            )
        ) and policy.comparison_pairs:
            _fail("invalid_criterion")

    # Flatten refs in the frozen occurrence order and resolve once each.
    for carrier, standalone_refs in (
        ("policy", policy.evidence_refs),
        *(
            (candidate.candidate_key, candidate.evidence_refs)
            for candidate in candidates
        ),
    ):
        seen_standalone: set[str] = set()
        for ref in standalone_refs:
            identity = _canonical_json(ref)
            if identity in seen_standalone:
                _fail("duplicate_evidence_ref")
            seen_standalone.add(identity)
    refs: list[GovernanceEvidenceRef] = list(policy.evidence_refs)
    for candidate in candidates:
        refs.extend(candidate.evidence_refs)
    refs.extend(x.source_ref for x in policy.explanations)
    refs.extend(x.source_ref for x in model_attributions)
    refs.extend(x.source_ref for x in prediction_profiles)
    refs.extend(x.source_ref for x in performance_evidence)
    resolved = [_resolve_ref(ref, owners, i) for i, ref in enumerate(refs)]
    ref_by_id = {id(ref): info for ref, info in zip(refs, resolved, strict=True)}

    # Authoritative time is checked after source/schema/fingerprint resolution.
    times: list[datetime | pd.Timestamp] = [policy.analysis_as_of]
    time_statuses: list[str] = []
    for ref in refs:
        owner = owners[ref.source_task][ref.source_result_position]
        value = _owner_time(ref.source_task, owner, ref)
        if value is None:
            time_statuses.append("unverified")
        else:
            if not _is_datetime(value):
                _fail("authoritative_time_mismatch")
            times.append(value)
            time_statuses.append("verified")
    # Whole-input safety is intentionally performed after owner structural
    # validation and source resolution, but before resource preflight and any
    # public materialization.  This includes inactive, diagnostic, unpaired and
    # unreferenced owner rows without inventing a second schema/value validator.
    for task, collection in owners.items():
        for owner in collection:
            times.extend(_whole_input_owner_times(task, owner))
    for declaration in model_attributions:
        if not _is_datetime(declaration.evidence_as_of):
            _fail("invalid_attribution")
        times.append(declaration.evidence_as_of)
    for declaration in prediction_profiles:
        if not _is_datetime(declaration.analysis_as_of):
            _fail("invalid_prediction_profile")
        times.append(declaration.analysis_as_of)
    for declaration in performance_evidence:
        if not all(
            _is_datetime(x)
            for x in (
                declaration.window_start,
                declaration.window_end,
                declaration.evidence_as_of,
            )
        ):
            _fail("invalid_performance_evidence")
        times.extend(
            (
                declaration.window_start,
                declaration.window_end,
                declaration.evidence_as_of,
            )
        )
    awareness = {_aware(x) for x in times}
    if len(awareness) != 1:
        _fail("datetime_awareness_mismatch")
    normalized = [_normal_time(x) for x in times]
    governance_time = normalized[0]
    if any(x > governance_time for x in normalized[1:]):
        _fail("future_evidence_time")
    aware = _aware(policy.analysis_as_of)

    _validate_performance_chronology(performance_evidence, ref_by_id)

    # Resource preflight, strictly in the contract order.
    primitive_resource_projection = (
        len(risk_validations),
        len(data_audits),
        len(decision_strategies),
        len(lifecycle_monitorings),
        len(candidates),
        len(policy.comparison_pairs),
        len(policy.criteria),
        len(policy.explanations),
        len(model_attributions),
        max((x.permutation_repeats or 0 for x in model_attributions), default=0),
        len(prediction_profiles),
        len(performance_evidence),
        sum(
            5
            + len(x.assumption_keys)
            + len(x.limitation_keys)
            + len(x.monitoring_thresholds)
            for x in policy.metadata
        ),
        len(refs),
        sum(len(x.target_values) for x in performance_evidence),
    )
    _resource_preflight(primitive_resource_projection + (0, 0))

    candidate_positions = {c.candidate_key: i for i, c in enumerate(candidates)}

    explanation_rows: list[dict[str, object]] = []
    for position, item in enumerate(policy.explanations):
        if (
            item.candidate_key not in candidate_positions
            or item.source_ref.source_use != "explanation"
        ):
            _fail("invalid_explanation")
        info = ref_by_id[id(item.source_ref)]
        explanation_rows.append(
            {
                "explanation_position": position,
                "explanation_key": item.explanation_key,
                "candidate_position": candidate_positions[item.candidate_key],
                "candidate_family": candidates[
                    candidate_positions[item.candidate_key]
                ].candidate_family,
                "method": item.method,
                "source_ref_position": info["ordinal"],
                "source_task": item.source_ref.source_task,
                "source_result_position": item.source_ref.source_result_position,
                "source_table": item.source_ref.source_table,
                "source_registry_position": info["registry_position"],
                "source_fingerprint": info["fingerprint"],
                "feature_key": item.feature_key,
                "relation": item.relation,
                "priority": item.priority,
                "evidence_time_status": time_statuses[info["ordinal"]],
                "source_status": info["status"],
                "source_reason": info["reason"],
                "status": item.status,
                "reason": item.reason,
                "finding_key": f"governance:explanation:{position}",
            }
        )

    attribution_rows: list[dict[str, object]] = []
    method_order = {
        "coefficient_direction": 0,
        "native_importance": 1,
        "permutation_importance": 2,
    }
    attribution_order = sorted(
        enumerate(model_attributions),
        key=lambda x: (
            candidate_positions.get(x[1].candidate_key, 10**9),
            method_order.get(x[1].method, 9),
            x[0],
        ),
    )
    seen_attributions: set[tuple[object, ...]] = set()
    for declaration_position, item in attribution_order:
        if (
            item.candidate_key not in candidate_positions
            or item.source_ref.source_use != "attribution_context"
            or not _is_safe_key(item.feature_key)
            or not _exact_float(item.value)
            or not _exact_int(item.support_n)
        ):
            _fail("invalid_attribution")
        identity = (
            item.candidate_key,
            item.method,
            item.feature_key,
            item.evaluation_scope,
            item.metric_key,
        )
        if identity in seen_attributions:
            _fail("invalid_attribution")
        seen_attributions.add(identity)
        if item.method == "coefficient_direction":
            relation = (
                "positive"
                if item.value > 0
                else "negative"
                if item.value < 0
                else "neutral"
            )
            if (
                item.relation != relation
                or item.metric_key is not None
                or item.evaluation_scope != "not_applicable"
                or any(
                    x is not None
                    for x in (
                        item.uncertainty_std,
                        item.permutation_repeats,
                        item.random_state,
                    )
                )
            ):
                _fail("invalid_attribution")
        elif item.method == "native_importance":
            if (
                item.value < 0
                or item.relation != "not_directional"
                or item.metric_key is not None
                or item.evaluation_scope != "not_applicable"
                or any(
                    x is not None
                    for x in (
                        item.uncertainty_std,
                        item.permutation_repeats,
                        item.random_state,
                    )
                )
            ):
                _fail("invalid_attribution")
        elif not (
            _exact_float(item.uncertainty_std)
            and _exact_int(item.permutation_repeats, minimum=1)
            and _exact_int(item.random_state)
            and item.evaluation_scope in ("holdout", "oof")
            and type(item.metric_key) is str
        ):
            _fail("invalid_attribution")
        info = ref_by_id[id(item.source_ref)]
        cp = candidate_positions[item.candidate_key]
        attribution_rows.append(
            {
                "candidate_position": cp,
                "attribution_position": declaration_position,
                "candidate_family": candidates[cp].candidate_family,
                "method": item.method,
                "feature_key": item.feature_key,
                "metric_key": item.metric_key,
                "value": item.value,
                "relation": item.relation,
                "evaluation_scope": item.evaluation_scope,
                "support_n": item.support_n,
                "uncertainty_std": item.uncertainty_std,
                "permutation_repeats": item.permutation_repeats,
                "random_state": item.random_state,
                "evidence_as_of": _normal_time(item.evidence_as_of),
                "evidence_time_status": "verified",
                "source_task": item.source_ref.source_task,
                "source_result_position": item.source_ref.source_result_position,
                "source_table": item.source_ref.source_table,
                "source_ref_position": info["ordinal"],
                "source_fingerprint": info["fingerprint"],
                "source_status": info["status"],
                "source_reason": info["reason"],
                "status": "available",
                "reason": pd.NA,
                "finding_key": f"governance:attribution:{cp}:{declaration_position}",
            }
        )

    drift_rows: list[dict[str, object]] = []
    profile_groups: dict[
        tuple[object, ...], list[tuple[int, GovernancePredictionProfile]]
    ] = {}
    for position, item in enumerate(prediction_profiles):
        if (
            item.candidate_key not in candidate_positions
            or item.source_ref.source_use != "drift_context"
            or len(item.bin_boundaries) != 9
            or len(item.bin_counts) != 10
        ):
            _fail("invalid_prediction_profile")
        if any(not _exact_float(x) for x in item.bin_boundaries) or any(
            a >= b for a, b in zip(item.bin_boundaries, item.bin_boundaries[1:])
        ):
            _fail("invalid_prediction_profile")
        if item.prediction_kind == "event_probability" and item.bin_boundaries != tuple(
            i / 10 for i in range(1, 10)
        ):
            _fail("invalid_prediction_profile")
        if (
            any(not _exact_int(x) for x in item.bin_counts)
            or sum(item.bin_counts) != item.support_n
            or not _exact_int(item.missing_n)
            or not 2 <= item.bootstrap_repeats <= 1000
            or not _exact_int(item.random_state)
        ):
            _fail("invalid_prediction_profile")
        key = (
            item.candidate_key,
            item.prediction_kind,
            item.scope_key,
            item.scope_position,
            item.reference_state_fingerprint,
        )
        profile_groups.setdefault(key, []).append((position, item))
    drift_draws = 0
    for values in profile_groups.values():
        references = [x for x in values if x[1].snapshot_role == "reference"]
        currents = [x for x in values if x[1].snapshot_role == "current"]
        if len(references) != 1 or len(currents) != 1:
            _fail("invalid_prediction_profile")
        rp, reference = references[0]
        cp_, current = currents[0]
        if (
            reference.bin_boundaries != current.bin_boundaries
            or reference.bootstrap_repeats != current.bootstrap_repeats
            or reference.random_state != current.random_state
            or _normal_time(reference.analysis_as_of)
            > _normal_time(current.analysis_as_of)
        ):
            _fail("invalid_prediction_profile")
        drift_draws += reference.bootstrap_repeats
        status, reason, tvd, std = "available", pd.NA, None, None
        if reference.support_n == 0 or current.support_n == 0:
            status, reason = "undefined", "insufficient_support"
        else:
            p = np.asarray(reference.bin_counts, dtype="float64") / reference.support_n
            q = np.asarray(current.bin_counts, dtype="float64") / current.support_n
            tvd = float(0.5 * np.abs(q - p).sum())
            rng = np.random.default_rng(reference.random_state)
            samples = [
                0.5
                * np.abs(
                    rng.multinomial(reference.support_n, p) / reference.support_n
                    - rng.multinomial(current.support_n, q) / current.support_n
                ).sum()
                for _ in range(reference.bootstrap_repeats)
            ]
            std = float(np.std(samples, ddof=0))
        cpos = candidate_positions[reference.candidate_key]
        drift_rows.append(
            {
                "candidate_position": cpos,
                "reference_profile_position": rp,
                "current_profile_position": cp_,
                "prediction_kind": reference.prediction_kind,
                "scope_key": reference.scope_key,
                "scope_position": reference.scope_position,
                "reference_snapshot_key": reference.snapshot_key,
                "current_snapshot_key": current.snapshot_key,
                "reference_analysis_as_of": _normal_time(reference.analysis_as_of),
                "current_analysis_as_of": _normal_time(current.analysis_as_of),
                "reference_time_status": "verified",
                "current_time_status": "verified",
                "reference_support_n": reference.support_n,
                "current_support_n": current.support_n,
                "reference_missing_n": reference.missing_n,
                "current_missing_n": current.missing_n,
                "bin_count": 10,
                "reference_state_fingerprint": reference.reference_state_fingerprint,
                "reference_source_fingerprint": ref_by_id[id(reference.source_ref)][
                    "fingerprint"
                ],
                "current_source_fingerprint": ref_by_id[id(current.source_ref)][
                    "fingerprint"
                ],
                "metric": "prediction_tvd",
                "prediction_tvd": tvd,
                "direction": "lower_is_better",
                "uncertainty_std": std,
                "bootstrap_repeats": reference.bootstrap_repeats,
                "random_state": reference.random_state,
                "status": status,
                "reason": reason,
                "finding_key": f"governance:drift:{cpos}:{rp}:{cp_}",
            }
        )
    _resource_preflight(primitive_resource_projection + (drift_draws, 0))

    stability_rows: list[dict[str, object]] = []
    performance_groups: dict[
        tuple[object, ...], list[tuple[int, GovernancePerformanceEvidence]]
    ] = {}
    performance_draws = 0
    for position, item in enumerate(performance_evidence):
        if (
            item.candidate_key not in candidate_positions
            or item.source_ref.source_use != "stability_context"
            or type(item.target_values) is not tuple
            or any(type(x) is not bool for x in item.target_values)
        ):
            _fail("invalid_performance_evidence")
        kinds = int(item.ranking_scores is not None) + int(
            item.event_probabilities is not None
        )
        if (
            kinds != 1
            or not 2 <= item.bootstrap_repeats <= 1000
            or not _exact_int(item.random_state)
            or not (
                _normal_time(item.window_start)
                < _normal_time(item.window_end)
                <= _normal_time(item.evidence_as_of)
            )
        ):
            _fail("invalid_performance_evidence")
        vector = (
            item.ranking_scores
            if item.ranking_scores is not None
            else item.event_probabilities
        )
        assert vector is not None
        if (
            type(vector) is not tuple
            or len(vector) != len(item.target_values)
            or any(not _exact_float(x) for x in vector)
            or (
                item.event_probabilities is not None
                and any(not 0 <= x <= 1 for x in vector)
            )
        ):
            _fail("invalid_performance_evidence")
        metric_kind = "roc_auc" if item.ranking_scores is not None else "brier_score"
        source_fp = ref_by_id[id(item.source_ref)]["fingerprint"]
        key = (
            item.candidate_key,
            item.evaluation_scope,
            item.scope_key,
            item.scope_position,
            metric_kind,
            source_fp,
        )
        performance_groups.setdefault(key, []).append((position, item))
    for values in performance_groups.values():
        references = [x for x in values if x[1].snapshot_role == "reference"]
        currents = [x for x in values if x[1].snapshot_role == "current"]
        if len(references) != 1 or len(currents) != 1:
            _fail("invalid_performance_evidence")
        rp, reference = references[0]
        cp_, current = currents[0]
        metric_kind = (
            "roc_auc" if reference.ranking_scores is not None else "brier_score"
        )
        performance_draws += (
            len(reference.target_values) * reference.bootstrap_repeats
            + len(current.target_values) * current.bootstrap_repeats
        )
        rv, rs, rr = _bootstrap_metric(reference, metric_kind)
        cv, cs, cr = _bootstrap_metric(current, metric_kind)
        if (
            reference.common_support != "verified"
            or current.common_support != "verified"
        ):
            status, reason = "not_verifiable", "common_support_unverified"
            rv = cv = rs = cs = None
        elif rr or cr:
            status, reason = "undefined", rr or cr
            rv = cv = rs = cs = None
        else:
            status, reason = "available", pd.NA
        cpos = candidate_positions[reference.candidate_key]
        stability_rows.append(
            {
                "candidate_position": cpos,
                "reference_evidence_position": rp,
                "current_evidence_position": cp_,
                "evaluation_scope": reference.evaluation_scope,
                "scope_key": reference.scope_key,
                "scope_position": reference.scope_position,
                "reference_snapshot_key": reference.snapshot_key,
                "current_snapshot_key": current.snapshot_key,
                "reference_window_start": _normal_time(reference.window_start),
                "reference_window_end": _normal_time(reference.window_end),
                "current_window_start": _normal_time(current.window_start),
                "current_window_end": _normal_time(current.window_end),
                "reference_evidence_as_of": _normal_time(reference.evidence_as_of),
                "current_evidence_as_of": _normal_time(current.evidence_as_of),
                "reference_time_status": "verified",
                "current_time_status": "verified",
                "metric": metric_kind,
                "reference_value": rv,
                "current_value": cv,
                "delta": None if rv is None or cv is None else cv - rv,
                "direction": "higher_is_better"
                if metric_kind == "roc_auc"
                else "lower_is_better",
                "reference_uncertainty_std": rs,
                "current_uncertainty_std": cs,
                "reference_bootstrap_repeats": reference.bootstrap_repeats,
                "current_bootstrap_repeats": current.bootstrap_repeats,
                "reference_random_state": reference.random_state,
                "current_random_state": current.random_state,
                "reference_support_n": len(reference.target_values),
                "current_support_n": len(current.target_values),
                "reference_assignment_mechanism": reference.assignment_mechanism,
                "current_assignment_mechanism": current.assignment_mechanism,
                "reference_common_support": reference.common_support,
                "current_common_support": current.common_support,
                "reference_source_fingerprint": ref_by_id[id(reference.source_ref)][
                    "fingerprint"
                ],
                "current_source_fingerprint": ref_by_id[id(current.source_ref)][
                    "fingerprint"
                ],
                "status": status,
                "reason": reason,
                "finding_key": f"governance:stability:{cpos}:{rp}:{cp_}",
            }
        )
    _resource_preflight(
        primitive_resource_projection + (drift_draws, performance_draws)
    )

    # Pair/criterion source-backed comparisons.
    candidate_refs: dict[
        str, list[tuple[GovernanceEvidenceRef, dict[str, object]]]
    ] = {}
    for candidate in candidates:
        candidate_refs[candidate.candidate_key] = [
            (ref, ref_by_id[id(ref)]) for ref in candidate.evidence_refs
        ]
    comparison_rows: list[dict[str, object]] = []
    evaluation_rows: list[dict[str, object]] = []
    for pair_position, (champion_key, challenger_key) in enumerate(
        policy.comparison_pairs
    ):
        c0 = candidates[candidate_positions[champion_key]]
        for criterion_position, criterion in enumerate(policy.criteria):
            matches: list[tuple[GovernanceEvidenceRef, dict[str, object]]] = []
            for key in (champion_key, challenger_key):
                found = [
                    (r, i)
                    for r, i in candidate_refs[key]
                    if r.source_use == "comparison_criterion"
                    and r.source_task == criterion.source_task
                    and r.source_table == criterion.source_table
                    and r.metric_key == criterion.metric_key
                    and r.scope_key == criterion.scope_key
                    and r.scope_position == criterion.scope_position
                    and r.rule_key == criterion.rule_key
                ]
                if len(found) != 1:
                    _fail("invalid_source_binding")
                matches.append(found[0])
            (r0, i0), (r1, i1) = matches
            snapshot_status, entity_status = _proof_status(
                r0.source_task,
                r0.source_result_position,
                r1.source_task,
                r1.source_result_position,
                alignment_required=policy.entity_alignment == "owner_verified",
            )
            s0, s1 = i0["status"], i1["status"]
            supports = (i0["support"], i1["support"])
            comparable = (
                s0 == "available"
                and s1 == "available"
                and not any(pd.isna(x) for x in supports)
                and supports[0] == supports[1]
                and int(supports[0]) >= criterion.minimum_support
                and i0["unit"] == i1["unit"]
            )
            proof = _comparison_proof_failure(
                criterion,
                r0,
                r1,
                time_statuses[i0["ordinal"]],
                time_statuses[i1["ordinal"]],
                alignment_required=policy.entity_alignment == "owner_verified",
            )
            if proof is not None:
                snapshot_status, entity_status, proof_reason = proof
            else:
                proof_reason = None
            status, reason, outcome = "available", pd.NA, "not_comparable"
            v0 = v1 = delta = None
            source_mappings = tuple(
                mapping
                for mapping in (
                    _task19_source_status_reason(s0, i0["reason"]),
                    _task19_source_status_reason(s1, i1["reason"]),
                )
                if mapping is not None
            )
            if source_mappings:
                status, reason = source_mappings[0]
            elif criterion.criterion_role == "diagnostic":
                if proof_reason is not None:
                    status, reason = "not_verifiable", proof_reason
                else:
                    status, reason, outcome = (
                        "not_applicable",
                        "diagnostic_only",
                        "not_directional",
                    )
                if comparable and proof_reason is None:
                    try:
                        v0, v1 = float(i0["value"]), float(i1["value"])
                    except (TypeError, ValueError):
                        _fail("invalid_owner_value")
                    if not math.isfinite(v0) or not math.isfinite(v1):
                        _fail("invalid_owner_value")
                    delta = v1 - v0
            elif proof_reason is not None:
                status, reason = "not_verifiable", proof_reason
            elif not comparable:
                status, reason = "not_verifiable", "support_not_comparable"
            else:
                try:
                    v0, v1 = float(i0["value"]), float(i1["value"])
                except (TypeError, ValueError):
                    _fail("invalid_owner_value")
                if not math.isfinite(v0) or not math.isfinite(v1):
                    _fail("invalid_owner_value")
                delta = v1 - v0
                if criterion.direction == "higher_is_better":
                    outcome = (
                        "challenger_better"
                        if v1 > v0
                        else "champion_better"
                        if v1 < v0
                        else "tie"
                    )
                elif criterion.direction == "lower_is_better":
                    outcome = (
                        "challenger_better"
                        if v1 < v0
                        else "champion_better"
                        if v1 > v0
                        else "tie"
                    )
                else:
                    low, high = criterion.target_low, criterion.target_high
                    assert low is not None and high is not None
                    d0 = low - v0 if v0 < low else v0 - high if v0 > high else 0.0
                    d1 = low - v1 if v1 < low else v1 - high if v1 > high else 0.0
                    outcome = (
                        "challenger_better"
                        if d1 < d0
                        else "champion_better"
                        if d1 > d0
                        else "tie"
                    )
            support_identity = (
                _canonical_json(
                    (
                        criterion.source_task,
                        criterion.source_table,
                        criterion.metric_key,
                        criterion.scope_key,
                        criterion.scope_position,
                        int(supports[0]),
                        i0["unit"],
                    )
                )
                if comparable and proof_reason is None
                else None
            )
            comparison_rows.append(
                {
                    "pair_position": pair_position,
                    "champion_candidate_position": candidate_positions[champion_key],
                    "challenger_candidate_position": candidate_positions[
                        challenger_key
                    ],
                    "candidate_family": c0.candidate_family,
                    "criterion_position": criterion_position,
                    "criterion_role": criterion.criterion_role,
                    "source_task": criterion.source_task,
                    "source_table": criterion.source_table,
                    "metric_key": criterion.metric_key,
                    "scope_key": criterion.scope_key,
                    "scope_position": criterion.scope_position,
                    "rule_key": criterion.rule_key,
                    "champion_source_result_position": r0.source_result_position,
                    "challenger_source_result_position": r1.source_result_position,
                    "champion_source_ref_position": i0["ordinal"],
                    "challenger_source_ref_position": i1["ordinal"],
                    "champion_source_fingerprint": i0["fingerprint"],
                    "challenger_source_fingerprint": i1["fingerprint"],
                    "champion_time_status": time_statuses[i0["ordinal"]],
                    "challenger_time_status": time_statuses[i1["ordinal"]],
                    "champion_source_status": s0,
                    "champion_source_reason": i0["reason"],
                    "challenger_source_status": s1,
                    "challenger_source_reason": i1["reason"],
                    "source_snapshot_status": snapshot_status,
                    "entity_alignment_status": entity_status,
                    "normalized_support_identity": support_identity,
                    "champion_value": v0,
                    "challenger_value": v1,
                    "delta": delta,
                    "champion_support_n": supports[0]
                    if comparable and proof_reason is None
                    else None,
                    "challenger_support_n": supports[1]
                    if comparable and proof_reason is None
                    else None,
                    "support_unit": i0["unit"]
                    if comparable and proof_reason is None
                    else None,
                    "direction": criterion.direction if proof_reason is None else None,
                    "target_low": criterion.target_low,
                    "target_high": criterion.target_high,
                    "comparison_outcome": outcome,
                    "support_comparable": comparable and proof_reason is None,
                    "status": status,
                    "reason": reason,
                    "finding_key": (
                        f"governance:comparison:{pair_position}:{criterion_position}"
                    ),
                }
            )
            eligible = comparable and proof_reason is None
            counts = (
                status == "available"
                and criterion.criterion_role == "decision"
                and outcome in ("challenger_better", "champion_better", "tie")
            )
            blocks = criterion.required_for_promotion and not counts
            evaluation_rows.append(
                {
                    "pair_position": pair_position,
                    "champion_candidate_position": candidate_positions[champion_key],
                    "challenger_candidate_position": candidate_positions[
                        challenger_key
                    ],
                    "criterion_position": criterion_position,
                    "criterion_role": criterion.criterion_role,
                    "required_for_promotion": criterion.required_for_promotion,
                    "priority": criterion.priority,
                    "comparison_outcome": outcome,
                    "comparable": eligible,
                    "counts_toward_minimum": counts,
                    "blocks_promotion": blocks,
                    "directional_contribution": outcome,
                    "evidence_time_status": "verified"
                    if time_statuses[i0["ordinal"]]
                    == time_statuses[i1["ordinal"]]
                    == "verified"
                    else "unverified",
                    "status": status,
                    "reason": reason,
                    "finding_key": (
                        f"governance:evaluation:{pair_position}:{criterion_position}"
                    ),
                }
            )

    recommendation_rows: list[dict[str, object]] = []
    for pair_position, (_, challenger_key) in enumerate(policy.comparison_pairs):
        subset = [
            x
            for x in evaluation_rows
            if x["pair_position"] == pair_position and x["criterion_role"] == "decision"
        ]
        challenger = candidates[candidate_positions[challenger_key]]
        available = [x for x in subset if x["counts_toward_minimum"]]
        better = sum(x["comparison_outcome"] == "challenger_better" for x in available)
        worse = sum(x["comparison_outcome"] == "champion_better" for x in available)
        tied = sum(x["comparison_outcome"] == "tie" for x in available)
        incomplete = sum(bool(x["blocks_promotion"]) for x in subset)
        hard = challenger.declared_state in ("rejected", "retired")
        if hard:
            rec, basis = "reject_challenger", "hard_state_veto"
        elif incomplete or len(available) < policy.minimum_comparable_criteria:
            rec, basis = (
                "insufficient_evidence",
                "required_evidence_incomplete" if incomplete else "minimum_not_met",
            )
        elif better and worse:
            rec, basis = "continue_evaluation", "mixed_evidence"
        elif tied == len(available):
            rec, basis = "continue_evaluation", "all_ties"
        elif better and not worse:
            rec, basis = "promote_challenger", "challenger_favorable"
        else:
            rec, basis = "retain_champion", "champion_favorable"
        review = (
            rec == "promote_challenger"
            or policy.human_review_mode == "all_recommendations"
        )
        recommendation_rows.append(
            {
                "pair_position": pair_position,
                "champion_candidate_position": candidate_positions[
                    champion.candidate_key
                ],
                "challenger_candidate_position": candidate_positions[challenger_key],
                "candidate_family": challenger.candidate_family,
                "recommendation": rec,
                "recommendation_basis": basis,
                "hard_veto": hard,
                "human_review_mode": policy.human_review_mode,
                "human_review_required": review,
                "minimum_comparable_criteria": policy.minimum_comparable_criteria,
                "criteria_available_n": len(available),
                "criteria_unavailable_n": len(subset) - len(available),
                "criteria_better_n": better,
                "criteria_worse_n": worse,
                "criteria_tied_n": tied,
                "required_incomplete_n": incomplete,
                "support_comparable": all(bool(x["comparable"]) for x in subset),
                "status": "available",
                "reason": pd.NA,
                "finding_key": f"governance:recommendation:{pair_position}",
            }
        )

    metadata_rows: list[dict[str, object]] = []
    if (
        sum(
            x.metadata_scope == "governance" and x.candidate_key is None
            for x in policy.metadata
        )
        != 1
    ):
        _fail("invalid_metadata")
    seen_metadata: set[tuple[str, str | None]] = set()
    for mp, item in enumerate(policy.metadata):
        identity = (item.metadata_scope, item.candidate_key)
        if (
            identity in seen_metadata
            or not _is_safe_key(item.metadata_key)
            or (item.metadata_scope == "candidate") != (item.candidate_key is not None)
        ):
            _fail("invalid_metadata")
        seen_metadata.add(identity)
        if (
            item.candidate_key is not None
            and item.candidate_key not in candidate_positions
        ):
            _fail("invalid_metadata")
        items: list[tuple[str, int | None, object, bool]] = [
            ("purpose", None, item.purpose_key, False),
            ("owner", None, item.owner_key, False),
            ("materiality", None, item.materiality, False),
        ]
        items.extend(
            ("assumption", i, x, False) for i, x in enumerate(item.assumption_keys)
        )
        items.extend(
            ("limitation", i, x, False) for i, x in enumerate(item.limitation_keys)
        )
        items.extend(
            ("threshold", i, x[1], True)
            for i, x in enumerate(item.monitoring_thresholds)
        )
        items.extend(
            (
                ("issue_status", None, item.issue_status, False),
                ("remediation_status", None, item.remediation_status, False),
            )
        )
        for fp, (key, ip, value, numeric) in enumerate(items):
            branch = (
                f"candidate:{candidate_positions[item.candidate_key]}"
                if item.candidate_key is not None
                else "governance"
            )
            metadata_rows.append(
                {
                    "metadata_position": mp,
                    "metadata_scope": item.metadata_scope,
                    "candidate_position": candidate_positions.get(item.candidate_key)
                    if item.candidate_key
                    else None,
                    "field_position": fp,
                    "field_key": key,
                    "item_position": ip,
                    "text_value": None if numeric else value,
                    "numeric_value": value if numeric else None,
                    "evidence_time_status": "not_applicable",
                    "status": "available",
                    "reason": pd.NA,
                    "finding_key": f"governance:metadata:{branch}:{mp}:{fp}",
                }
            )

    summary_rows: list[dict[str, object]] = []
    for cp, candidate in enumerate(candidates):
        related = [
            x
            for x in comparison_rows
            if cp
            in (x["champion_candidate_position"], x["challenger_candidate_position"])
        ]
        summary_rows.append(
            {
                "candidate_position": cp,
                "candidate_family": candidate.candidate_family,
                "declared_role": candidate.declared_role,
                "declared_state": candidate.declared_state,
                "source_task": candidate.source_task,
                "source_result_position": candidate.source_result_position,
                "source_candidate_position": cp
                if candidate.source_candidate_key is not None
                else None,
                "source_snapshot_status": "not_applicable",
                "entity_alignment_status": "not_applicable",
                "evidence_time_status": "verified"
                if all(x == "verified" for x in time_statuses)
                else "unverified",
                "criterion_count": len(related),
                "available_criterion_count": sum(
                    x["status"] == "available" for x in related
                ),
                "unavailable_criterion_count": sum(
                    x["status"] in ("unavailable", "undefined") for x in related
                ),
                "not_verifiable_criterion_count": sum(
                    x["status"] == "not_verifiable" for x in related
                ),
                "attribution_count": sum(
                    x["candidate_position"] == cp for x in attribution_rows
                ),
                "prediction_drift_count": sum(
                    x["candidate_position"] == cp for x in drift_rows
                ),
                "performance_stability_count": sum(
                    x["candidate_position"] == cp for x in stability_rows
                ),
                "recommendation_count": sum(
                    cp
                    in (
                        x["champion_candidate_position"],
                        x["challenger_candidate_position"],
                    )
                    for x in recommendation_rows
                ),
                "human_review_required_count": sum(
                    cp
                    in (
                        x["champion_candidate_position"],
                        x["challenger_candidate_position"],
                    )
                    and x["human_review_required"]
                    for x in recommendation_rows
                ),
                "status": "available",
                "reason": pd.NA,
                "finding_key": f"governance:summary:{cp}",
            }
        )

    policy_fp = _fingerprint(policy)
    fingerprints = [
        policy_fp,
        _fingerprint(policy.candidates),
        _fingerprint(policy.criteria),
        _fingerprint(
            tuple(
                (r.source_task, r.source_result_position, r.source_table, r.source_use)
                for r in refs
            )
        ),
        _fingerprint(policy.explanations),
        _fingerprint(model_attributions),
        _fingerprint(prediction_profiles),
        _fingerprint(performance_evidence),
        _fingerprint(policy.metadata),
    ]
    structured_fp = _fingerprint(tuple(fingerprints[4:9]))
    fingerprints.append(structured_fp)
    governance_fp = _fingerprint(
        (
            "task19-contract-targeted-fixed-v2",
            "0.1.0",
            policy.analysis_as_of,
            "task19-source-registry-38-v1",
            "task19-direction-registry-92-v1",
            "task19-recommendation-matrix-5-v1",
            tuple(fingerprints),
            tuple((k, len(v)) for k, v in owners.items()),
        )
    )
    fingerprints.append(governance_fp)
    source_snapshot_status = "not_applicable" if not refs else "unverified"
    alignment_status = (
        "not_applicable" if policy.entity_alignment == "not_requested" else "verified"
    )
    evidence_time_status = (
        "not_applicable"
        if not refs
        and not model_attributions
        and not prediction_profiles
        and not performance_evidence
        else "verified"
        if all(x == "verified" for x in time_statuses)
        else "unverified"
    )
    provenance_keys = (
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
    provenance_values: tuple[object, ...] = (
        "task19-contract-targeted-fixed-v2",
        "0.1.0",
        "task19-canonical-json-v1",
        "task19-source-registry-38-v1",
        "task19-direction-registry-92-v1",
        "task19-recommendation-matrix-5-v1",
        policy.analysis_as_of,
        "aware_utc" if aware else "naive_wall_clock",
        fingerprints[0],
        fingerprints[1],
        fingerprints[2],
        fingerprints[3],
        tuple(
            (
                i,
                r.source_task,
                r.source_result_position,
                r.source_table,
                r.source_use,
                ref_by_id[id(r)]["registry_position"],
            )
            for i, r in enumerate(refs)
        ),
        fingerprints[4],
        fingerprints[5],
        fingerprints[6],
        fingerprints[7],
        fingerprints[8],
        fingerprints[9],
        fingerprints[10],
        source_snapshot_status,
        alignment_status,
        evidence_time_status,
        len(candidates),
        len(policy.comparison_pairs),
        len(policy.criteria),
        len(risk_validations),
        len(data_audits),
        len(decision_strategies),
        len(lifecycle_monitorings),
        tuple("unavailable" for _ in risk_validations),
        tuple(x.config_fingerprint for x in data_audits),
        tuple(x.strategy_fingerprint for x in decision_strategies),
        tuple(x.monitoring_fingerprint for x in lifecycle_monitorings),
        tuple(time_statuses),
    )
    provenance_rows = [
        {
            "provenance_position": i,
            "provenance_key": key,
            "provenance_value": _canonical_json(value),
            "status": "available",
            "reason": pd.NA,
            "finding_key": f"governance:provenance:{i}",
        }
        for i, (key, value) in enumerate(
            zip(provenance_keys, provenance_values, strict=True)
        )
    ]

    _validate_fixed_invariants(
        prediction_profiles=len(prediction_profiles),
        source_evidence_rows=len(resolved),
        evidence_refs=len(refs),
        prediction_drift_rows=len(drift_rows),
        performance_evidence=len(performance_evidence),
        performance_stability_rows=len(stability_rows),
        pairs=len(policy.comparison_pairs),
        criteria=len(policy.criteria),
        candidate_comparison_rows=len(comparison_rows),
        governance_evaluation_rows=len(evaluation_rows),
        recommendation_rows=len(recommendation_rows),
        candidates=len(candidates),
        governance_summary_rows=len(summary_rows),
        provenance_rows=len(provenance_rows),
    )

    frames = {
        name: _frame(name, rows, aware=aware)
        for name, rows in (
            ("explanations", explanation_rows),
            ("model_attributions", attribution_rows),
            ("prediction_drift", drift_rows),
            ("performance_stability", stability_rows),
            ("candidate_comparisons", comparison_rows),
            ("governance_evaluations", evaluation_rows),
            ("recommendations", recommendation_rows),
            ("governance_summary", summary_rows),
            ("governance_metadata", metadata_rows),
            ("provenance", provenance_rows),
        )
    }
    warnings: list[str] = []
    limitations: list[str] = []
    for name in _SCHEMAS:
        table = frames[name]
        for row in table.loc[
            table["status"].isin(("unavailable", "undefined", "not_verifiable"))
        ].to_dict("records"):
            warnings.append(row["finding_key"])
            if row["reason"] not in limitations:
                limitations.append(row["reason"])
    return GovernanceResult(
        policy.governance_key,
        policy.governance_version,
        governance_fp,
        _normal_time(policy.analysis_as_of),
        len(candidates),
        len(policy.comparison_pairs),
        len(policy.criteria),
        len(policy.explanations),
        source_snapshot_status,
        alignment_status,
        evidence_time_status,
        frames["explanations"],
        frames["model_attributions"],
        frames["prediction_drift"],
        frames["performance_stability"],
        frames["candidate_comparisons"],
        frames["governance_evaluations"],
        frames["recommendations"],
        frames["governance_summary"],
        frames["governance_metadata"],
        frames["provenance"],
        tuple(warnings),
        tuple(limitations),
    )


assert len(_SOURCE_REGISTRY) == 38
assert len(_DIRECTION_REGISTRY) == 92
assert len(_ERROR_KEYS) == 76
assert len(_REASONS) == 15
assert len(_SCHEMAS) == 10
