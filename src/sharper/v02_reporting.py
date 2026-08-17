"""Static Markdown/HTML reporting for the Sharper v0.2 integration result."""

from __future__ import annotations

import html
import shutil
from dataclasses import dataclass, fields
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from sharper.data_audit import DataAuditResult
from sharper.decision_strategy import DecisionStrategyResult
from sharper.lifecycle_monitoring import LifecycleMonitoringResult
from sharper.model_governance import GovernanceResult
from sharper.reporting import ReportArtifact
from sharper.risk_validation import BinaryRiskValidationResult
from sharper.v02_workflow import V02WorkflowResult
from sharper.visualization import plot_binary_risk_validation, plot_model_governance

_REPORT_DEFAULT_TITLE = "Sharper v0.2 Integration Report"
_REPORT_FIGURE_LIMIT = 9
_REPORT_PNG_BYTES_LIMIT = 64_000_000
_PATH_ORDER = ("score_validation", "audit", "preloan", "postloan", "governance")
_PATH_STATUS_COLUMNS = ("path_key", "enabled", "status", "reason")

_SCORE_SCALARS = (
    "validation_mode",
    "prediction_scope",
    "score_source",
    "score_direction",
    "probability_provenance",
    "input_n_rows",
    "eligible_n_rows",
    "predicted_n_rows",
    "evaluable_n_rows",
    "requested_threshold_count",
    "actual_threshold_count",
    "observed_loss_maturity_mode",
    "observed_loss_mature_n",
    "observed_loss_excluded_n",
)
_AUDIT_SCALARS = (
    "n_rows",
    "n_columns",
    "reference_n_rows",
    "reference_n_columns",
)
_PRELOAN_SCALARS = (
    "strategy_key",
    "strategy_version",
    "input_n_rows",
    "decided_n_rows",
    "unavailable_n_rows",
    "requested_rule_count",
    "active_rule_count",
    "requested_constraint_count",
)
_POSTLOAN_SCALARS = (
    "monitoring_key",
    "monitoring_version",
    "input_n_rows",
    "entity_count",
    "evaluable_observation_count",
    "requested_scenario_count",
    "requested_rule_count",
    "active_rule_count",
    "requested_state_count",
)
_GOVERNANCE_SCALARS = (
    "governance_key",
    "governance_version",
    "analysis_as_of",
    "candidate_count",
    "comparison_pair_count",
    "criterion_count",
    "explanation_count",
    "source_snapshot_status",
    "entity_alignment_status",
    "evidence_time_status",
)

_METRICS_COLUMNS = (
    "scope",
    "fold_id",
    "metric",
    "statistic",
    "value",
    "at_threshold",
    "status",
    "reason",
    "n_rows",
    "n_positive",
    "n_negative",
)
_BUSINESS_COLUMNS = (
    "segment_kind",
    "segment_value",
    "metric",
    "value",
    "status",
    "reason",
    "n_rows",
    "n_evaluable_rows",
    "n_observed_loss_mature_rows",
    "unit",
)
_GAINS_PLOT_COLUMNS = (
    "scope",
    "fold_id",
    "actual_fraction",
    "capture",
    "capture_status",
    "capture_reason",
    "lift",
    "lift_status",
    "lift_reason",
)
_CALIBRATION_PLOT_COLUMNS = (
    "scope",
    "n_rows",
    "mean_predicted_probability",
    "observed_event_rate",
    "status",
    "reason",
)
_THRESHOLD_PLOT_COLUMNS = (
    "scope",
    "fold_id",
    "sensitivity",
    "sensitivity_status",
    "sensitivity_reason",
    "specificity",
    "specificity_status",
    "specificity_reason",
    "precision",
    "precision_status",
    "precision_reason",
    "predicted_positive_rate",
    "predicted_positive_rate_status",
    "predicted_positive_rate_reason",
)

_DATASET_PROFILE_COLUMNS = (
    "side",
    "n_rows",
    "n_columns",
    "profiled_column_count",
    "declared_feature_count",
    "feature_status",
    "feature_reason",
    "duplicate_row_count",
    "duplicate_row_rate",
    "duplicate_row_status",
    "duplicate_row_reason",
    "duplicate_index_count",
    "duplicate_index_rate",
    "duplicate_index_status",
    "duplicate_index_reason",
    "memory_usage_bytes",
    "finding_key",
)
_POINT_IN_TIME_COLUMNS = (
    "side",
    "scope",
    "column",
    "evaluated_count",
    "violation_count",
    "not_verifiable_count",
    "status",
    "reason",
    "finding_key",
)
_MISSINGNESS_DRIFT_COLUMNS = (
    "column",
    "reference_present",
    "current_present",
    "reference_n",
    "current_n",
    "reference_missing_count",
    "current_missing_count",
    "reference_missing_rate",
    "current_missing_rate",
    "absolute_rate_change",
    "relative_rate_change",
    "new_all_missing",
    "recovered",
    "count_status",
    "count_reason",
    "rate_status",
    "rate_reason",
    "reference_count_status",
    "reference_count_reason",
    "current_count_status",
    "current_count_reason",
    "reference_rate_status",
    "reference_rate_reason",
    "current_rate_status",
    "current_rate_reason",
    "absolute_change_status",
    "absolute_change_reason",
    "relative_change_status",
    "relative_change_reason",
    "finding_key",
)

_ROW_DECISION_COLUMNS = (
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
_PRELOAN_BUSINESS_COLUMNS = (
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
_PRELOAN_RULE_COLUMNS = (
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
_MONITORING_SUMMARY_COLUMNS = (
    "scope_key",
    "scope_position",
    "scenario_key",
    "rule_key",
    "metric",
    "metric_value",
    "numerator",
    "denominator",
    "support_n",
    "support_unit",
    "mature_n",
    "censored_n",
    "unit",
    "status",
    "reason",
    "finding_key",
)
_LIFECYCLE_SUMMARY_COLUMNS = (
    "scope_key",
    "scope_position",
    "from_state_key",
    "to_state_key",
    "metric",
    "metric_value",
    "numerator",
    "denominator",
    "support_n",
    "support_unit",
    "unit",
    "status",
    "reason",
    "finding_key",
)
_OBSERVATION_HISTORY_COLUMNS = (
    "row_position",
    "entity_position",
    "observation_time",
    "observation_status",
    "observation_reason",
    "primary_scenario_key",
    "primary_rule_key",
    "primary_alert_level",
    "primary_alert_rank",
    "active_rule_count",
    "emitted_notification_count",
    "maturity_status",
    "effective_state_key",
    "effective_state_rank",
    "state_status",
    "state_reason",
)
_GOVERNANCE_SUMMARY_COLUMNS = (
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
)
_RECOMMENDATION_COLUMNS = (
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
)
_COMPARISON_COLUMNS = (
    "pair_position",
    "champion_candidate_position",
    "challenger_candidate_position",
    "candidate_family",
    "criterion_position",
    "source_task",
    "source_table",
    "metric_key",
    "scope_key",
    "scope_position",
    "champion_value",
    "challenger_value",
    "delta",
    "comparison_outcome",
    "support_comparable",
    "status",
    "reason",
    "finding_key",
)
_GOVERNANCE_EVALUATION_COLUMNS = (
    "pair_position",
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
)
_EXPLANATION_COLUMNS = (
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
)
_PREDICTION_DRIFT_COLUMNS = (
    "candidate_position",
    "reference_profile_position",
    "current_profile_position",
    "prediction_kind",
    "scope_key",
    "scope_position",
    "metric",
    "prediction_tvd",
    "direction",
    "status",
    "reason",
    "finding_key",
)
_PERFORMANCE_STABILITY_COLUMNS = (
    "candidate_position",
    "reference_evidence_position",
    "current_evidence_position",
    "evaluation_scope",
    "scope_key",
    "scope_position",
    "metric",
    "reference_value",
    "current_value",
    "delta",
    "direction",
    "reference_support_n",
    "current_support_n",
    "reference_assignment_mechanism",
    "current_assignment_mechanism",
    "reference_common_support",
    "current_common_support",
    "status",
    "reason",
    "finding_key",
)
_AUDIT_PROVENANCE_COLUMNS = (
    "provenance_key",
    "value_type",
    "numeric_value",
    "text_value",
    "count_value",
    "boolean_value",
    "status",
    "reason",
)
_OWNER_PROVENANCE_COLUMNS = (
    "provenance_key",
    "provenance_value",
    "status",
    "reason",
    "finding_key",
)
_GOVERNANCE_PROVENANCE_COLUMNS = (
    "provenance_position",
    "provenance_key",
    "provenance_value",
    "status",
    "reason",
    "finding_key",
)


@dataclass(frozen=True)
class _Table:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class _Slot:
    ordinal: int
    key: str
    owner: Literal["risk", "governance"]
    kind: str
    section: str
    enabled: bool


@dataclass(frozen=True)
class _Section:
    title: str
    marker: str | None = None
    fields: tuple[tuple[str, str], ...] = ()
    tables: tuple[_Table, ...] = ()
    slots: tuple[_Slot, ...] = ()


@dataclass(frozen=True)
class _ReportModel:
    title: str
    sections: tuple[_Section, ...]
    slots: tuple[_Slot, ...]


class _AssetBudgetExceeded(Exception):
    """Internal marker for the staged PNG aggregate gate."""


def _report_error(key: str) -> None:
    raise ValueError(f"sharper task20: report_{key}")


def generate_v02_report(
    result: V02WorkflowResult,
    output_path: str | Path,
    *,
    title: str = _REPORT_DEFAULT_TITLE,
    format: Literal["markdown", "html"] = "markdown",
    overwrite: bool = True,
) -> ReportArtifact:
    """Write a static v0.2 report and fixed-slot PNG asset bundle.

    Parameters
    ----------
    result
        A complete :class:`~sharper.v02_workflow.V02WorkflowResult`; only its
        frozen scalars and owner evidence tables are read.
    output_path
        A string or :class:`pathlib.Path` naming the report file.
    title
        Report title. Line boundaries are replaced with one ASCII space.
    format
        Exactly ``"markdown"`` or ``"html"``.
    overwrite
        Whether an existing report/asset bundle may be transactionally replaced.

    Returns
    -------
    sharper.reporting.ReportArtifact
        The existing Task 13 artifact type for the written report.

    Raises
    ------
    ValueError
        For a Task20 report contract, format, result, title, path, or budget
        violation. Task20 report errors use the ``sharper task20: report_*``
        prefix.
    FileExistsError
        If overwrite is false and a target, staging, or backup path exists.
    OSError
        If filesystem staging, replacement, cleanup, or rollback fails.

    Notes
    -----
    No owner analysis or model operation is performed. Acquired Figures are
    owned by this function and closed exactly once on every post-acquisition
    path.
    """
    _validate_result_shell(result)
    cleaned_title = _clean_title(title)
    _validate_format(format)
    path = _validate_output_path(output_path)
    if type(overwrite) is not bool:
        _report_error("overwrite")

    assets = path.parent / f"{path.stem}_assets"
    staging_report = path.parent / f".{path.name}.sharper-staging"
    staging_assets = path.parent / f".{assets.name}.sharper-staging"
    backup_report = path.parent / f".{path.name}.sharper-backup"
    backup_assets = path.parent / f".{assets.name}.sharper-backup"

    if not overwrite and (path.exists() or assets.exists()):
        raise FileExistsError("output file or asset directory already exists")
    if any(
        candidate.exists()
        for candidate in (staging_report, staging_assets, backup_report, backup_assets)
    ):
        raise FileExistsError("staging or backup path already exists")

    try:
        model = _build_model(result, cleaned_title)
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception as error:
        raise ValueError("sharper task20: report_result") from error

    if sum(slot.enabled for slot in model.slots) > _REPORT_FIGURE_LIMIT:
        _report_error("asset_budget")
    try:
        body = _render(model, format, assets.name)
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception as error:
        raise ValueError("sharper task20: report_result") from error

    figures: list[tuple[_Slot, Figure]] = []
    assets_backed_up = False
    report_backed_up = False
    assets_committed = False
    try:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            raise OSError("failed to write report output") from error

        try:
            _acquire_figures(result, model.slots, figures)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except OSError:
            raise
        except Exception as error:
            raise ValueError("sharper task20: report_result") from error

        try:
            staging_assets.mkdir()
            png_total = 0
            for slot, figure in figures:
                png_path = staging_assets / f"plot-{slot.ordinal:03d}.png"
                figure.savefig(png_path)
                png_total += int(png_path.stat().st_size)
                if png_total > _REPORT_PNG_BYTES_LIMIT:
                    raise _AssetBudgetExceeded
            with staging_report.open("w", encoding="utf-8") as handle:
                handle.write(body)
                handle.flush()
        except _AssetBudgetExceeded as error:
            try:
                _clean_staging(staging_report, staging_assets)
            except Exception as compensation_error:
                raise OSError("failed to write report output") from compensation_error
            raise ValueError("sharper task20: report_asset_budget") from error
        except Exception as error:
            try:
                _clean_staging(staging_report, staging_assets)
            except Exception as compensation_error:
                raise OSError("failed to write report output") from compensation_error
            raise OSError("failed to write report output") from error

        try:
            if assets.exists():
                assets.replace(backup_assets)
                assets_backed_up = True
            if path.exists():
                path.replace(backup_report)
                report_backed_up = True
        except Exception as error:
            try:
                if assets_backed_up:
                    _restore(backup_assets, assets)
                _clean_staging(staging_report, staging_assets)
            except Exception as compensation_error:
                raise OSError("failed to write report output") from compensation_error
            raise OSError("failed to write report output") from error

        try:
            staging_assets.replace(assets)
            assets_committed = True
            staging_report.replace(path)
        except Exception as error:
            try:
                if assets_committed:
                    _remove(assets)
                if report_backed_up:
                    _restore(backup_report, path)
                if assets_backed_up:
                    _restore(backup_assets, assets)
                _clean_staging(staging_report, staging_assets)
            except Exception as compensation_error:
                raise OSError("failed to write report output") from compensation_error
            raise OSError("failed to write report output") from error

        try:
            _remove(backup_report)
        except Exception as error:
            raise OSError("failed to write report output") from error
        try:
            _remove(backup_assets)
        except Exception as error:
            raise OSError("failed to write report output") from error
        try:
            return ReportArtifact(path=path, format=format, title=cleaned_title)
        except Exception as error:
            raise ValueError("sharper task20: report_result") from error
    finally:
        for _, figure in figures:
            plt.close(figure)


def _validate_result_shell(result: object) -> None:
    expected = tuple(field.name for field in fields(V02WorkflowResult))
    if type(result) is not V02WorkflowResult:
        _report_error("result")
    if tuple(result.__dataclass_fields__) != expected:  # type: ignore[union-attr]
        _report_error("result")
    if result.contract_version != "task20-integration-v1":  # type: ignore[union-attr]
        _report_error("result")
    if type(result.enabled_paths) is not tuple:  # type: ignore[union-attr]
        _report_error("result")
    if any(type(value) is not str for value in result.enabled_paths):  # type: ignore[union-attr]
        _report_error("result")
    if tuple(result.enabled_paths) != tuple(  # type: ignore[union-attr]
        key
        for key in _PATH_ORDER
        if getattr(result, key if key != "audit" else "data_audit") is not None
    ):
        _report_error("result")
    for name in ("call_trace", "warnings", "limitations"):
        value = getattr(result, name)
        if type(value) is not tuple or any(type(item) is not str for item in value):
            _report_error("result")


def _clean_title(title: object) -> str:
    if type(title) is not str:
        _report_error("title")
    return " ".join(title.splitlines()).strip() or _REPORT_DEFAULT_TITLE


def _validate_format(format: object) -> None:
    if type(format) is not str or format not in {"markdown", "html"}:
        _report_error("format")


def _validate_output_path(output_path: object) -> Path:
    if not isinstance(output_path, (str, Path)):
        _report_error("path")
    path = Path(output_path)
    if path.exists() and path.is_dir():
        _report_error("path")
    return path


def _build_model(result: V02WorkflowResult, title: str) -> _ReportModel:
    _validate_path_status(result)
    score = _validated_owner(result.score_validation, BinaryRiskValidationResult)
    audit = _validated_owner(result.data_audit, DataAuditResult)
    preloan = _validated_owner(result.preloan, DecisionStrategyResult)
    postloan = _validated_owner(result.postloan, LifecycleMonitoringResult)
    governance = _validated_owner(result.governance, GovernanceResult)

    slots = _slot_plan(score, governance)
    sections = (
        _run_context(result),
        _path_context(result),
        _score_section(score, slots),
        _audit_section(audit),
        _preloan_section(preloan),
        _postloan_section(postloan),
        _governance_section(governance, slots),
        _cross_path_section(governance),
        _reason_section(preloan, postloan, governance),
        _stability_section(score, preloan, postloan, governance),
        _warnings_section(result),
        _provenance_section(result, score, audit, preloan, postloan, governance),
    )
    return _ReportModel(title, sections, slots)


def _validated_owner(value: object, owner_type: type) -> object | None:
    if value is None:
        return None
    if type(value) is not owner_type:
        _report_error("result")
    return value


def _validate_path_status(result: V02WorkflowResult) -> None:
    table = result.path_status
    if type(table) is not pd.DataFrame or tuple(table.columns) != _PATH_STATUS_COLUMNS:
        _report_error("result")
    if len(table) != len(_PATH_ORDER):
        _report_error("result")
    expected_enabled = {
        "score_validation": result.score_validation is not None,
        "audit": result.data_audit is not None,
        "preloan": result.preloan is not None,
        "postloan": result.postloan is not None,
        "governance": result.governance is not None,
    }
    for row, path_key in zip(table.itertuples(index=False, name=None), _PATH_ORDER):
        if (
            row[0] != path_key
            or type(row[1]) is not bool
            or row[1] != expected_enabled[path_key]
        ):
            _report_error("result")
        if type(row[2]) is not str or (
            not _is_missing(row[3]) and type(row[3]) is not str
        ):
            _report_error("result")


def _require_columns(table: object, columns: tuple[str, ...]) -> pd.DataFrame:
    if type(table) is not pd.DataFrame or not table.columns.is_unique:
        _report_error("result")
    if not set(columns).issubset(table.columns):
        _report_error("result")
    positions = [table.columns.get_loc(column) for column in columns]
    if positions != sorted(positions):
        _report_error("result")
    return table


def _table(title: str, table: object, columns: tuple[str, ...]) -> _Table:
    frame = _require_columns(table, columns)
    rows = tuple(
        tuple(_value_text(value) for value in row)
        for row in frame.loc[:, list(columns)].itertuples(index=False, name=None)
    )
    return _Table(title, columns, rows)


def _fields(owner: object, names: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((name, _value_text(getattr(owner, name))) for name in names)


def _run_context(result: V02WorkflowResult) -> _Section:
    return _Section(
        "Run Context",
        fields=(
            ("contract_version", _value_text(result.contract_version)),
            ("enabled_paths", _tuple_text(result.enabled_paths)),
            ("call_trace", _tuple_text(result.call_trace)),
        ),
    )


def _path_context(result: V02WorkflowResult) -> _Section:
    return _Section(
        "Path Status",
        tables=(_table("path_status", result.path_status, _PATH_STATUS_COLUMNS),),
    )


def _score_section(
    owner: BinaryRiskValidationResult | None, slots: tuple[_Slot, ...]
) -> _Section:
    if owner is None:
        return _Section("Score Validation", marker="not_requested")
    tables = (
        _table("metrics", owner.metrics, _METRICS_COLUMNS),
        _table("business_metrics", owner.business_metrics, _BUSINESS_COLUMNS),
        _table("gains", owner.gains, _GAINS_PLOT_COLUMNS),
        _table("calibration", owner.calibration, _CALIBRATION_PLOT_COLUMNS),
        _table("threshold_analysis", owner.threshold_analysis, _THRESHOLD_PLOT_COLUMNS),
    )
    if all(not table.rows for table in tables):
        return _Section("Score Validation", marker="empty_result:score_validation")
    return _Section(
        "Score Validation",
        fields=_fields(owner, _SCORE_SCALARS),
        tables=tables[:2],
        slots=tuple(slot for slot in slots if slot.section == "Score Validation"),
    )


def _audit_section(owner: DataAuditResult | None) -> _Section:
    if owner is None:
        return _Section("Data Audit and Leakage", marker="not_requested")
    tables = (
        _table("dataset_profile", owner.dataset_profile, _DATASET_PROFILE_COLUMNS),
        _table(
            "point_in_time_profile", owner.point_in_time_profile, _POINT_IN_TIME_COLUMNS
        ),
        _table(
            "missingness_drift", owner.missingness_drift, _MISSINGNESS_DRIFT_COLUMNS
        ),
    )
    if all(not table.rows for table in tables):
        return _Section("Data Audit and Leakage", marker="empty_result:data_audit")
    return _Section(
        "Data Audit and Leakage", fields=_fields(owner, _AUDIT_SCALARS), tables=tables
    )


def _preloan_section(owner: DecisionStrategyResult | None) -> _Section:
    if owner is None:
        return _Section("Pre-loan Eligibility", marker="not_requested")
    tables = (
        _table("row_decisions", owner.row_decisions, _ROW_DECISION_COLUMNS),
        _table("business_summary", owner.business_summary, _PRELOAN_BUSINESS_COLUMNS),
        _table("constraint_summary", owner.constraint_summary, _CONSTRAINT_COLUMNS),
    )
    if all(not table.rows for table in tables):
        return _Section("Pre-loan Eligibility", marker="empty_result:preloan")
    return _Section(
        "Pre-loan Eligibility", fields=_fields(owner, _PRELOAN_SCALARS), tables=tables
    )


def _postloan_section(owner: LifecycleMonitoringResult | None) -> _Section:
    if owner is None:
        return _Section("Post-loan Warning", marker="not_requested")
    tables = (
        _table(
            "monitoring_summary", owner.monitoring_summary, _MONITORING_SUMMARY_COLUMNS
        ),
        _table(
            "lifecycle_summary", owner.lifecycle_summary, _LIFECYCLE_SUMMARY_COLUMNS
        ),
        _table(
            "observation_history",
            owner.observation_history,
            _OBSERVATION_HISTORY_COLUMNS,
        ),
    )
    if all(not table.rows for table in tables):
        return _Section("Post-loan Warning", marker="empty_result:postloan")
    return _Section(
        "Post-loan Warning", fields=_fields(owner, _POSTLOAN_SCALARS), tables=tables
    )


def _governance_section(
    owner: GovernanceResult | None, slots: tuple[_Slot, ...]
) -> _Section:
    if owner is None:
        return _Section("Governance", marker="not_requested")
    tables = (
        _table(
            "governance_summary", owner.governance_summary, _GOVERNANCE_SUMMARY_COLUMNS
        ),
        _table("recommendations", owner.recommendations, _RECOMMENDATION_COLUMNS),
        _table(
            "model_attributions",
            owner.model_attributions,
            ("feature_key", "value", "status"),
        ),
        _table(
            "candidate_comparisons",
            owner.candidate_comparisons,
            ("criterion_position", "delta", "status"),
        ),
        _table(
            "prediction_drift",
            owner.prediction_drift,
            ("scope_key", "prediction_tvd", "status"),
        ),
        _table(
            "performance_stability",
            owner.performance_stability,
            ("scope_key", "delta", "status"),
        ),
    )
    if all(not table.rows for table in tables[:2]):
        return _Section("Governance", marker="empty_result:governance")
    return _Section(
        "Governance",
        fields=_fields(owner, _GOVERNANCE_SCALARS),
        tables=tables[:2],
        slots=tuple(slot for slot in slots if slot.section == "Governance"),
    )


def _cross_path_section(owner: GovernanceResult | None) -> _Section:
    if owner is None:
        return _Section("Cross-path Comparison", marker="not_requested")
    tables = (
        _table(
            "candidate_comparisons", owner.candidate_comparisons, _COMPARISON_COLUMNS
        ),
        _table(
            "governance_evaluations",
            owner.governance_evaluations,
            _GOVERNANCE_EVALUATION_COLUMNS,
        ),
        _table("recommendations", owner.recommendations, _RECOMMENDATION_COLUMNS),
    )
    if all(not table.rows for table in tables):
        return _Section(
            "Cross-path Comparison", marker="empty_result:cross_path_comparison"
        )
    return _Section("Cross-path Comparison", tables=tables)


def _reason_section(
    preloan: DecisionStrategyResult | None,
    postloan: LifecycleMonitoringResult | None,
    governance: GovernanceResult | None,
) -> _Section:
    if preloan is None and postloan is None and governance is None:
        return _Section("Reason and Override Trace", marker="not_requested")
    tables: list[_Table] = []
    if preloan is not None:
        tables.extend(
            (
                _table(
                    "task17.row_decisions", preloan.row_decisions, _ROW_DECISION_COLUMNS
                ),
                _table(
                    "task17.rule_evaluations",
                    preloan.rule_evaluations,
                    _PRELOAN_RULE_COLUMNS,
                ),
            )
        )
    if postloan is not None:
        tables.extend(
            (
                _table(
                    "task18.rule_evaluations",
                    postloan.rule_evaluations,
                    (
                        "row_position",
                        "entity_position",
                        "observation_time",
                        "scenario_key",
                        "scenario_order",
                        "rule_key",
                        "rule_order",
                        "alert_level",
                        "alert_rank",
                        "path_status",
                        "truth",
                        "true_streak",
                        "false_streak",
                        "episode_status",
                        "notification_status",
                        "status",
                        "reason",
                        "finding_key",
                    ),
                ),
                _table(
                    "task18.state_history",
                    postloan.state_history,
                    (
                        "row_position",
                        "entity_position",
                        "observation_time",
                        "candidate_state_key",
                        "candidate_state_rank",
                        "candidate_state_priority",
                        "effective_state_key",
                        "effective_state_rank",
                        "matching_state_count",
                        "status",
                        "reason",
                        "finding_key",
                    ),
                ),
            )
        )
    if governance is not None:
        tables.append(
            _table("task19.explanations", governance.explanations, _EXPLANATION_COLUMNS)
        )
    if all(not table.rows for table in tables):
        return _Section(
            "Reason and Override Trace", marker="empty_result:reason_override_trace"
        )
    return _Section("Reason and Override Trace", tables=tuple(tables))


def _stability_section(
    score: BinaryRiskValidationResult | None,
    preloan: DecisionStrategyResult | None,
    postloan: LifecycleMonitoringResult | None,
    governance: GovernanceResult | None,
) -> _Section:
    if score is None and preloan is None and postloan is None and governance is None:
        return _Section("Stability and Business Evidence", marker="not_requested")
    tables: list[_Table] = []
    if score is not None:
        tables.append(
            _table("task15.business_metrics", score.business_metrics, _BUSINESS_COLUMNS)
        )
    if governance is not None:
        tables.extend(
            (
                _table(
                    "task19.prediction_drift",
                    governance.prediction_drift,
                    _PREDICTION_DRIFT_COLUMNS,
                ),
                _table(
                    "task19.performance_stability",
                    governance.performance_stability,
                    _PERFORMANCE_STABILITY_COLUMNS,
                ),
            )
        )
    if all(not table.rows for table in tables):
        return _Section(
            "Stability and Business Evidence", marker="empty_result:stability_business"
        )
    return _Section("Stability and Business Evidence", tables=tuple(tables))


def _warnings_section(result: V02WorkflowResult) -> _Section:
    fields_: list[tuple[str, str]] = []
    fields_.extend(
        (f"warning_{i + 1:03d}", value) for i, value in enumerate(result.warnings)
    )
    fields_.extend(
        (f"limitation_{i + 1:03d}", value) for i, value in enumerate(result.limitations)
    )
    if not fields_:
        return _Section(
            "Warnings and Limitations", marker="empty_result:warnings_limitations"
        )
    return _Section("Warnings and Limitations", fields=tuple(fields_))


def _provenance_section(
    result: V02WorkflowResult,
    score: BinaryRiskValidationResult | None,
    audit: DataAuditResult | None,
    preloan: DecisionStrategyResult | None,
    postloan: LifecycleMonitoringResult | None,
    governance: GovernanceResult | None,
) -> _Section:
    fields_: list[tuple[str, str]] = [
        ("contract_version", _value_text(result.contract_version)),
        ("call_trace", _tuple_text(result.call_trace)),
    ]
    tables: list[_Table] = []
    if score is not None:
        fields_.extend(
            _fields(
                score, ("score_source", "score_direction", "probability_provenance")
            )
        )
    if audit is not None:
        tables.append(
            _table("task16.provenance", audit.provenance, _AUDIT_PROVENANCE_COLUMNS)
        )
    if preloan is not None:
        tables.append(
            _table("task17.provenance", preloan.provenance, _OWNER_PROVENANCE_COLUMNS)
        )
    if postloan is not None:
        tables.append(
            _table("task18.provenance", postloan.provenance, _OWNER_PROVENANCE_COLUMNS)
        )
    if governance is not None:
        tables.append(
            _table(
                "task19.provenance",
                governance.provenance,
                _GOVERNANCE_PROVENANCE_COLUMNS,
            )
        )
    return _Section(
        "Provenance and Release Readiness", fields=tuple(fields_), tables=tuple(tables)
    )


def _slot_plan(
    score: BinaryRiskValidationResult | None, governance: GovernanceResult | None
) -> tuple[_Slot, ...]:
    slots = (
        _Slot(1, "score_validation_gains", "risk", "gains", "Score Validation", False),
        _Slot(2, "score_validation_lift", "risk", "lift", "Score Validation", False),
        _Slot(
            3,
            "score_validation_calibration",
            "risk",
            "calibration",
            "Score Validation",
            False,
        ),
        _Slot(
            4,
            "score_validation_threshold",
            "risk",
            "threshold",
            "Score Validation",
            False,
        ),
        _Slot(
            5, "governance_importance", "governance", "importance", "Governance", False
        ),
        _Slot(
            6,
            "governance_candidate_comparison",
            "governance",
            "candidate_comparison",
            "Governance",
            False,
        ),
        _Slot(
            7,
            "governance_prediction_drift",
            "governance",
            "prediction_drift",
            "Governance",
            False,
        ),
        _Slot(
            8,
            "governance_performance_stability",
            "governance",
            "performance_stability",
            "Governance",
            False,
        ),
        _Slot(
            9,
            "governance_summary",
            "governance",
            "governance_summary",
            "Governance",
            False,
        ),
    )
    result: list[_Slot] = []
    for slot in slots:
        enabled = False
        if slot.owner == "risk" and score is not None:
            if slot.kind == "gains":
                enabled = _has_gains(score.gains, "capture")
            elif slot.kind == "lift":
                enabled = _has_gains(score.gains, slot.kind)
            elif slot.kind == "calibration":
                enabled = _has_calibration(score.calibration)
            else:
                enabled = _has_threshold(score.threshold_analysis)
        elif slot.owner == "governance" and governance is not None:
            enabled = _has_governance_slot(governance, slot.kind)
        result.append(
            _Slot(slot.ordinal, slot.key, slot.owner, slot.kind, slot.section, enabled)
        )
    return tuple(result)


def _has_gains(table: pd.DataFrame, field: str) -> bool:
    frame = _require_columns(table, _GAINS_PLOT_COLUMNS)
    for row in frame.to_dict("records"):
        if (
            row["scope"] == "overall"
            and _is_missing(row["fold_id"])
            and _finite(row["actual_fraction"])
            and row[f"{field}_status"] == "available"
            and _is_missing(row[f"{field}_reason"])
            and _finite(row[field])
        ):
            return True
    return False


def _has_calibration(table: pd.DataFrame) -> bool:
    frame = _require_columns(table, _CALIBRATION_PLOT_COLUMNS)
    for row in frame.to_dict("records"):
        if (
            row["scope"] == "overall"
            and _finite(row["mean_predicted_probability"])
            and _finite(row["observed_event_rate"])
            and row["status"] == "available"
            and _is_missing(row["reason"])
            and not _is_missing(row["n_rows"])
            and float(row["n_rows"]) > 0
        ):
            return True
    return False


def _has_threshold(table: pd.DataFrame) -> bool:
    frame = _require_columns(table, _THRESHOLD_PLOT_COLUMNS)
    values = ("predicted_positive_rate", "sensitivity", "precision", "specificity")
    for row in frame.to_dict("records"):
        if row["scope"] != "overall" or not _is_missing(row["fold_id"]):
            continue
        if all(
            row[f"{value}_status"] == "available"
            and _is_missing(row[f"{value}_reason"])
            and _finite(row[value])
            for value in values
        ):
            return True
    return False


def _has_governance_slot(owner: GovernanceResult, kind: str) -> bool:
    columns = {
        "importance": (owner.model_attributions, "feature_key", "value"),
        "candidate_comparison": (
            owner.candidate_comparisons,
            "criterion_position",
            "delta",
        ),
        "prediction_drift": (owner.prediction_drift, "scope_key", "prediction_tvd"),
        "performance_stability": (owner.performance_stability, "scope_key", "delta"),
        "governance_summary": (
            owner.governance_summary,
            "candidate_position",
            "available_criterion_count",
        ),
    }[kind]
    frame = _require_columns(columns[0], (columns[1], columns[2], "status"))
    return any(
        row["status"] == "available"
        and not _is_missing(row[columns[1]])
        and _finite(row[columns[2]])
        for row in frame.to_dict("records")
    )


def _acquire_figures(
    result: V02WorkflowResult,
    slots: tuple[_Slot, ...],
    acquired: list[tuple[_Slot, Figure]],
) -> None:
    seen: set[int] = {id(figure) for _, figure in acquired}
    for slot in slots:
        if not slot.enabled:
            continue
        if slot.owner == "risk":
            figure = plot_binary_risk_validation(
                result.score_validation, kind=slot.kind
            )  # type: ignore[arg-type]
        else:
            figure = plot_model_governance(result.governance, kind=slot.kind)  # type: ignore[arg-type]
        if type(figure) is not Figure or id(figure) in seen:
            _report_error("result")
        seen.add(id(figure))
        acquired.append((slot, figure))


def _render(model: _ReportModel, format: str, asset_name: str) -> str:
    if format == "markdown":
        blocks = [f"# {_markdown(model.title)}"]
        blocks.extend(
            _render_markdown_section(section, asset_name) for section in model.sections
        )
        return "\n\n".join(blocks).rstrip("\n") + "\n"
    content = [
        "<!doctype html>",
        "<html><body>",
        f"<h1>{html.escape(model.title, quote=True)}</h1>",
    ]
    for section in model.sections:
        content.append(_render_html_section(section, asset_name))
    content.extend(("</body></html>", ""))
    return "\n".join(content)


def _render_markdown_section(section: _Section, asset_name: str) -> str:
    parts = [f"## {_markdown(section.title)}"]
    if section.marker is not None:
        parts.append(_markdown(section.marker))
        return "\n\n".join(parts)
    if section.fields:
        parts.append(_markdown_table(("field", "value"), section.fields))
    parts.extend(_markdown_table(table.columns, table.rows) for table in section.tables)
    if section.slots:
        parts.append(_markdown_slots(section.slots, asset_name))
    return "\n\n".join(parts)


def _render_html_section(section: _Section, asset_name: str) -> str:
    parts = [f"<h2>{html.escape(section.title, quote=True)}</h2>"]
    if section.marker is not None:
        parts.append(f"<p>{html.escape(section.marker, quote=True)}</p>")
        return "\n".join(parts)
    if section.fields:
        parts.append(_html_table(("field", "value"), section.fields))
    parts.extend(_html_table(table.columns, table.rows) for table in section.tables)
    if section.slots:
        parts.append(_html_slots(section.slots, asset_name))
    return "\n".join(parts)


def _markdown_table(columns: tuple[str, ...], rows: object) -> str:
    row_values = tuple(rows)
    lines = [
        "| " + " | ".join(_markdown(value) for value in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_markdown(value) for value in row) + " |"
        for row in row_values
    )
    return "\n".join(lines)


def _html_table(columns: tuple[str, ...], rows: object) -> str:
    row_values = tuple(rows)
    head = "".join(
        f"<th>{html.escape(str(value), quote=True)}</th>" for value in columns
    )
    data = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(value), quote=True)}</td>" for value in row)
        + "</tr>"
        for row in row_values
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{data}</tbody></table>"


def _markdown_slots(slots: tuple[_Slot, ...], asset_name: str) -> str:
    lines: list[str] = []
    for slot in slots:
        if slot.enabled:
            path = f"{asset_name}/plot-{slot.ordinal:03d}.png"
            lines.append(f"![{_markdown(slot.key)}]({_markdown(path)})")
        else:
            lines.append(_markdown(f"plot_unavailable:{slot.key}"))
    return "\n".join(lines)


def _html_slots(slots: tuple[_Slot, ...], asset_name: str) -> str:
    lines: list[str] = []
    for slot in slots:
        if slot.enabled:
            path = f"{asset_name}/plot-{slot.ordinal:03d}.png"
            lines.append(
                f'<img src="{html.escape(path, quote=True)}" '
                f'alt="{html.escape(slot.key, quote=True)}">'
            )
        else:
            lines.append(
                f"<p>{html.escape(f'plot_unavailable:{slot.key}', quote=True)}</p>"
            )
    return "\n".join(lines)


def _markdown(value: object) -> str:
    text = str(value)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("\\", "\\\\").replace("|", "\\|").replace("`", "\\`")
    return "".join(
        " " if ord(character) < 32 or ord(character) == 127 else character
        for character in text
    )


def _value_text(value: object) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (str, int, float, np.integer, np.floating)):
        return str(value)
    if isinstance(value, (datetime, date, time, timedelta, pd.Timestamp, pd.Timedelta)):
        return str(value)
    _report_error("result")


def _tuple_text(value: tuple[str, ...]) -> str:
    return "[" + ", ".join(_value_text(item) for item in value) + "]"


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _finite(value: object) -> bool:
    if _is_missing(value) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError, OverflowError):
        return False


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _restore(source: Path, destination: Path) -> None:
    if source.exists():
        source.replace(destination)


def _clean_staging(report: Path, assets: Path) -> None:
    _remove(report)
    _remove(assets)
