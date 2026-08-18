"""Smoke tests for the public import contract."""

import inspect
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import MISSING, fields, is_dataclass
from datetime import date, datetime, timedelta
from importlib.metadata import entry_points
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, get_type_hints

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.base import ClassifierMixin, RegressorMixin
from typer.testing import CliRunner

import sharper
from sharper.cli import app

_V01_EXPORTS = (
    "__version__",
    "load_csv",
    "load_excel",
    "ColumnSchema",
    "TargetCandidate",
    "SchemaReport",
    "infer_schema",
    "DataFrameSummary",
    "summarize_dataframe",
    "QualityIssue",
    "QualityReport",
    "check_data_quality",
    "AnalysisRun",
    "run_analysis",
    "ReportArtifact",
    "generate_analysis_report",
    "NumericAnalysis",
    "CategoricalAnalysis",
    "CorrelationAnalysis",
    "OutlierAnalysis",
    "analyze_numeric_features",
    "analyze_categorical_features",
    "compute_correlations",
    "detect_outliers",
    "GroupComparison",
    "TargetAnalysis",
    "compare_groups",
    "analyze_target_relationships",
    "FeatureSuggestion",
    "FeatureSuggestionReport",
    "FeatureDerivationResult",
    "suggest_feature_derivations",
    "derive_features",
    "TrainingResult",
    "train_classifier",
    "RegressionTrainingResult",
    "train_regressor",
    "ClassificationEvaluation",
    "evaluate_classifier",
    "RegressionEvaluation",
    "evaluate_regressor",
    "evaluate_model",
    "PlotResult",
    "PlotCollection",
    "plot_distributions",
    "plot_missingness",
    "plot_correlations",
    "plot_outliers",
    "plot_group_comparison",
    "plot_target_relationships",
    "plot_classification_evaluation",
    "plot_regression_evaluation",
)
_TASK15_EXPORTS = (
    "BinaryRiskValidationConfig",
    "ExternalRiskPredictions",
    "BinaryRiskValidationResult",
    "validate_binary_risk",
    "plot_binary_risk_validation",
)
_TASK16_EXPORTS = (
    "DataAuditRoles",
    "ColumnAuditRule",
    "DataAuditConfig",
    "DataAuditResult",
    "audit_data_quality",
)
_TASK17_EXPORTS = (
    "StrategyCondition",
    "DecisionRule",
    "DecisionConstraint",
    "DecisionStrategyConfig",
    "DecisionStrategyResult",
    "simulate_decision_strategy",
)
_TASK18_EXPORTS = (
    "MonitoringCondition",
    "EarlyWarningRule",
    "WarningScenario",
    "LifecycleState",
    "LifecycleMonitoringConfig",
    "LifecycleMonitoringResult",
    "monitor_lifecycle",
)
_TASK19_EXPORTS = (
    "GovernanceEvidenceRef",
    "GovernanceCandidate",
    "GovernanceCriterion",
    "GovernanceExplanation",
    "GovernanceAttributionEvidence",
    "GovernancePredictionProfile",
    "GovernancePerformanceEvidence",
    "GovernanceMetadata",
    "GovernancePolicy",
    "GovernanceResult",
    "evaluate_governance",
    "plot_model_governance",
)
_TASK20_EXPORTS = (
    "V02ScoreValidationRequest",
    "V02AuditRequest",
    "V02PreLoanRequest",
    "V02PostLoanRequest",
    "V02GovernanceRequest",
    "V02WorkflowRequest",
    "V02WorkflowResult",
    "run_v02_workflow",
    "generate_v02_report",
)


def test_task18_public_api_contract() -> None:
    """Task 18 appends exactly its frozen seven-symbol public suffix."""
    expected_fields = {
        sharper.MonitoringCondition: (
            "kind",
            "operator",
            "left_kind",
            "left",
            "right_kind",
            "right",
            "window",
            "children",
        ),
        sharper.EarlyWarningRule: (
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
        ),
        sharper.WarningScenario: ("scenario_key", "scenario_kind", "rules"),
        sharper.LifecycleState: (
            "state_key",
            "state_rank",
            "priority",
            "condition",
            "terminal",
            "enabled",
            "description_key",
        ),
    }
    for result_type, names in expected_fields.items():
        assert is_dataclass(result_type)
        assert result_type.__dataclass_params__.frozen is True
        assert tuple(field.name for field in fields(result_type)) == names
        assert result_type.__doc__
    assert str(inspect.signature(sharper.monitor_lifecycle)) == (
        "(data: 'pd.DataFrame', config: 'LifecycleMonitoringConfig', *, "
        "risk_validation: 'BinaryRiskValidationResult | None' = None, "
        "data_audit: 'DataAuditResult | None' = None) -> 'LifecycleMonitoringResult'"
    )


def test_version_contract() -> None:
    """The package exposes the approved Task 20 target version."""
    assert sharper.__version__ == "0.2.0"


def test_console_entry_point_contract() -> None:
    """The sole frozen console entry point targets the public CLI app."""
    matches = [
        entry
        for entry in entry_points(group="console_scripts")
        if entry.name == "sharper"
    ]
    assert len(matches) == 1
    assert matches[0].value == "sharper.cli:app"


def test_all_contains_only_implemented_public_api() -> None:
    """The v0.1 prefix stays stable and Task 15 is one separate opt-in suffix."""
    assert tuple(sharper.__all__[: len(_V01_EXPORTS)]) == _V01_EXPORTS
    task15_end = len(_V01_EXPORTS) + len(_TASK15_EXPORTS)
    assert tuple(sharper.__all__[len(_V01_EXPORTS) : task15_end]) == _TASK15_EXPORTS
    task16_end = task15_end + len(_TASK16_EXPORTS)
    assert tuple(sharper.__all__[task15_end:task16_end]) == _TASK16_EXPORTS
    task17_end = task16_end + len(_TASK17_EXPORTS)
    assert tuple(sharper.__all__[task16_end:task17_end]) == _TASK17_EXPORTS
    task18_end = task17_end + len(_TASK18_EXPORTS)
    assert tuple(sharper.__all__[task17_end:task18_end]) == _TASK18_EXPORTS
    task19_end = task18_end + len(_TASK19_EXPORTS)
    assert tuple(sharper.__all__[task18_end:task19_end]) == _TASK19_EXPORTS
    assert tuple(sharper.__all__[task19_end:]) == _TASK20_EXPORTS
    assert all(not name.startswith("_types") for name in sharper.__all__)


def test_v02_public_symbols_signatures_and_fields() -> None:
    """Task 20 exposes exactly the frozen carrier fields and function signatures."""
    expected_fields = {
        sharper.V02ScoreValidationRequest: (
            "target",
            "config",
            "positive_label",
            "estimator",
            "external_predictions",
            "features",
            "exclude_columns",
        ),
        sharper.V02AuditRequest: ("reference", "roles", "config"),
        sharper.V02PreLoanRequest: ("config",),
        sharper.V02PostLoanRequest: ("config",),
        sharper.V02GovernanceRequest: (
            "policy",
            "model_attributions",
            "prediction_profiles",
            "performance_evidence",
        ),
        sharper.V02WorkflowRequest: (
            "data",
            "score_validation",
            "audit",
            "preloan",
            "postloan",
            "governance",
        ),
        sharper.V02WorkflowResult: (
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
    for data_type, names in expected_fields.items():
        assert is_dataclass(data_type)
        assert data_type.__dataclass_params__.frozen is True
        assert tuple(field.name for field in fields(data_type)) == names
        assert data_type.__doc__
    assert str(inspect.signature(sharper.run_v02_workflow)) == (
        "(request: 'V02WorkflowRequest') -> 'V02WorkflowResult'"
    )
    assert str(inspect.signature(sharper.generate_v02_report)) == (
        "(result: 'V02WorkflowResult', output_path: 'str | Path', *, "
        "title: 'str' = 'Sharper v0.2 Integration Report', "
        "format: \"Literal['markdown', 'html']\" = 'markdown', "
        "overwrite: 'bool' = True) -> 'ReportArtifact'"
    )


def test_current_release_surface_appends_task20() -> None:
    """The current surface is the permanent prefix plus the exact Task 20 suffix."""
    assert tuple(sharper.__all__[-len(_TASK20_EXPORTS) :]) == _TASK20_EXPORTS
    assert all(hasattr(sharper, name) for name in _TASK20_EXPORTS)
    assert len(sharper.__all__) == sum(
        len(exports)
        for exports in (
            _V01_EXPORTS,
            _TASK15_EXPORTS,
            _TASK16_EXPORTS,
            _TASK17_EXPORTS,
            _TASK18_EXPORTS,
            _TASK19_EXPORTS,
            _TASK20_EXPORTS,
        )
    )


def test_v02_version_transition_gate() -> None:
    """The source version and root CLI report the same approved target."""
    result = CliRunner().invoke(app, ["--version"])
    assert sharper.__version__ == "0.2.0"
    assert result.exit_code == 0
    assert result.stdout == "sharper 0.2.0\n"


def test_v02_examples_and_documentation_smoke() -> None:
    """The five examples run externally and the six authorized docs are current."""
    root = Path(__file__).parents[1]
    examples = (
        "v02_score_validation.py",
        "v02_preloan.py",
        "v02_postloan.py",
        "v02_combined_report.py",
        "v02_cli_json.py",
    )
    with TemporaryDirectory(prefix="sharper-v02-public-") as directory:
        environment = os.environ.copy()
        environment["MPLCONFIGDIR"] = str(Path(directory) / "mpl")
        for name in examples:
            result = subprocess.run(
                [sys.executable, str(root / "examples" / name)],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, result.stderr or result.stdout
            assert "Traceback" not in result.stderr
    required_docs = (
        "README.md",
        "docs/quickstart.md",
        "docs/analysis-guide.md",
        "docs/api.md",
        "docs/v02-integration-guide.md",
        "docs/release-readiness.md",
    )
    for name in required_docs:
        content = (root / name).read_text()
        assert content
        assert "0.2.0" in content
    assert (
        "Release Ready — Not Released"
        in (root / "docs/release-readiness.md").read_text()
    )


def test_task18_api_documentation_matches_frozen_inventory() -> None:
    """Task 18 API docs retain the export, table, scope, and schema contract."""
    api = (Path(__file__).parents[1] / "docs" / "api.md").read_text()
    for symbol in _TASK18_EXPORTS:
        assert f"`{symbol}`" in api
    for table in (
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
    ):
        assert f"`{table}`" in api
    for column in (
        "reference_scenario_key",
        "comparator_scenario_key",
        "scope_key",
        "scope_position",
        "rule_key",
        "finding_key",
    ):
        assert f"`{column}`" in api
    for scope in (
        "scenario",
        "scenario_rule",
        "scenario_alert_level",
        "scenario_segment",
        "scenario_time",
        "scenario_cohort",
        "scenario_vintage",
        "scenario_state",
        "scenario_transition",
    ):
        assert f"`{scope}`" in api
    for scope in ("overall", "segment_time", "cohort_time", "vintage_state"):
        assert f"`{scope}`" in api
    assert "monitoring_summary_rows" in api
    assert "lifecycle_summary_rows" in api
    assert "scenario_comparison_rows" in api
    assert "200,000" in api


def test_task16_public_api_contract() -> None:
    """Task 16 appends exactly five opt-in public symbols."""
    assert str(inspect.signature(sharper.audit_data_quality)) == (
        "(data: 'pd.DataFrame', *, reference: 'pd.DataFrame | None' = None, "
        "roles: 'DataAuditRoles | None' = None, "
        "config: 'DataAuditConfig | None' = None) "
        "-> 'DataAuditResult'"
    )
    for data_type in (
        sharper.DataAuditRoles,
        sharper.ColumnAuditRule,
        sharper.DataAuditConfig,
        sharper.DataAuditResult,
    ):
        assert is_dataclass(data_type)
        assert data_type.__dataclass_params__.frozen is True
    assert not any(name.startswith("_Condition") for name in sharper.__all__)


def test_task17_public_api_contract() -> None:
    """Task 17 appends exactly six opt-in symbols after the Task 16 suffix."""
    assert str(inspect.signature(sharper.simulate_decision_strategy)) == (
        "(data: 'pd.DataFrame', config: 'DecisionStrategyConfig', *, "
        "risk_validation: 'BinaryRiskValidationResult | None' = None, "
        "data_audit: 'DataAuditResult | None' = None) -> 'DecisionStrategyResult'"
    )
    expected_fields = {
        sharper.StrategyCondition: (
            "kind",
            "operator",
            "left_kind",
            "left",
            "right_kind",
            "right",
            "children",
        ),
        sharper.DecisionRule: (
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
        ),
        sharper.DecisionConstraint: (
            "constraint_key",
            "metric",
            "operator",
            "threshold",
            "action_name",
            "action_role",
            "minimum_support",
        ),
        sharper.DecisionStrategyConfig: (
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
        ),
        sharper.DecisionStrategyResult: (
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
        ),
    }
    for data_type, names in expected_fields.items():
        assert is_dataclass(data_type)
        assert data_type.__dataclass_params__.frozen is True
        assert tuple(field.name for field in fields(data_type)) == names
    assert not hasattr(sharper, "_compile_condition")
    assert not any(name.startswith("_Condition") for name in sharper.__all__)


def test_task17_public_dataclass_docstrings_cover_contract_boundaries() -> None:
    """Task 17 public containers document every field and safety boundary."""
    expected = {
        sharper.StrategyCondition: {
            "kind": ("Required.",),
            "operator": ("Default: ``None``.", "Boolean nodes"),
            "left_kind": ("Default: ``None``.", "request no left source"),
            "left": ("Default: ``None``.", "have no column name"),
            "right_kind": ("Default: ``None``.", "have no right source"),
            "right": ("Default: ``None``.", "No right operand"),
            "children": ("Default: ``()``.", "No child conditions"),
        },
        sharper.DecisionRule: {
            "rule_key": ("Required.",),
            "phase": ("Required.",),
            "priority": ("Required.",),
            "condition": ("Required.",),
            "action_name": ("Required.",),
            "stop_on_hit": ("Default: ``True``.", "stops later rule application"),
            "enabled": ("Default: ``True``.", "participates"),
            "effective_from": ("Default: ``None``.", "inherits the strategy start"),
            "expires_at": ("Default: ``None``.", "inherits the strategy expiry"),
            "description_key": ("Default: ``None``.", "No safe description"),
        },
        sharper.DecisionConstraint: {
            "constraint_key": ("Required.",),
            "metric": ("Required.",),
            "operator": ("Required.",),
            "threshold": ("Required.",),
            "action_name": ("Default: ``None``.", "No action-key scope"),
            "action_role": ("Default: ``None``.", "No closed-role scope"),
            "minimum_support": ("Default: ``1``.", "supporting row"),
        },
        sharper.DecisionStrategyConfig: {
            "strategy_key": ("Required.",),
            "strategy_version": ("Required.",),
            "effective_from": ("Required.",),
            "expires_at": ("Required.", "``None`` means", "no exclusive expiry"),
            "evaluation_time": ("Required.",),
            "rules": ("Required.", "``()``", "no rules"),
            "default_action_name": ("Required.",),
            "unknown_action_name": ("Required.",),
            "action_role_mapping": ("Required.",),
            "constraints": ("Default: ``()``.", "No evidence-only constraints"),
            "ranking_score_column": (
                "Default: ``None``.",
                "No DataFrame ranking-score column",
            ),
            "ranking_score_direction": (
                "Default: ``None``.",
                "No DataFrame ranking direction",
            ),
            "historical_action_column": (
                "Default: ``None``.",
                "No historical action column",
            ),
            "historical_action_mapping": (
                "Default: ``()``.",
                "No historical raw-value-to-action mappings",
            ),
            "historical_policy_version": (
                "Default: ``None``.",
                "No sanitized historical policy version",
            ),
            "exposure_column": (
                "Default: ``None``.",
                "No row-level exposure evidence",
            ),
            "loss_fraction": (
                "Default: ``None``.",
                "No constant or DataFrame loss-fraction evidence",
            ),
            "action_assumptions": (
                "Default: ``()``.",
                "No action value or cost assumptions",
            ),
            "exposure_unit": (
                "Default: ``None``.",
                "No common opaque exposure/loss unit",
            ),
            "segment_columns": (
                "Default: ``()``.",
                "No segment or segment-time stability scopes",
            ),
            "time_slice_column": (
                "Default: ``None``.",
                "No time-slice or segment-time scopes",
            ),
        },
        sharper.DecisionStrategyResult: {
            "strategy_key": ("Required.",),
            "strategy_version": ("Required.",),
            "strategy_fingerprint": ("Required.",),
            "input_n_rows": ("Required.",),
            "decided_n_rows": ("Required.",),
            "unavailable_n_rows": ("Required.",),
            "requested_rule_count": ("Required.",),
            "active_rule_count": ("Required.",),
            "requested_constraint_count": ("Required.",),
            "row_decisions": ("Required.",),
            "rule_evaluations": ("Required.",),
            "rule_summary": ("Required.",),
            "action_summary": ("Required.",),
            "business_summary": ("Required.",),
            "constraint_summary": ("Required.",),
            "historical_transitions": ("Required.",),
            "provenance": ("Required.",),
            "warnings": ("Required.", "``()`` means no warning"),
            "limitations": ("Required.", "``()`` means no limitation"),
        },
    }
    expected_defaults = {
        sharper.StrategyCondition: {
            "operator": None,
            "left_kind": None,
            "left": None,
            "right_kind": None,
            "right": None,
            "children": (),
        },
        sharper.DecisionRule: {
            "stop_on_hit": True,
            "enabled": True,
            "effective_from": None,
            "expires_at": None,
            "description_key": None,
        },
        sharper.DecisionConstraint: {
            "action_name": None,
            "action_role": None,
            "minimum_support": 1,
        },
        sharper.DecisionStrategyConfig: {
            "constraints": (),
            "ranking_score_column": None,
            "ranking_score_direction": None,
            "historical_action_column": None,
            "historical_action_mapping": (),
            "historical_policy_version": None,
            "exposure_column": None,
            "loss_fraction": None,
            "action_assumptions": (),
            "exposure_unit": None,
            "segment_columns": (),
            "time_slice_column": None,
        },
        sharper.DecisionStrategyResult: {},
    }
    forbidden = (
        "production approval execution",
        "automatic optimization",
        "event_probability_column",
        "task 18",
        "task 19",
        "task 20",
    )
    for data_type, expected_fields in expected.items():
        doc = inspect.getdoc(data_type)
        assert doc is not None
        lowered = doc.lower()
        for section in (
            "summary",
            "attributes",
            "validation / errors",
            "missing / unavailable behavior",
            "side effects / immutability",
            "example",
        ):
            assert section in lowered
        attributes = doc.split("Attributes\n----------", maxsplit=1)[1].split(
            "Validation / Errors", maxsplit=1
        )[0]
        entries: dict[str, list[str]] = {}
        current: str | None = None
        for line in attributes.splitlines():
            if line and not line.startswith(" "):
                current = line.strip()
                entries[current] = []
            elif current is not None and line.strip():
                entries[current].append(line.strip())
        runtime_fields = {field.name: field for field in fields(data_type)}
        actual_fields = set(runtime_fields)
        assert set(expected_fields) == actual_fields
        assert set(entries) == actual_fields
        assert set(expected_defaults[data_type]) <= actual_fields
        for field_name, tokens in expected_fields.items():
            field_doc = " ".join(entries[field_name])
            assert all(token in field_doc for token in tokens)
            if field_name not in expected_defaults[data_type]:
                assert runtime_fields[field_name].default is MISSING
                assert "Default:" not in field_doc
            else:
                assert (
                    runtime_fields[field_name].default
                    == expected_defaults[data_type][field_name]
                )
                assert "Required." not in field_doc
        assert "validat" in lowered
        assert "frozen" in lowered
        assert not any(claim in lowered for claim in forbidden)


def test_task15_public_api_signatures_fields_and_export_order() -> None:
    config_fields = [
        "validation_mode",
        "n_splits",
        "test_size",
        "random_state",
        "group_column",
        "observation_time_column",
        "event_time_column",
        "outcome_end_time_column",
        "label_available_time_column",
        "maturity_source",
        "prediction_horizon",
        "prediction_horizon_column",
        "reporting_delay",
        "fold_cutoffs",
        "validation_end",
        "analysis_as_of",
        "thresholds",
        "threshold_kind",
        "operating_metric",
        "calibration_bins",
        "gain_fractions",
        "exposure_column",
        "observed_loss_column",
        "observed_loss_available_time_column",
        "observed_loss_is_mature_snapshot",
        "loss_fraction",
        "exposure_unit",
    ]
    external_fields = [
        "row_positions",
        "fold_ids",
        "fold_fit_row_positions",
        "ranking_scores",
        "ranking_direction",
        "event_probabilities",
        "probability_positive_label",
        "probability_provenance",
    ]
    result_fields = [
        "target",
        "positive_label",
        "validation_mode",
        "config",
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
        "observed_loss_analysis_as_of",
        "observed_loss_mature_n",
        "observed_loss_excluded_n",
        "folds",
        "predictions",
        "excluded_rows",
        "metrics",
        "gains",
        "calibration",
        "threshold_analysis",
        "operating_point",
        "business_metrics",
        "warnings",
        "limitations",
    ]
    for data_type, names in (
        (sharper.BinaryRiskValidationConfig, config_fields),
        (sharper.ExternalRiskPredictions, external_fields),
        (sharper.BinaryRiskValidationResult, result_fields),
    ):
        assert is_dataclass(data_type)
        assert data_type.__dataclass_params__.frozen is True
        assert [field.name for field in fields(data_type)] == names
        assert data_type.__doc__
        assert "shallow frozen" in data_type.__doc__.lower()

    mode = Literal[
        "stratified_holdout",
        "stratified_kfold",
        "group_holdout",
        "group_kfold",
        "time_holdout",
        "time_forward",
    ]
    config_hints = {
        "validation_mode": mode,
        "n_splits": int | None,
        "test_size": float | None,
        "random_state": int,
        "group_column": str | None,
        "observation_time_column": str | None,
        "event_time_column": str | None,
        "outcome_end_time_column": str | None,
        "label_available_time_column": str | None,
        "maturity_source": Literal[
            "label_available_time", "observation_horizon", "outcome_end"
        ]
        | None,
        "prediction_horizon": timedelta | None,
        "prediction_horizon_column": str | None,
        "reporting_delay": timedelta,
        "fold_cutoffs": tuple[datetime, ...],
        "validation_end": datetime | None,
        "analysis_as_of": datetime | None,
        "thresholds": tuple[float, ...],
        "threshold_kind": Literal["ranking_score", "event_probability"] | None,
        "operating_metric": Literal[
            "sensitivity", "specificity", "precision", "negative_predictive_value", "f1"
        ]
        | None,
        "calibration_bins": int,
        "gain_fractions": tuple[float, ...],
        "exposure_column": str | None,
        "observed_loss_column": str | None,
        "observed_loss_available_time_column": str | None,
        "observed_loss_is_mature_snapshot": bool,
        "loss_fraction": float | str | None,
        "exposure_unit": str | None,
    }
    external_hints = {
        "row_positions": tuple[int, ...],
        "fold_ids": tuple[int, ...],
        "fold_fit_row_positions": tuple[tuple[int, tuple[int, ...]], ...],
        "ranking_scores": tuple[float, ...] | None,
        "ranking_direction": Literal["higher_risk", "lower_risk"] | None,
        "event_probabilities": tuple[float, ...] | None,
        "probability_positive_label": str | int | bool | np.generic | None,
        "probability_provenance": Literal[
            "predict_proba", "fold_safe_calibrated", "external_declared"
        ]
        | None,
    }
    result_hints = {
        "target": str,
        "positive_label": str | int | bool,
        "validation_mode": mode,
        "config": sharper.BinaryRiskValidationConfig,
        "prediction_scope": Literal["validation", "oof"],
        "score_source": Literal[
            "estimator_predict_proba",
            "estimator_decision_function",
            "external_ranking_score",
            "external_event_probability",
            "external_ranking_and_probability",
        ],
        "score_direction": Literal["higher_positive_event_risk"],
        "probability_provenance": Literal[
            "predict_proba", "fold_safe_calibrated", "external_declared"
        ]
        | None,
        "input_n_rows": int,
        "eligible_n_rows": int,
        "predicted_n_rows": int,
        "evaluable_n_rows": int,
        "requested_threshold_count": int,
        "actual_threshold_count": int,
        "observed_loss_maturity_mode": Literal[
            "not_provided", "availability_column", "mature_snapshot"
        ],
        "observed_loss_analysis_as_of": datetime | None,
        "observed_loss_mature_n": int,
        "observed_loss_excluded_n": int,
        "folds": pd.DataFrame,
        "predictions": pd.DataFrame,
        "excluded_rows": pd.DataFrame,
        "metrics": pd.DataFrame,
        "gains": pd.DataFrame,
        "calibration": pd.DataFrame,
        "threshold_analysis": pd.DataFrame,
        "operating_point": pd.DataFrame,
        "business_metrics": pd.DataFrame,
        "warnings": tuple[str, ...],
        "limitations": tuple[str, ...],
    }
    assert get_type_hints(sharper.BinaryRiskValidationConfig) == config_hints
    assert get_type_hints(sharper.ExternalRiskPredictions) == external_hints
    assert get_type_hints(sharper.BinaryRiskValidationResult) == result_hints

    signature = inspect.signature(sharper.validate_binary_risk)
    assert list(signature.parameters) == [
        "df",
        "target",
        "positive_label",
        "config",
        "estimator",
        "external_predictions",
        "features",
        "exclude_columns",
    ]
    assert signature.parameters["positive_label"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["config"].kind is inspect.Parameter.KEYWORD_ONLY
    hints = get_type_hints(sharper.validate_binary_risk)
    assert hints == {
        "df": pd.DataFrame,
        "target": str,
        "positive_label": str | int | bool | np.generic | None,
        "config": sharper.BinaryRiskValidationConfig,
        "estimator": ClassifierMixin | None,
        "external_predictions": sharper.ExternalRiskPredictions | None,
        "features": Sequence[str] | None,
        "exclude_columns": Sequence[str],
        "return": sharper.BinaryRiskValidationResult,
    }
    plot_signature = inspect.signature(sharper.plot_binary_risk_validation)
    assert list(plot_signature.parameters) == ["result", "kind"]
    assert plot_signature.parameters["kind"].kind is inspect.Parameter.KEYWORD_ONLY
    assert get_type_hints(sharper.plot_binary_risk_validation) == {
        "result": sharper.BinaryRiskValidationResult,
        "kind": Literal["gains", "lift", "calibration", "threshold"],
        "return": Figure,
    }
    assert sharper.validate_binary_risk.__doc__
    assert sharper.plot_binary_risk_validation.__doc__
    task15_start = len(_V01_EXPORTS)
    assert (
        tuple(sharper.__all__[task15_start : task15_start + len(_TASK15_EXPORTS)])
        == _TASK15_EXPORTS
    )


def test_load_csv_public_contract() -> None:
    """The public CSV loader has the documented signature and typing."""
    signature = inspect.signature(sharper.load_csv)
    assert list(signature.parameters) == ["path", "read_options"]
    assert signature.parameters["read_options"].kind is inspect.Parameter.VAR_KEYWORD

    hints = get_type_hints(sharper.load_csv)
    assert hints == {
        "path": str | Path,
        "read_options": Any,
        "return": pd.DataFrame,
    }
    assert sharper.load_csv.__doc__


def test_load_excel_public_contract() -> None:
    """The public Excel loader has the documented signature and typing."""
    signature = inspect.signature(sharper.load_excel)
    assert list(signature.parameters) == ["path", "sheet_name", "read_options"]
    assert signature.parameters["sheet_name"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["sheet_name"].default == 0
    assert signature.parameters["read_options"].kind is inspect.Parameter.VAR_KEYWORD

    hints = get_type_hints(sharper.load_excel)
    assert hints == {
        "path": str | Path,
        "sheet_name": str | int,
        "read_options": Any,
        "return": pd.DataFrame,
    }
    assert sharper.load_excel.__doc__


def test_task03_function_signatures_and_typing() -> None:
    """Schema and summary functions expose their frozen signatures."""
    schema_signature = inspect.signature(sharper.infer_schema)
    assert list(schema_signature.parameters) == ["df", "target", "id_threshold"]
    assert schema_signature.parameters["target"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        schema_signature.parameters["id_threshold"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert get_type_hints(sharper.infer_schema) == {
        "df": pd.DataFrame,
        "target": str | None,
        "id_threshold": float,
        "return": sharper.SchemaReport,
    }

    summary_signature = inspect.signature(sharper.summarize_dataframe)
    assert list(summary_signature.parameters) == ["df", "schema"]
    assert summary_signature.parameters["schema"].kind is inspect.Parameter.KEYWORD_ONLY
    assert get_type_hints(sharper.summarize_dataframe) == {
        "df": pd.DataFrame,
        "schema": sharper.SchemaReport | None,
        "return": sharper.DataFrameSummary,
    }
    assert sharper.infer_schema.__doc__
    assert sharper.summarize_dataframe.__doc__


def test_task03_dataclass_fields_are_frozen() -> None:
    """Public result dataclasses contain exactly the approved fields."""
    contracts = {
        sharper.ColumnSchema: [
            "name",
            "pandas_dtype",
            "logical_type",
            "nullable",
            "missing_count",
            "missing_rate",
            "unique_count",
            "unique_rate",
            "is_constant",
            "is_id_like",
            "confidence",
            "reasons",
        ],
        sharper.TargetCandidate: [
            "name",
            "suggested_task_type",
            "confidence",
            "reasons",
        ],
        sharper.SchemaReport: [
            "n_rows",
            "n_columns",
            "columns",
            "logical_type_counts",
            "target_candidates",
        ],
        sharper.DataFrameSummary: [
            "n_rows",
            "n_columns",
            "memory_usage_bytes",
            "total_missing_cells",
            "total_missing_rate",
            "schema",
            "column_summary",
        ],
    }

    for result_type, expected_fields in contracts.items():
        assert is_dataclass(result_type)
        assert result_type.__dataclass_params__.frozen is True
        assert [field.name for field in fields(result_type)] == expected_fields
        assert result_type.__doc__


def test_task04_function_signature_and_typing() -> None:
    """The quality function exposes its frozen signature and type hints."""
    signature = inspect.signature(sharper.check_data_quality)
    assert list(signature.parameters) == ["df", "schema", "missing_threshold"]
    assert signature.parameters["schema"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        signature.parameters["missing_threshold"].kind is inspect.Parameter.KEYWORD_ONLY
    )
    assert signature.parameters["schema"].default is None
    assert signature.parameters["missing_threshold"].default == 0.40
    assert get_type_hints(sharper.check_data_quality) == {
        "df": pd.DataFrame,
        "schema": sharper.SchemaReport | None,
        "missing_threshold": float,
        "return": sharper.QualityReport,
    }
    assert sharper.check_data_quality.__doc__


def test_task04_dataclass_fields_are_frozen() -> None:
    """Quality result dataclasses contain exactly the approved fields."""
    contracts = {
        sharper.QualityIssue: [
            "code",
            "severity",
            "scope",
            "column",
            "count",
            "ratio",
            "threshold",
            "message",
            "suggestion",
        ],
        sharper.QualityReport: [
            "n_rows",
            "n_columns",
            "issue_count",
            "severity_counts",
            "issues",
        ],
    }

    for result_type, expected_fields in contracts.items():
        assert is_dataclass(result_type)
        assert result_type.__dataclass_params__.frozen is True
        assert [field.name for field in fields(result_type)] == expected_fields
        assert result_type.__doc__


def test_task13_function_signatures_and_typing() -> None:
    """Workflow and reporting functions expose their frozen signatures."""
    run_signature = inspect.signature(sharper.run_analysis)
    assert list(run_signature.parameters) == [
        "df",
        "target",
        "task",
        "include_model",
        "id_columns",
        "exclude_columns",
        "features",
        "time_column",
        "group_by",
        "reference_date",
        "max_suggestions",
        "test_size",
        "random_state",
    ]
    assert get_type_hints(sharper.run_analysis)["return"] is sharper.AnalysisRun

    report_signature = inspect.signature(sharper.generate_analysis_report)
    assert list(report_signature.parameters) == [
        "run",
        "output_path",
        "title",
        "format",
        "overwrite",
    ]
    assert get_type_hints(sharper.generate_analysis_report) == {
        "run": sharper.AnalysisRun,
        "output_path": str | Path,
        "title": str,
        "format": Literal["markdown", "html"],
        "overwrite": bool,
        "return": sharper.ReportArtifact,
    }
    assert sharper.run_analysis.__doc__
    assert sharper.generate_analysis_report.__doc__


def test_task07_function_signatures_and_typing() -> None:
    """Non-target analysis functions expose their frozen signatures and hints."""
    contracts = {
        sharper.analyze_numeric_features: (
            ["df", "columns"],
            {"columns": None},
            {
                "df": pd.DataFrame,
                "columns": Sequence[str] | None,
                "return": sharper.NumericAnalysis,
            },
        ),
        sharper.analyze_categorical_features: (
            ["df", "columns", "top_n"],
            {"columns": None, "top_n": 10},
            {
                "df": pd.DataFrame,
                "columns": Sequence[str] | None,
                "top_n": int,
                "return": sharper.CategoricalAnalysis,
            },
        ),
        sharper.compute_correlations: (
            ["df", "columns", "method", "max_columns", "min_periods"],
            {
                "columns": None,
                "method": "pearson",
                "max_columns": 50,
                "min_periods": 2,
            },
            {
                "df": pd.DataFrame,
                "columns": Sequence[str] | None,
                "method": str,
                "max_columns": int,
                "min_periods": int,
                "return": sharper.CorrelationAnalysis,
            },
        ),
        sharper.detect_outliers: (
            ["df", "columns", "method", "threshold"],
            {"columns": None, "method": "iqr", "threshold": 1.5},
            {
                "df": pd.DataFrame,
                "columns": Sequence[str] | None,
                "method": str,
                "threshold": float,
                "return": sharper.OutlierAnalysis,
            },
        ),
    }

    for function, (names, defaults, expected_hints) in contracts.items():
        signature = inspect.signature(function)
        assert list(signature.parameters) == names
        assert (
            signature.parameters["df"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        for parameter, default in defaults.items():
            assert (
                signature.parameters[parameter].kind is inspect.Parameter.KEYWORD_ONLY
            )
            assert signature.parameters[parameter].default == default
        hints = get_type_hints(function)
        assert hints == expected_hints
        assert function.__doc__


def test_task07_dataclass_fields_are_frozen() -> None:
    """Task 07 result dataclasses contain exactly the frozen fields."""
    common_hints = {
        "n_rows": int,
        "requested_columns": tuple[str, ...] | None,
        "analyzed_columns": tuple[str, ...],
        "skipped_columns": tuple[str, ...],
        "skipped_reasons": dict[str, str],
    }
    contracts = {
        sharper.NumericAnalysis: {
            **common_hints,
            "summary": pd.DataFrame,
        },
        sharper.CategoricalAnalysis: {
            **common_hints,
            "top_n": int,
            "summary": pd.DataFrame,
            "top_categories": pd.DataFrame,
        },
        sharper.CorrelationAnalysis: {
            **common_hints,
            "method": str,
            "max_columns": int,
            "min_periods": int,
            "truncated": bool,
            "correlations": pd.DataFrame,
        },
        sharper.OutlierAnalysis: {
            **common_hints,
            "method": str,
            "threshold": float,
            "summary": pd.DataFrame,
            "outliers": pd.DataFrame,
        },
    }

    for result_type, expected_hints in contracts.items():
        assert is_dataclass(result_type)
        assert result_type.__dataclass_params__.frozen is True
        assert [field.name for field in fields(result_type)] == list(expected_hints)
        assert get_type_hints(result_type) == expected_hints
        assert result_type.__doc__


def test_task08_function_signatures_and_typing() -> None:
    """Group and target APIs expose exactly the frozen Task 08 signatures."""
    group_signature = inspect.signature(sharper.compare_groups)
    assert list(group_signature.parameters) == [
        "df",
        "group_by",
        "values",
        "max_groups",
    ]
    assert (
        group_signature.parameters["group_by"].kind
        is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    assert group_signature.parameters["values"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        group_signature.parameters["max_groups"].kind is inspect.Parameter.KEYWORD_ONLY
    )
    assert group_signature.parameters["values"].default is None
    assert group_signature.parameters["max_groups"].default == 20
    assert get_type_hints(sharper.compare_groups) == {
        "df": pd.DataFrame,
        "group_by": str,
        "values": Sequence[str] | None,
        "max_groups": int,
        "return": sharper.GroupComparison,
    }

    target_signature = inspect.signature(sharper.analyze_target_relationships)
    assert list(target_signature.parameters) == ["df", "target", "task", "features"]
    assert (
        target_signature.parameters["target"].kind
        is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    assert target_signature.parameters["task"].kind is inspect.Parameter.KEYWORD_ONLY
    assert target_signature.parameters["task"].default is inspect.Parameter.empty
    assert (
        target_signature.parameters["features"].kind is inspect.Parameter.KEYWORD_ONLY
    )
    assert target_signature.parameters["features"].default is None
    assert get_type_hints(sharper.analyze_target_relationships) == {
        "df": pd.DataFrame,
        "target": str,
        "task": Literal["classification", "regression"],
        "features": Sequence[str] | None,
        "return": sharper.TargetAnalysis,
    }
    assert sharper.compare_groups.__doc__
    assert sharper.analyze_target_relationships.__doc__


def test_task08_dataclass_fields_are_frozen() -> None:
    """Task 08 result dataclasses contain exactly the frozen fields and hints."""
    contracts = {
        sharper.GroupComparison: {
            "n_rows": int,
            "group_by": str,
            "requested_values": tuple[str, ...] | None,
            "analyzed_values": tuple[str, ...],
            "skipped_values": tuple[str, ...],
            "skipped_reasons": dict[str, str],
            "max_groups": int,
            "available_group_count": int,
            "displayed_group_count": int,
            "missing_group_count": int,
            "truncated": bool,
            "truncation_reason": str | None,
            "summary": pd.DataFrame,
        },
        sharper.TargetAnalysis: {
            "n_rows": int,
            "target": str,
            "task": str,
            "requested_features": tuple[str, ...] | None,
            "analyzed_features": tuple[str, ...],
            "skipped_features": tuple[str, ...],
            "skipped_reasons": dict[str, str],
            "max_features": int,
            "max_categories": int,
            "available_feature_count": int,
            "truncated": bool,
            "truncation_reason": str | None,
            "numeric_details": pd.DataFrame,
            "category_details": pd.DataFrame,
            "statistical_tests": pd.DataFrame,
            "limitations": tuple[str, ...],
        },
    }
    for result_type, expected_hints in contracts.items():
        assert is_dataclass(result_type)
        assert result_type.__dataclass_params__.frozen is True
        assert [field.name for field in fields(result_type)] == list(expected_hints)
        assert get_type_hints(result_type) == expected_hints
        assert result_type.__doc__


def test_task09_function_signatures_and_typing() -> None:
    """Feature APIs expose exactly the frozen Task 09 signatures."""
    suggest_signature = inspect.signature(sharper.suggest_feature_derivations)
    assert list(suggest_signature.parameters) == [
        "df",
        "schema",
        "target",
        "exclude_columns",
        "reference_date",
        "max_suggestions",
    ]
    for name in list(suggest_signature.parameters)[1:]:
        assert suggest_signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert get_type_hints(sharper.suggest_feature_derivations) == {
        "df": pd.DataFrame,
        "schema": sharper.SchemaReport | None,
        "target": str | None,
        "exclude_columns": Sequence[str],
        "reference_date": str | date | datetime | pd.Timestamp | None,
        "max_suggestions": int,
        "return": sharper.FeatureSuggestionReport,
    }

    derive_signature = inspect.signature(sharper.derive_features)
    assert list(derive_signature.parameters) == ["df", "suggestions", "copy"]
    assert derive_signature.parameters["copy"].kind is inspect.Parameter.KEYWORD_ONLY
    assert get_type_hints(sharper.derive_features) == {
        "df": pd.DataFrame,
        "suggestions": Sequence[sharper.FeatureSuggestion],
        "copy": bool,
        "return": sharper.FeatureDerivationResult,
    }
    assert sharper.suggest_feature_derivations.__doc__
    assert sharper.derive_features.__doc__


def test_task09_dataclass_fields_are_frozen() -> None:
    """Task 09 result dataclasses contain exactly the frozen fields and hints."""
    contracts = {
        sharper.FeatureSuggestion: {
            "name": str,
            "feature_type": str,
            "source_columns": tuple[str, ...],
            "formula": str | None,
            "parameters": tuple[tuple[str, str], ...],
            "reason": str,
            "risk": str,
            "requires_fit": bool,
            "priority": int,
        },
        sharper.FeatureSuggestionReport: {
            "n_rows": int,
            "requested_target": str | None,
            "requested_exclusions": tuple[str, ...],
            "reference_date": str | None,
            "eligible_columns": tuple[str, ...],
            "excluded_columns": tuple[str, ...],
            "skipped_columns": tuple[str, ...],
            "skipped_reasons": dict[str, str],
            "max_suggestions": int,
            "type_budgets": dict[str, int],
            "available_counts": dict[str, int],
            "available_suggestion_count": int,
            "truncated": bool,
            "truncation_reason": str | None,
            "suggestions": tuple[sharper.FeatureSuggestion, ...],
        },
        sharper.FeatureDerivationResult: {
            "data": pd.DataFrame,
            "applied_suggestions": tuple[str, ...],
            "skipped_suggestions": tuple[str, ...],
            "skipped_reasons": dict[str, str],
            "copy": bool,
        },
    }
    for result_type, expected_hints in contracts.items():
        assert is_dataclass(result_type)
        assert result_type.__dataclass_params__.frozen is True
        assert [field.name for field in fields(result_type)] == list(expected_hints)
        assert get_type_hints(result_type) == expected_hints
        assert result_type.__doc__


def test_task11_public_signatures_and_frozen_fields() -> None:
    """Task 11 exposes only its frozen classification contract."""
    training = inspect.signature(sharper.train_classifier)
    assert list(training.parameters) == [
        "df",
        "target",
        "features",
        "exclude_columns",
        "time_column",
        "estimator",
        "test_size",
        "random_state",
    ]
    assert all(
        training.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in list(training.parameters)[2:]
    )
    assert training.parameters["features"].default is None
    assert training.parameters["exclude_columns"].default == ()
    assert training.parameters["time_column"].default is None
    assert training.parameters["estimator"].default is None
    assert training.parameters["test_size"].default == 0.20
    assert training.parameters["random_state"].default == 42
    for function in (
        sharper.train_classifier,
        sharper.evaluate_classifier,
        sharper.evaluate_model,
        sharper.plot_classification_evaluation,
    ):
        assert function.__doc__
    assert [field.name for field in fields(sharper.TrainingResult)] == [
        "task",
        "target",
        "feature_columns",
        "excluded_columns",
        "time_column",
        "schema",
        "pipeline",
        "estimator",
        "classes",
        "train_row_positions",
        "test_row_positions",
        "X_test",
        "y_test",
        "test_size",
        "random_state",
        "warnings",
        "limitations",
    ]
    assert [field.name for field in fields(sharper.ClassificationEvaluation)] == [
        "task",
        "target",
        "holdout_positions",
        "classes",
        "y_true",
        "y_pred",
        "score_kind",
        "positive_label",
        "scores",
        "roc_curve",
        "metrics",
        "confusion_matrix",
        "roc_auc",
        "limitations",
    ]
    assert sharper.TrainingResult.__dataclass_params__.frozen is True
    assert sharper.ClassificationEvaluation.__dataclass_params__.frozen is True


def test_task12_public_signatures_and_frozen_fields() -> None:
    """Task 12 exposes its independent frozen regression contract."""
    signature = inspect.signature(sharper.train_regressor)
    assert list(signature.parameters) == [
        "df",
        "target",
        "features",
        "exclude_columns",
        "time_column",
        "estimator",
        "test_size",
        "random_state",
    ]
    assert all(
        signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in list(signature.parameters)[2:]
    )
    assert signature.parameters["features"].default is None
    assert signature.parameters["exclude_columns"].default == ()
    assert signature.parameters["time_column"].default is None
    assert signature.parameters["estimator"].default is None
    assert signature.parameters["test_size"].default == 0.20
    assert signature.parameters["random_state"].default == 42
    assert get_type_hints(sharper.train_regressor) == {
        "df": pd.DataFrame,
        "target": str,
        "features": Sequence[str] | None,
        "exclude_columns": Sequence[str],
        "time_column": str | None,
        "estimator": RegressorMixin | None,
        "test_size": float,
        "random_state": int | None,
        "return": sharper.RegressionTrainingResult,
    }
    assert [field.name for field in fields(sharper.RegressionTrainingResult)] == [
        "task",
        "target",
        "feature_columns",
        "excluded_columns",
        "time_column",
        "schema",
        "pipeline",
        "estimator",
        "train_row_positions",
        "test_row_positions",
        "X_test",
        "y_test",
        "test_size",
        "random_state",
        "warnings",
        "limitations",
    ]
    assert [field.name for field in fields(sharper.RegressionEvaluation)] == [
        "task",
        "target",
        "holdout_positions",
        "predictions",
        "metrics",
        "limitations",
    ]
    assert sharper.RegressionTrainingResult.__dataclass_params__.frozen is True
    assert sharper.RegressionEvaluation.__dataclass_params__.frozen is True
    for function in (
        sharper.train_regressor,
        sharper.evaluate_regressor,
        sharper.evaluate_model,
        sharper.plot_regression_evaluation,
    ):
        assert function.__doc__
