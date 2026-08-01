"""Opt-in data-quality, missingness, and leakage evidence audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from datetime import datetime
from math import isfinite

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from sharper._condition_kernel import (
    _ConditionOperand,
    _evaluate_atomic_condition,
    _is_allowed_scalar,
)
from sharper.schema import infer_schema

_KERNEL_VERSION = "task16-audit-v1"
_CONFIG_VERSION = "task16-data-audit-config-v1"


@dataclass(frozen=True)
class DataAuditRoles:
    """Declare column roles used by the opt-in audit; shallow frozen."""

    target: str | None = None
    features: tuple[str, ...] | None = None
    score_columns: tuple[str, ...] = ()
    excluded_columns: tuple[str, ...] = ()
    row_identifier: str | None = None
    group: str | None = None
    partition: str | None = None
    fold: str | None = None
    selection: str | None = None
    historical_action: str | None = None
    historical_policy: str | None = None
    cost_columns: tuple[str, ...] = ()
    exposure_columns: tuple[str, ...] = ()
    constraint_input_columns: tuple[str, ...] = ()
    observation_time: str | None = None
    event_time: str | None = None
    shared_feature_available_time: str | None = None
    feature_available_time_map: tuple[tuple[str, str], ...] = ()
    label_available_time: str | None = None
    outcome_end_time: str | None = None
    partition_cutoff: str | None = None
    window_start: str | None = None
    window_end: str | None = None
    horizon_end: str | None = None
    analysis_as_of: str | None = None
    post_outcome_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ColumnAuditRule:
    """Declare one closed column audit rule; shallow frozen."""

    column: str
    minimum: object | None = None
    maximum: object | None = None
    minimum_inclusive: bool = True
    maximum_inclusive: bool = True
    allowed_values: tuple[object, ...] = ()
    special_values: tuple[object, ...] = ()
    not_after_columns: tuple[str, ...] = ()
    nondecreasing: bool = False


@dataclass(frozen=True)
class DataAuditConfig:
    """Configure bounded deterministic audit policies; shallow frozen."""

    positive_label: object | None = None
    missing_warning_rate: float = 0.40
    near_constant_rate: float = 0.95
    high_cardinality_count: int = 50
    high_cardinality_rate: float = 0.50
    identifier_rate: float = 0.98
    identifier_min_non_missing: int = 20
    rare_class_count: int = 20
    rare_class_rate: float = 0.05
    proxy_min_support: int = 20
    near_copy_rate: float = 0.99
    collinearity_threshold: float = 0.999
    collinearity_min_periods: int = 20
    missingness_drift_absolute_threshold: float = 0.10
    missingness_drift_relative_threshold: float = 0.50
    minimum_drift_rows: int = 30
    partition_target_rate_shift_threshold: float | None = None
    partition_target_min_support: int = 30
    max_columns: int = 500
    max_missing_patterns: int = 100
    max_finding_samples: int = 20
    max_unique_inspection_rows: int = 100_000
    max_category_levels: int = 1_000
    max_collinearity_columns: int = 50
    duplicate_scan_row_limit: int = 1_000_000
    max_column_rules: int = 100
    column_rules: tuple[ColumnAuditRule, ...] = ()


@dataclass(frozen=True)
class DataAuditResult:
    """Contain fourteen bounded evidence tables; shallow frozen."""

    config_fingerprint: str
    n_rows: int
    n_columns: int
    reference_n_rows: int | None
    reference_n_columns: int | None
    dataset_profile: pd.DataFrame
    column_profile: pd.DataFrame
    numeric_profile: pd.DataFrame
    categorical_profile: pd.DataFrame
    target_profile: pd.DataFrame
    slice_profile: pd.DataFrame
    missingness_patterns: pd.DataFrame
    missingness_drift: pd.DataFrame
    schema_drift: pd.DataFrame
    collinearity: pd.DataFrame
    point_in_time_profile: pd.DataFrame
    resource_usage: pd.DataFrame
    provenance: pd.DataFrame
    findings: pd.DataFrame
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]


_TABLE_SCHEMAS: dict[str, tuple[tuple[str, str], ...]] = {
    "dataset_profile": tuple((x, "string") for x in ("side",))
    + tuple((x, "int64") for x in ("n_rows", "n_columns", "profiled_column_count"))
    + (("declared_feature_count", "Int64"),)
    + tuple((x, "string") for x in ("feature_status", "feature_reason"))
    + (("duplicate_row_count", "Int64"), ("duplicate_row_rate", "Float64"))
    + tuple((x, "string") for x in ("duplicate_row_status", "duplicate_row_reason"))
    + (("duplicate_index_count", "Int64"), ("duplicate_index_rate", "Float64"))
    + tuple((x, "string") for x in ("duplicate_index_status", "duplicate_index_reason"))
    + (("memory_usage_bytes", "int64"), ("finding_key", "string")),
    "column_profile": (
        ("side", "string"),
        ("column", "string"),
        ("column_position", "int64"),
        ("role", "string"),
        ("pandas_dtype", "string"),
        ("logical_type", "string"),
        ("n_rows", "int64"),
        ("non_missing_count", "int64"),
        ("missing_count", "int64"),
        ("missing_rate", "Float64"),
        ("missing_status", "string"),
        ("missing_reason", "string"),
        ("mixed_python_type_count", "Int64"),
        ("empty_string_count", "Int64"),
        ("whitespace_only_count", "Int64"),
        ("unique_count", "Int64"),
        ("unique_rate", "Float64"),
        ("top_count", "Int64"),
        ("top_rate", "Float64"),
        ("all_missing", "boolean"),
        ("constant", "boolean"),
        ("near_constant", "boolean"),
        ("high_cardinality", "boolean"),
        ("suspected_identifier", "boolean"),
        ("value_profile_status", "string"),
        ("value_profile_reason", "string"),
        ("finding_key", "string"),
    ),
    "numeric_profile": tuple((x, "string") for x in ("side", "column"))
    + tuple(
        (x, "int64")
        for x in (
            "n_rows",
            "non_missing_count",
            "missing_count",
            "finite_count",
            "positive_inf_count",
            "negative_inf_count",
        )
    )
    + tuple(
        (x, "Float64")
        for x in ("mean", "std", "minimum", "q25", "median", "q75", "maximum")
    )
    + tuple(
        (x, "string")
        for x in (
            "count_status",
            "count_reason",
            "finite_status",
            "finite_reason",
            "location_status",
            "location_reason",
            "dispersion_status",
            "dispersion_reason",
            "range_status",
            "range_reason",
            "quantile_status",
            "quantile_reason",
            "finding_key",
        )
    ),
    "categorical_profile": tuple((x, "string") for x in ("side", "column"))
    + (
        ("non_missing_count", "int64"),
        ("unique_count", "Int64"),
        ("unique_rate", "Float64"),
        ("top_count", "Int64"),
        ("top_rate", "Float64"),
        ("singleton_level_count", "Int64"),
        ("unseen_in_current_count", "Int64"),
        ("unseen_in_current_rate", "Float64"),
    )
    + tuple(
        (x, "string")
        for x in (
            "count_status",
            "count_reason",
            "cardinality_status",
            "cardinality_reason",
            "cardinality_rate_status",
            "cardinality_rate_reason",
            "frequency_status",
            "frequency_reason",
            "concentration_status",
            "concentration_reason",
            "comparison_status",
            "comparison_reason",
            "finding_key",
        )
    ),
    "target_profile": (
        ("side", "string"),
        ("class_position", "int64"),
        ("is_positive", "boolean"),
        ("positive_label_declared", "boolean"),
        ("positive_label_type", "string"),
        ("count", "int64"),
        ("rate", "Float64"),
        ("target_non_missing_n", "int64"),
    )
    + tuple(
        (x, "string")
        for x in (
            "class_status",
            "class_reason",
            "binary_status",
            "binary_reason",
            "balance_status",
            "balance_reason",
            "positive_class_status",
            "positive_class_reason",
            "finding_key",
        )
    ),
    "slice_profile": (
        ("side", "string"),
        ("slice_role", "string"),
        ("row_kind", "string"),
        ("slice_ordinal", "Int64"),
        ("partition_ordinal", "Int64"),
        ("fold_ordinal", "Int64"),
        ("missing_bucket", "boolean"),
        ("row_count", "Int64"),
        ("target_non_missing_count", "Int64"),
        ("target_non_missing_rate", "Float64"),
        ("positive_count", "Int64"),
        ("event_rate", "Float64"),
    )
    + tuple(
        (x, "string")
        for x in (
            "size_status",
            "size_reason",
            "target_rate_status",
            "target_rate_reason",
            "event_status",
            "event_reason",
            "quality_status",
            "quality_reason",
            "finding_key",
        )
    ),
    "missingness_patterns": (
        ("pattern_key", "string"),
        ("pattern_bits", "string"),
        ("aggregated", "boolean"),
        ("source_pattern_count", "int64"),
        ("missing_count", "Int64"),
        ("row_count", "int64"),
        ("row_rate", "Float64"),
        ("missing_cell_count", "int64"),
        ("min_missing_count", "Int64"),
        ("max_missing_count", "Int64"),
        ("sample_positions", "object"),
        ("reference_row_count", "Int64"),
        ("reference_row_rate", "Float64"),
        ("absolute_rate_change", "Float64"),
    )
    + tuple(
        (x, "string")
        for x in (
            "count_status",
            "count_reason",
            "rate_status",
            "rate_reason",
            "reference_count_status",
            "reference_count_reason",
            "reference_rate_status",
            "reference_rate_reason",
            "comparison_status",
            "comparison_reason",
            "finding_key",
        )
    ),
    "missingness_drift": (
        ("column", "string"),
        ("reference_present", "boolean"),
        ("current_present", "boolean"),
        ("reference_n", "Int64"),
        ("current_n", "Int64"),
        ("reference_missing_count", "Int64"),
        ("current_missing_count", "Int64"),
        ("reference_missing_rate", "Float64"),
        ("current_missing_rate", "Float64"),
        ("absolute_rate_change", "Float64"),
        ("relative_rate_change", "Float64"),
        ("new_all_missing", "boolean"),
        ("recovered", "boolean"),
    )
    + tuple(
        (x, "string")
        for x in (
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
    ),
    "schema_drift": (
        ("column", "string"),
        ("reference_position", "Int64"),
        ("current_position", "Int64"),
        ("reference_dtype", "string"),
        ("current_dtype", "string"),
        ("reference_logical_type", "string"),
        ("current_logical_type", "string"),
        ("reference_role", "string"),
        ("current_role", "string"),
    )
    + tuple(
        (x, "boolean")
        for x in (
            "column_added",
            "column_removed",
            "dtype_changed",
            "logical_type_changed",
            "role_changed",
        )
    )
    + tuple(
        (x, "string") for x in ("primary_change", "status", "reason", "finding_key")
    ),
    "collinearity": (
        ("left_column", "string"),
        ("right_column", "string"),
        ("valid_n", "int64"),
        ("pearson_r", "Float64"),
        ("absolute_r", "Float64"),
        ("threshold", "Float64"),
        ("status", "string"),
        ("reason", "string"),
        ("finding_key", "string"),
    ),
    "point_in_time_profile": (
        ("side", "string"),
        ("scope", "string"),
        ("column", "string"),
        ("evaluated_count", "Int64"),
        ("violation_count", "Int64"),
        ("not_verifiable_count", "Int64"),
        ("status", "string"),
        ("reason", "string"),
        ("finding_key", "string"),
    ),
    "resource_usage": (
        ("side", "string"),
        ("resource", "string"),
        ("requested", "Int64"),
        ("available", "Int64"),
        ("actual", "Int64"),
        ("truncated", "boolean"),
        ("status", "string"),
        ("reason", "string"),
        ("finding_key", "string"),
    ),
    "provenance": (
        ("provenance_key", "string"),
        ("value_type", "string"),
        ("numeric_value", "Float64"),
        ("text_value", "string"),
        ("count_value", "Int64"),
        ("boolean_value", "boolean"),
        ("status", "string"),
        ("reason", "string"),
    ),
    "findings": (
        ("finding_key", "string"),
        ("category", "string"),
        ("scope", "string"),
        ("dataset_role", "string"),
        ("column", "string"),
        ("column_position", "Int64"),
        ("role", "string"),
        ("severity", "string"),
        ("status", "string"),
        ("reason", "string"),
        ("metric_key", "string"),
        ("value", "Float64"),
        ("threshold", "Float64"),
        ("count", "Int64"),
        ("denominator", "Int64"),
        ("affected_rate", "Float64"),
        ("sample_positions", "object"),
        ("detail_table", "string"),
        ("detail_row_ordinal", "Int64"),
        ("recommendation", "string"),
        ("limitation", "string"),
        ("provenance", "string"),
    ),
}


def _frame(name: str, rows: list[dict[str, object]]) -> pd.DataFrame:
    schema = _TABLE_SCHEMAS[name]
    result = pd.DataFrame({column: pd.Series(dtype=dtype) for column, dtype in schema})
    if rows:
        result = pd.DataFrame(rows, columns=[column for column, _ in schema])
        for column, dtype in schema:
            result[column] = result[column].astype(dtype)
    return result


def _audit_config_error(key: str) -> ValueError:
    return ValueError(f"data audit config is invalid: {key}")


def _audit_input_error(key: str) -> ValueError:
    return ValueError(f"data audit input is invalid: {key}")


def _safe_scalar_scan(frame: pd.DataFrame) -> None:
    for position in range(len(frame.index)):
        if not _is_allowed_scalar(frame.index[position]):
            raise _audit_input_error("unsupported_scalar_type")
    for column_position in range(frame.shape[1]):
        series = frame.iloc[:, column_position]
        if series.dtype == object or isinstance(
            series.dtype, (pd.CategoricalDtype, pd.StringDtype)
        ):
            if isinstance(series.dtype, pd.CategoricalDtype):
                for value in series.cat.categories.tolist():
                    if not _is_allowed_scalar(value):
                        raise _audit_input_error("unsupported_scalar_type")
            for row_position in range(len(series)):
                if not _is_allowed_scalar(series.iat[row_position]):
                    raise _audit_input_error("unsupported_scalar_type")


_SCALAR_ROLES = (
    "target",
    "row_identifier",
    "group",
    "partition",
    "fold",
    "selection",
    "historical_action",
    "historical_policy",
    "observation_time",
    "event_time",
    "shared_feature_available_time",
    "label_available_time",
    "outcome_end_time",
    "partition_cutoff",
    "window_start",
    "window_end",
    "horizon_end",
    "analysis_as_of",
)
_TUPLE_ROLES = (
    "score_columns",
    "excluded_columns",
    "cost_columns",
    "exposure_columns",
    "constraint_input_columns",
    "post_outcome_columns",
)


def _validate_roles(frame: pd.DataFrame, roles: DataAuditRoles) -> None:
    scalar_values: list[str] = []
    for name in _SCALAR_ROLES:
        value = getattr(roles, name)
        if value is not None:
            if type(value) is not str or not value:
                raise _audit_input_error("invalid_selector")
            scalar_values.append(value)
    if len(scalar_values) != len(set(scalar_values)):
        raise _audit_input_error("conflicting_roles")
    tuple_values: dict[str, tuple[str, ...]] = {}
    if roles.features is not None:
        tuple_values["features"] = roles.features
    for name in _TUPLE_ROLES:
        tuple_values[name] = getattr(roles, name)
    for name, values in tuple_values.items():
        if type(values) is not tuple or any(
            type(value) is not str or not value for value in values
        ):
            raise _audit_input_error("invalid_selector")
        if len(values) != len(set(values)):
            raise _audit_input_error("duplicate_selector")
    exclusive_names = (
        "score_columns",
        "cost_columns",
        "exposure_columns",
        "constraint_input_columns",
    )
    exclusive = [value for name in exclusive_names for value in tuple_values[name]]
    if len(exclusive) != len(set(exclusive)) or set(exclusive) & set(scalar_values):
        raise _audit_input_error("conflicting_roles")
    all_nonexcluded = (
        set(scalar_values)
        | set(exclusive)
        | set(roles.features or ())
        | set(roles.post_outcome_columns)
    )
    if set(roles.excluded_columns) & all_nonexcluded:
        raise _audit_input_error("conflicting_roles")
    if roles.target in roles.post_outcome_columns:
        raise _audit_input_error("conflicting_roles")
    mapping = roles.feature_available_time_map
    if type(mapping) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or any(type(x) is not str or not x for x in item)
        for item in mapping
    ):
        raise _audit_input_error("invalid_feature_availability_mapping")
    if roles.shared_feature_available_time is not None and mapping:
        raise _audit_input_error("invalid_feature_availability_mapping")
    if (
        roles.shared_feature_available_time is not None or mapping
    ) and not roles.features:
        raise _audit_input_error("invalid_feature_availability_mapping")
    mapped_features = [item[0] for item in mapping]
    if len(mapped_features) != len(set(mapped_features)) or any(
        feature not in roles.features or feature == available
        for feature, available in mapping
    ):
        raise _audit_input_error("invalid_feature_availability_mapping")
    selectors = (
        all_nonexcluded
        | set(roles.excluded_columns)
        | {item for pair in mapping for item in pair}
    )
    if any(selector not in frame.columns for selector in selectors):
        raise _audit_input_error("unknown_selector")


def _validate_config(config: DataAuditConfig, roles: DataAuditRoles) -> None:
    if type(config.column_rules) is not tuple:
        raise _audit_config_error("invalid_selector")
    if len(config.column_rules) > config.max_column_rules:
        raise _audit_config_error("invalid_budget")
    rate_names = (
        "missing_warning_rate",
        "near_constant_rate",
        "high_cardinality_rate",
        "identifier_rate",
        "rare_class_rate",
        "near_copy_rate",
    )
    nonnegative = (
        "missingness_drift_absolute_threshold",
        "missingness_drift_relative_threshold",
    )
    for name in rate_names:
        value = getattr(config, name)
        if (
            type(value) not in (float, int, np.float16, np.float32, np.float64)
            or type(value) is bool
            or not isfinite(float(value))
            or not 0 <= float(value) <= 1
        ):
            raise _audit_config_error("invalid_threshold")
    if (
        config.near_constant_rate <= 0
        or config.identifier_rate <= 0
        or config.near_copy_rate <= 0
        or not 0 < config.collinearity_threshold <= 1
    ):
        raise _audit_config_error("invalid_threshold")
    for name in nonnegative:
        value = getattr(config, name)
        if (
            type(value) not in (float, int, np.float16, np.float32, np.float64)
            or type(value) is bool
            or not isfinite(float(value))
            or value < 0
        ):
            raise _audit_config_error("invalid_threshold")
    if config.partition_target_rate_shift_threshold is not None and (
        type(config.partition_target_rate_shift_threshold)
        not in (float, int, np.float16, np.float32, np.float64)
        or type(config.partition_target_rate_shift_threshold) is bool
        or not 0 <= float(config.partition_target_rate_shift_threshold) <= 1
    ):
        raise _audit_config_error("invalid_threshold")
    bounds = {
        "max_columns": (1, 500),
        "max_missing_patterns": (1, 1000),
        "max_finding_samples": (0, 100),
        "max_unique_inspection_rows": (1, 1_000_000),
        "max_category_levels": (2, 100_000),
        "max_collinearity_columns": (2, 200),
        "duplicate_scan_row_limit": (1, 5_000_000),
        "max_column_rules": (0, 500),
    }
    count_names = (
        "high_cardinality_count",
        "identifier_min_non_missing",
        "rare_class_count",
        "proxy_min_support",
        "collinearity_min_periods",
        "minimum_drift_rows",
        "partition_target_min_support",
    )
    for name in count_names:
        value = getattr(config, name)
        if type(value) is not int or value <= 0:
            raise _audit_config_error("invalid_budget")
    for name, (low, high) in bounds.items():
        value = getattr(config, name)
        if type(value) is not int or not low <= value <= high:
            raise _audit_config_error("invalid_budget")
    if config.positive_label is not None:
        if roles.target is None:
            raise _audit_config_error("positive_label_without_target")
        if (
            not _is_allowed_scalar(config.positive_label)
            or config.positive_label is pd.NA
            or config.positive_label is pd.NaT
        ):
            raise _audit_config_error("invalid_positive_label")
    seen: set[str] = set()
    for rule in config.column_rules:
        if (
            type(rule) is not ColumnAuditRule
            or type(rule.column) is not str
            or not rule.column
        ):
            raise _audit_config_error("invalid_selector")
        if rule.column in seen:
            raise _audit_config_error("duplicate_column_rule")
        seen.add(rule.column)
        if rule.column not in roles.excluded_columns and not any(
            (
                rule.minimum is not None,
                rule.maximum is not None,
                rule.allowed_values,
                rule.special_values,
                rule.not_after_columns,
                rule.nondecreasing,
            )
        ):
            raise _audit_config_error("empty_column_rule")


def _role_for(column: str, roles: DataAuditRoles) -> str:
    ordered = (
        ("feature", roles.features or ()),
        ("score", roles.score_columns),
        ("target", (roles.target,)),
        ("row_identifier", (roles.row_identifier,)),
        ("group", (roles.group,)),
        ("partition", (roles.partition,)),
        ("fold", (roles.fold,)),
        ("selection", (roles.selection,)),
        ("historical_action", (roles.historical_action,)),
        ("historical_policy", (roles.historical_policy,)),
        ("cost", roles.cost_columns),
        ("exposure", roles.exposure_columns),
        ("constraint_input", roles.constraint_input_columns),
        ("observation_time", (roles.observation_time,)),
        ("event_time", (roles.event_time,)),
        (
            "feature_available_time",
            tuple(
                [roles.shared_feature_available_time]
                if roles.shared_feature_available_time
                else []
            )
            + tuple(x[1] for x in roles.feature_available_time_map),
        ),
        ("label_available_time", (roles.label_available_time,)),
        ("outcome_end_time", (roles.outcome_end_time,)),
        ("partition_cutoff", (roles.partition_cutoff,)),
        ("window_start", (roles.window_start,)),
        ("window_end", (roles.window_end,)),
        ("horizon_end", (roles.horizon_end,)),
        ("analysis_as_of", (roles.analysis_as_of,)),
        ("post_outcome", roles.post_outcome_columns),
    )
    found = [name for name, values in ordered if column in values]
    return "|".join(found) if found else "unassigned"


def _profile_columns(frame: pd.DataFrame, roles: DataAuditRoles) -> list[str]:
    if roles.features is None:
        return [
            column for column in frame.columns if column not in roles.excluded_columns
        ]
    explicit = list(roles.features)
    for role_field in fields(roles):
        value = getattr(roles, role_field.name)
        if type(value) is str:
            explicit.append(value)
        elif type(value) is tuple and role_field.name not in {
            "features",
            "excluded_columns",
            "feature_available_time_map",
        }:
            explicit.extend(value)
    explicit.extend(item[1] for item in roles.feature_available_time_map)
    selected = set(explicit) - set(roles.excluded_columns)
    return [column for column in frame.columns if column in selected]


def _value_key(value: object) -> tuple[str, object]:
    if (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or (type(value) is float and np.isnan(value))
    ):
        return ("missing", 0)
    if type(value) in (
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
    ):
        value = value.item()
    return (
        "bool"
        if type(value) is bool
        else "int"
        if type(value) is int
        else "float"
        if type(value) is float
        else "str"
        if type(value) is str
        else "datetime"
        if type(value) in (datetime, pd.Timestamp)
        else "date",
        value,
    )


def _is_missing(value: object) -> bool:
    return (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or (
            type(value) in (float, np.float16, np.float32, np.float64)
            and np.isnan(value)
        )
    )


def _profile_side(
    frame: pd.DataFrame,
    side: str,
    roles: DataAuditRoles,
    config: DataAuditConfig,
    schema: object,
) -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]], list[str]]:
    rows: dict[str, list[dict[str, object]]] = {
        name: []
        for name in (
            "dataset_profile",
            "column_profile",
            "numeric_profile",
            "categorical_profile",
            "target_profile",
            "slice_profile",
        )
    }
    findings: list[dict[str, object]] = []
    warnings: list[str] = []
    columns = _profile_columns(frame, roles)
    n = len(frame)
    scan_ok = n <= config.duplicate_scan_row_limit
    dup_rows = int(frame.duplicated(keep=False).sum()) if scan_ok else pd.NA
    dup_index = int(frame.index.duplicated(keep=False).sum())
    rows["dataset_profile"].append(
        {
            "side": side,
            "n_rows": n,
            "n_columns": frame.shape[1],
            "profiled_column_count": len(columns),
            "declared_feature_count": pd.NA
            if roles.features is None
            else len(roles.features),
            "feature_status": "unavailable" if roles.features is None else "available",
            "feature_reason": "role_not_declared"
            if roles.features is None
            else "computed",
            "duplicate_row_count": dup_rows,
            "duplicate_row_rate": (dup_rows / n if scan_ok and n else pd.NA),
            "duplicate_row_status": "available" if scan_ok else "not_verifiable",
            "duplicate_row_reason": "computed" if scan_ok else "duplicate_scan_budget",
            "duplicate_index_count": dup_index,
            "duplicate_index_rate": dup_index / n if n else pd.NA,
            "duplicate_index_status": "available",
            "duplicate_index_reason": "computed" if n else "no_rows",
            "memory_usage_bytes": int(frame.memory_usage(index=True, deep=True).sum()),
            "finding_key": pd.NA,
        }
    )
    if n == 0:
        findings.append(
            _finding(
                "dataset_structure",
                "dataset",
                side,
                None,
                -1,
                "empty_dataset",
                "n_rows",
                0,
                None,
                0,
                0,
                (),
                "dataset_profile",
                0,
                "pandas_structure",
            )
        )
    if roles.features == ():
        findings.append(
            _finding(
                "dataset_structure",
                "dataset",
                side,
                None,
                -1,
                "zero_feature_dataset",
                "declared_feature_count",
                0,
                None,
                0,
                n,
                (),
                "dataset_profile",
                0,
                "caller_roles",
            )
        )
    if scan_ok and dup_rows:
        findings.append(
            _finding(
                "dataset_structure",
                "row_set",
                side,
                None,
                -1,
                "duplicate_rows",
                "duplicate_row_count",
                float(dup_rows),
                None,
                dup_rows,
                n,
                tuple(
                    np.flatnonzero(frame.duplicated(keep=False))[
                        : config.max_finding_samples
                    ]
                ),
                "dataset_profile",
                0,
                "pandas_structure",
            )
        )
    if dup_index:
        findings.append(
            _finding(
                "dataset_structure",
                "row_set",
                side,
                None,
                -1,
                "duplicate_index",
                "duplicate_index_count",
                float(dup_index),
                None,
                dup_index,
                n,
                tuple(
                    np.flatnonzero(frame.index.duplicated(keep=False))[
                        : config.max_finding_samples
                    ]
                ),
                "dataset_profile",
                0,
                "pandas_structure",
            )
        )
    schema_by = {item.name: item for item in schema.columns}
    for pos, column in enumerate(frame.columns):
        if column not in columns:
            continue
        series = frame[column]
        values = [series.iat[i] for i in range(n)]
        missing = [_is_missing(v) for v in values]
        non = [v for v, m in zip(values, missing, strict=True) if not m]
        mc = sum(missing)
        nn = len(non)
        budget_ok = n <= config.max_unique_inspection_rows
        keys = [_value_key(v) for v in non] if budget_ok else []
        counts: dict[tuple[str, object], int] = {}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        unique = len(counts) if budget_ok else pd.NA
        top = max(counts.values(), default=0) if budget_ok else pd.NA
        unique_rate = unique / nn if budget_ok and nn else pd.NA
        top_rate = top / nn if budget_ok and nn else pd.NA
        all_missing = bool(n > 0 and mc == n)
        constant = bool(nn > 0 and budget_ok and unique == 1)
        near = bool(
            budget_ok and nn and not constant and top_rate >= config.near_constant_rate
        )
        high = bool(
            budget_ok
            and nn
            and unique > config.high_cardinality_count
            and unique_rate > config.high_cardinality_rate
        )
        identifier = bool(
            budget_ok
            and nn >= config.identifier_min_non_missing
            and unique_rate >= config.identifier_rate
        )
        families = len({key[0] for key in keys}) if budget_ok else pd.NA
        empty = sum(type(v) is str and v == "" for v in non)
        whitespace = sum(type(v) is str and v != "" and v.strip() == "" for v in non)
        rows["column_profile"].append(
            {
                "side": side,
                "column": column,
                "column_position": pos,
                "role": _role_for(column, roles),
                "pandas_dtype": str(series.dtype),
                "logical_type": schema_by[column].logical_type,
                "n_rows": n,
                "non_missing_count": nn,
                "missing_count": mc,
                "missing_rate": mc / n if n else pd.NA,
                "missing_status": "available" if n else "undefined",
                "missing_reason": "computed" if n else "no_rows",
                "mixed_python_type_count": families,
                "empty_string_count": empty,
                "whitespace_only_count": whitespace,
                "unique_count": unique,
                "unique_rate": unique_rate,
                "top_count": top,
                "top_rate": top_rate,
                "all_missing": all_missing,
                "constant": constant,
                "near_constant": near,
                "high_cardinality": high,
                "suspected_identifier": identifier,
                "value_profile_status": "available" if budget_ok else "not_verifiable",
                "value_profile_reason": "computed" if budget_ok else "budget_exceeded",
                "finding_key": pd.NA,
            }
        )
        detail = len(rows["column_profile"]) - 1
        rate = mc / n if n else None
        for condition, reason, value, threshold, severity_prov in (
            (all_missing, "all_missing_column", rate, 1.0, "pandas_structure"),
            (
                bool(n and rate >= config.missing_warning_rate and not all_missing),
                "high_missing_column",
                rate,
                config.missing_warning_rate,
                "pandas_structure",
            ),
            (constant, "constant_column", None, None, "pandas_structure"),
            (
                near,
                "near_constant_column",
                top_rate,
                config.near_constant_rate,
                "pandas_structure",
            ),
            (
                high,
                "high_cardinality_column",
                unique_rate,
                config.high_cardinality_rate,
                "pandas_structure",
            ),
            (
                identifier,
                "suspected_identifier",
                unique_rate,
                config.identifier_rate,
                "pandas_structure",
            ),
            (
                bool(budget_ok and families > 1),
                "mixed_python_types",
                float(families) if budget_ok else None,
                None,
                "pandas_structure",
            ),
            (bool(empty), "empty_strings", float(empty), None, "pandas_structure"),
            (
                bool(whitespace),
                "whitespace_only",
                float(whitespace),
                None,
                "pandas_structure",
            ),
        ):
            if condition:
                findings.append(
                    _finding(
                        "missingness" if "missing" in reason else "column_quality",
                        "column",
                        side,
                        column,
                        pos,
                        reason,
                        reason,
                        value,
                        threshold,
                        int(mc if "missing" in reason else nn),
                        n,
                        tuple(i for i, m in enumerate(missing) if m)[
                            : config.max_finding_samples
                        ]
                        if "missing" in reason
                        else (),
                        "column_profile",
                        detail,
                        severity_prov,
                    )
                )
        if is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype):
            floats = (
                np.asarray(non, dtype="float64")
                if non
                else np.array([], dtype="float64")
            )
            finite = floats[np.isfinite(floats)]
            fn = len(finite)
            posinf = int(np.isposinf(floats).sum())
            neginf = int(np.isneginf(floats).sum())
            loc = "available" if fn else "undefined"
            loc_reason = "computed" if fn else "no_finite_values"
            disp = "available" if fn >= 2 else "undefined"
            disp_reason = (
                "computed"
                if fn >= 2
                else ("insufficient_rows" if fn else "no_finite_values")
            )
            rows["numeric_profile"].append(
                {
                    "side": side,
                    "column": column,
                    "n_rows": n,
                    "non_missing_count": nn,
                    "missing_count": mc,
                    "finite_count": fn,
                    "positive_inf_count": posinf,
                    "negative_inf_count": neginf,
                    "mean": float(finite.mean()) if fn else pd.NA,
                    "std": float(finite.std(ddof=1)) if fn >= 2 else pd.NA,
                    "minimum": float(finite.min()) if fn else pd.NA,
                    "q25": float(np.quantile(finite, 0.25)) if fn else pd.NA,
                    "median": float(np.quantile(finite, 0.5)) if fn else pd.NA,
                    "q75": float(np.quantile(finite, 0.75)) if fn else pd.NA,
                    "maximum": float(finite.max()) if fn else pd.NA,
                    "count_status": "available",
                    "count_reason": "computed",
                    "finite_status": "available",
                    "finite_reason": "computed",
                    "location_status": loc,
                    "location_reason": loc_reason,
                    "dispersion_status": disp,
                    "dispersion_reason": disp_reason,
                    "range_status": loc,
                    "range_reason": loc_reason,
                    "quantile_status": loc,
                    "quantile_reason": loc_reason,
                    "finding_key": pd.NA,
                }
            )
            if posinf + neginf:
                findings.append(
                    _finding(
                        "column_quality",
                        "column",
                        side,
                        column,
                        pos,
                        "nonfinite_values",
                        "finite_count",
                        float(posinf + neginf),
                        None,
                        posinf + neginf,
                        nn,
                        tuple(
                            i
                            for i, v in enumerate(values)
                            if type(v) in (float, np.float16, np.float32, np.float64)
                            and np.isinf(v)
                        )[: config.max_finding_samples],
                        "numeric_profile",
                        len(rows["numeric_profile"]) - 1,
                        "pandas_structure",
                    )
                )
        elif schema_by[column].logical_type in {
            "categorical",
            "boolean",
            "text",
            "identifier",
        }:
            comparison_status = (
                "not_applicable" if side == "reference" else "unavailable"
            )
            comparison_reason = (
                "mode_not_applicable"
                if side == "reference"
                else "reference_not_provided"
            )
            rows["categorical_profile"].append(
                {
                    "side": side,
                    "column": column,
                    "non_missing_count": nn,
                    "unique_count": unique,
                    "unique_rate": unique_rate,
                    "top_count": top,
                    "top_rate": top_rate,
                    "singleton_level_count": sum(x == 1 for x in counts.values())
                    if budget_ok
                    else pd.NA,
                    "unseen_in_current_count": pd.NA,
                    "unseen_in_current_rate": pd.NA,
                    "count_status": "available",
                    "count_reason": "computed",
                    "cardinality_status": "available"
                    if budget_ok
                    else "not_verifiable",
                    "cardinality_reason": "computed"
                    if budget_ok
                    else "budget_exceeded",
                    "cardinality_rate_status": "available"
                    if nn and budget_ok
                    else "undefined",
                    "cardinality_rate_reason": "computed"
                    if nn and budget_ok
                    else "no_non_missing_values",
                    "frequency_status": "available"
                    if nn and budget_ok
                    else "undefined",
                    "frequency_reason": "computed"
                    if nn and budget_ok
                    else "no_non_missing_values",
                    "concentration_status": "available"
                    if nn and budget_ok
                    else "undefined",
                    "concentration_reason": "computed"
                    if nn and budget_ok
                    else "no_non_missing_values",
                    "comparison_status": comparison_status,
                    "comparison_reason": comparison_reason,
                    "finding_key": pd.NA,
                }
            )
    _target_rows(frame, side, roles, config, rows, findings)
    _slice_rows(frame, side, roles, config, rows, findings)
    return rows, findings, warnings


def _label_family(value: object) -> str:
    if value is None:
        return "not_declared"
    if type(value) in (np.bool_,):
        return "bool"
    if type(value) in (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ):
        return "int"
    if type(value) in (np.float16, np.float32, np.float64):
        return "float"
    return (
        "bool"
        if type(value) is bool
        else "int"
        if type(value) is int
        else "float"
        if type(value) is float
        else "str"
        if type(value) is str
        else "unsupported"
    )


def _target_rows(
    frame: pd.DataFrame,
    side: str,
    roles: DataAuditRoles,
    config: DataAuditConfig,
    rows: dict[str, list[dict[str, object]]],
    findings: list[dict[str, object]],
) -> None:
    if roles.target is None or roles.target not in frame.columns:
        return
    values = [frame[roles.target].iat[i] for i in range(len(frame))]
    present = [(i, v) for i, v in enumerate(values) if not _is_missing(v)]
    counts: dict[tuple[str, object], int] = {}
    first: dict[tuple[str, object], int] = {}
    for i, v in present:
        key = _value_key(v)
        counts[key] = counts.get(key, 0) + 1
        first.setdefault(key, i)
    positive = (
        _value_key(config.positive_label) if config.positive_label is not None else None
    )
    ordered = sorted(counts, key=lambda key: (0 if key == positive else 1, first[key]))
    binary = (
        "available"
        if len(counts) == 2
        else "not_verifiable"
        if any(key[0] not in {"bool", "int", "float", "str"} for key in counts)
        else "available"
    )
    binary_reason = "computed" if binary == "available" else "unsupported_target_values"
    for ordinal, key in enumerate(ordered):
        count = counts[key]
        is_positive = key == positive if positive is not None else pd.NA
        pos_status = (
            "available"
            if positive is not None and positive in counts and len(counts) == 2
            else ("unavailable" if positive is None else "not_verifiable")
        )
        pos_reason = (
            "computed"
            if pos_status == "available"
            else (
                "positive_label_not_declared"
                if positive is None
                else "unsupported_target_label"
            )
        )
        rows["target_profile"].append(
            {
                "side": side,
                "class_position": ordinal,
                "is_positive": is_positive,
                "positive_label_declared": positive is not None,
                "positive_label_type": _label_family(config.positive_label),
                "count": count,
                "rate": count / len(present) if present else pd.NA,
                "target_non_missing_n": len(present),
                "class_status": "available",
                "class_reason": "computed",
                "binary_status": binary,
                "binary_reason": binary_reason,
                "balance_status": "available" if len(counts) == 2 else "undefined",
                "balance_reason": "computed"
                if len(counts) == 2
                else "insufficient_support",
                "positive_class_status": pos_status,
                "positive_class_reason": pos_reason,
                "finding_key": pd.NA,
            }
        )
        rate = count / len(present) if present else 0.0
        if count < config.rare_class_count or rate < config.rare_class_rate:
            sample_positions = tuple(
                position for position, value in present if _value_key(value) == key
            )[: config.max_finding_samples]
            findings.append(
                _finding(
                    "target_quality",
                    "column",
                    side,
                    roles.target,
                    frame.columns.get_loc(roles.target),
                    "rare_target_class",
                    "class_rate",
                    rate,
                    config.rare_class_rate,
                    count,
                    len(present),
                    sample_positions,
                    "target_profile",
                    len(rows["target_profile"]) - 1,
                    "task15_label_semantics",
                )
            )
    pos = frame.columns.get_loc(roles.target)
    missing = len(values) - len(present)
    if missing:
        findings.append(
            _finding(
                "target_quality",
                "column",
                side,
                roles.target,
                pos,
                "target_missing",
                "missing_count",
                float(missing),
                config.missing_warning_rate,
                missing,
                len(values),
                tuple(i for i, v in enumerate(values) if _is_missing(v))[
                    : config.max_finding_samples
                ],
                "target_profile",
                0,
                "task15_label_semantics",
            )
        )
    if len(counts) == 1:
        findings.append(
            _finding(
                "target_quality",
                "column",
                side,
                roles.target,
                pos,
                "target_constant",
                "unique_count",
                1,
                None,
                len(present),
                len(present),
                (),
                "target_profile",
                0,
                "task15_label_semantics",
            )
        )
    elif len(counts) != 2:
        findings.append(
            _finding(
                "target_quality",
                "column",
                side,
                roles.target,
                pos,
                "target_non_binary",
                "unique_count",
                float(len(counts)),
                None,
                len(counts),
                len(present),
                (),
                "target_profile",
                0,
                "task15_label_semantics",
            )
        )


def _slice_rows(
    frame: pd.DataFrame,
    side: str,
    roles: DataAuditRoles,
    config: DataAuditConfig,
    rows: dict[str, list[dict[str, object]]],
    findings: list[dict[str, object]],
) -> None:
    positive_key = (
        _value_key(config.positive_label) if config.positive_label is not None else None
    )

    def target_metrics(positions: list[int]) -> tuple[object, ...]:
        if roles.target is None or roles.target not in frame.columns:
            return (
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                "unavailable",
                "target_not_declared",
                "unavailable",
                "target_not_declared",
            )
        target_values = [frame[roles.target].iat[i] for i in positions]
        present = [value for value in target_values if not _is_missing(value)]
        target_count = len(present)
        target_rate = target_count / len(positions) if positions else pd.NA
        if positive_key is None:
            return (
                target_count,
                target_rate,
                pd.NA,
                pd.NA,
                "available",
                "computed",
                "unavailable",
                "positive_label_not_declared",
            )
        positive_count = sum(_value_key(value) == positive_key for value in present)
        event_rate = positive_count / target_count if target_count else pd.NA
        return (
            target_count,
            target_rate,
            positive_count,
            event_rate,
            "available" if positions else "undefined",
            "computed" if positions else "no_rows",
            "available" if target_count else "undefined",
            "computed" if target_count else "no_non_missing_values",
        )

    for role_name in ("partition", "fold"):
        column = getattr(roles, role_name)
        if column is None:
            rows["slice_profile"].append(
                {
                    "side": side,
                    "slice_role": role_name,
                    "row_kind": "summary",
                    "slice_ordinal": pd.NA,
                    "partition_ordinal": pd.NA,
                    "fold_ordinal": pd.NA,
                    "missing_bucket": False,
                    "row_count": pd.NA,
                    "target_non_missing_count": pd.NA,
                    "target_non_missing_rate": pd.NA,
                    "positive_count": pd.NA,
                    "event_rate": pd.NA,
                    "size_status": "not_applicable",
                    "size_reason": "role_not_declared",
                    "target_rate_status": "not_applicable",
                    "target_rate_reason": "role_not_declared",
                    "event_status": "not_applicable",
                    "event_reason": "role_not_declared",
                    "quality_status": "not_applicable",
                    "quality_reason": "role_not_declared",
                    "finding_key": pd.NA,
                }
            )
    for role_name in (
        "group",
        "partition",
        "fold",
        "selection",
        "historical_action",
        "historical_policy",
    ):
        column = getattr(roles, role_name)
        if column is None:
            continue
        values = [frame[column].iat[i] for i in range(len(frame))]
        groups: dict[tuple[str, object], list[int]] = {}
        order: dict[tuple[str, object], int] = {}
        for i, value in enumerate(values):
            key = ("missing", 0) if _is_missing(value) else _value_key(value)
            groups.setdefault(key, []).append(i)
            order.setdefault(key, i)
        summary_detail = len(rows["slice_profile"])
        rows["slice_profile"].append(
            {
                "side": side,
                "slice_role": role_name,
                "row_kind": "summary",
                "slice_ordinal": pd.NA,
                "partition_ordinal": pd.NA,
                "fold_ordinal": pd.NA,
                "missing_bucket": False,
                "row_count": len(frame),
                "target_non_missing_count": target_metrics(list(range(len(frame))))[0],
                "target_non_missing_rate": target_metrics(list(range(len(frame))))[1],
                "positive_count": target_metrics(list(range(len(frame))))[2],
                "event_rate": target_metrics(list(range(len(frame))))[3],
                "size_status": "available",
                "size_reason": "computed",
                "target_rate_status": target_metrics(list(range(len(frame))))[4],
                "target_rate_reason": target_metrics(list(range(len(frame))))[5],
                "event_status": target_metrics(list(range(len(frame))))[6],
                "event_reason": target_metrics(list(range(len(frame))))[7],
                "quality_status": "available",
                "quality_reason": "computed",
                "finding_key": pd.NA,
            }
        )
        value_details: list[
            tuple[int, tuple[str, object], list[int], tuple[object, ...]]
        ] = []
        for ordinal, key in enumerate(sorted(groups, key=lambda k: order[k])):
            positions = groups[key]
            metrics = target_metrics(positions)
            missing = key[0] == "missing"
            reason = (
                "missing_partition_value"
                if role_name == "partition"
                else "missing_fold_value"
                if role_name == "fold"
                else "missing_slice_value"
            )
            quality = "not_verifiable" if missing else "available"
            rows["slice_profile"].append(
                {
                    "side": side,
                    "slice_role": role_name,
                    "row_kind": "value",
                    "slice_ordinal": ordinal,
                    "partition_ordinal": ordinal if role_name == "partition" else pd.NA,
                    "fold_ordinal": ordinal if role_name == "fold" else pd.NA,
                    "missing_bucket": missing,
                    "row_count": len(positions),
                    "target_non_missing_count": metrics[0],
                    "target_non_missing_rate": metrics[1],
                    "positive_count": metrics[2],
                    "event_rate": metrics[3],
                    "size_status": "available",
                    "size_reason": "computed",
                    "target_rate_status": metrics[4],
                    "target_rate_reason": metrics[5],
                    "event_status": metrics[6],
                    "event_reason": metrics[7],
                    "quality_status": quality,
                    "quality_reason": reason if missing else "computed",
                    "finding_key": pd.NA,
                }
            )
            value_details.append(
                (len(rows["slice_profile"]) - 1, key, positions, metrics)
            )
            if missing and role_name in {"partition", "fold"}:
                findings.append(
                    _finding(
                        "partition_leakage",
                        "partition" if role_name == "partition" else "fold",
                        side,
                        column,
                        frame.columns.get_loc(column),
                        reason,
                        "missing_count",
                        float(len(positions)),
                        None,
                        len(positions),
                        len(frame),
                        tuple(positions[: config.max_finding_samples]),
                        "slice_profile",
                        len(rows["slice_profile"]) - 1,
                        "caller_roles",
                    )
                )
        if role_name == "selection":
            support_rates = [
                (detail, float(metrics[1]))
                for detail, key, _, metrics in value_details
                if key[0] != "missing" and not pd.isna(metrics[1])
            ]
            if len(support_rates) >= 2:
                minimum_rate = min(rate for _, rate in support_rates)
                maximum_rate = max(rate for _, rate in support_rates)
                gap = maximum_rate - minimum_rate
                if gap >= config.missingness_drift_absolute_threshold:
                    findings.append(
                        _finding(
                            "target_quality",
                            "row_set",
                            side,
                            column,
                            frame.columns.get_loc(column),
                            "selection_outcome_support_gap",
                            "target_non_missing_rate_gap",
                            gap,
                            config.missingness_drift_absolute_threshold,
                            None,
                            len(frame),
                            (),
                            "slice_profile",
                            summary_detail,
                            "caller_roles|task15_label_semantics",
                        )
                    )
        if (
            role_name == "partition"
            and config.partition_target_rate_shift_threshold is not None
            and roles.target is not None
            and positive_key is not None
        ):
            pooled = target_metrics(list(range(len(frame))))
            pooled_support = int(pooled[0])
            pooled_rate = pooled[3]
            target_keys = {
                _value_key(frame[roles.target].iat[position])
                for position in range(len(frame))
                if not _is_missing(frame[roles.target].iat[position])
            }
            evaluable = len(target_keys) == 2 and not pd.isna(pooled_rate)
            for detail, key, positions, metrics in value_details:
                if key[0] == "missing":
                    continue
                support = int(metrics[0])
                if (
                    not evaluable
                    or support < config.partition_target_min_support
                    or pooled_support < config.partition_target_min_support
                ):
                    rows["slice_profile"][detail]["quality_status"] = "undefined"
                    rows["slice_profile"][detail]["quality_reason"] = (
                        "insufficient_support"
                    )
                    continue
                shift = abs(float(metrics[3]) - float(pooled_rate))
                if shift >= config.partition_target_rate_shift_threshold:
                    findings.append(
                        _finding(
                            "partition_leakage",
                            "partition",
                            side,
                            column,
                            frame.columns.get_loc(column),
                            "target_distribution_shift",
                            "event_rate_shift",
                            shift,
                            config.partition_target_rate_shift_threshold,
                            support,
                            pooled_support,
                            tuple(positions[: config.max_finding_samples]),
                            "slice_profile",
                            detail,
                            "caller_roles|task15_label_semantics",
                        )
                    )


def _finding(
    category: str,
    scope: str,
    dataset_role: str,
    column: str | None,
    column_position: int,
    reason: str,
    metric_key: str,
    value: float | None,
    threshold: float | None,
    count: int | None,
    denominator: int | None,
    samples: tuple[int, ...],
    detail_table: str,
    detail: int,
    provenance: str,
) -> dict[str, object]:
    severity = (
        "error"
        if reason
        in {
            "empty_dataset",
            "all_missing_column",
            "datetime_role_mismatch",
            "audit_input_dtype_mismatch",
            "target_constant",
            "target_non_binary",
            "target_in_features",
            "exact_target_copy",
            "post_outcome_feature",
            "range_violation",
            "allowed_value_violation",
            "cross_column_order_violation",
            "monotonic_time_violation",
            "feature_after_observation",
            "observation_after_outcome_end",
            "event_before_observation",
            "event_after_outcome_end",
            "label_before_outcome_end",
            "observation_at_or_after_cutoff",
            "feature_after_cutoff",
            "window_start_after_window_end",
            "window_end_after_analysis_as_of",
            "horizon_end_after_analysis_as_of",
        }
        else "info"
        if reason
        in {
            "duplicate_index",
            "high_cardinality_column",
            "suspected_identifier",
            "whitespace_only",
            "target_distribution_shift",
            "singleton_group",
            "budget_exceeded",
        }
        else "warning"
    )
    samples = tuple(int(item) for item in samples)
    row = {
        "finding_key": "",
        "category": category,
        "scope": scope,
        "dataset_role": dataset_role,
        "column": column if column is not None else pd.NA,
        "column_position": column_position if column is not None else pd.NA,
        "role": "unassigned",
        "severity": severity,
        "status": "available",
        "reason": reason,
        "metric_key": metric_key,
        "value": value if value is not None else pd.NA,
        "threshold": threshold if threshold is not None else pd.NA,
        "count": count if count is not None else pd.NA,
        "denominator": denominator if denominator is not None else pd.NA,
        "affected_rate": (
            count / denominator if count is not None and denominator else pd.NA
        ),
        "sample_positions": tuple(samples),
        "detail_table": detail_table,
        "detail_row_ordinal": detail,
        "recommendation": "review evidence and source data",
        "limitation": "association_not_causation"
        if reason in {"near_target_copy", "deterministic_categorical_proxy"}
        else "no_automatic_leakage_repair",
        "provenance": provenance,
    }
    row["finding_key"] = _finding_key(row)
    return row


def _finding_key(row: dict[str, object]) -> str:
    samples = row["sample_positions"]
    sample = min(samples) if samples else -1
    column_position = row["column_position"]
    payload = [
        row["category"],
        row["scope"],
        row["dataset_role"],
        -1 if pd.isna(column_position) else int(column_position),
        row["metric_key"],
        row["reason"],
        row["detail_table"],
        int(row["detail_row_ordinal"]),
        sample,
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _relink_findings(
    findings: list[dict[str, object]], tables: dict[str, pd.DataFrame]
) -> None:
    column_tables = {
        "column_profile",
        "numeric_profile",
        "categorical_profile",
        "missingness_drift",
        "schema_drift",
    }
    for finding in findings:
        table_name = finding["detail_table"]
        table = tables.get(str(table_name))
        column = finding["column"]
        if table_name == "dataset_profile" and table is not None:
            positions = np.flatnonzero(
                table["side"].eq(finding["dataset_role"]).to_numpy()
            )
            if positions.size != 1:
                raise RuntimeError(
                    "data audit internal detail linkage invariant failed"
                )
            finding["detail_row_ordinal"] = int(positions[0])
            finding["finding_key"] = _finding_key(finding)
            continue
        if table_name == "target_profile" and table is not None:
            mask = table["side"].eq(finding["dataset_role"])
            positions = np.flatnonzero(mask.to_numpy())
            if positions.size:
                local_ordinal = min(
                    int(finding["detail_row_ordinal"]), positions.size - 1
                )
                finding["detail_row_ordinal"] = int(positions[local_ordinal])
            finding["finding_key"] = _finding_key(finding)
            continue
        if table is None or table_name not in column_tables or pd.isna(column):
            continue
        mask = table["column"].eq(column)
        if "side" in table.columns and finding["dataset_role"] in {
            "current",
            "reference",
        }:
            mask &= table["side"].eq(finding["dataset_role"])
        positions = np.flatnonzero(mask.to_numpy())
        if positions.size:
            finding["detail_row_ordinal"] = int(positions[0])
        finding["finding_key"] = _finding_key(finding)


def _pattern_counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, list[int]]:
    patterns: dict[str, list[int]] = {}
    for position in range(len(frame)):
        bits = "".join(
            "1" if _is_missing(frame[column].iat[position]) else "0"
            for column in columns
        )
        patterns.setdefault(bits, []).append(position)
    return patterns


def _missing_patterns(
    data: pd.DataFrame,
    reference: pd.DataFrame | None,
    columns: list[str],
    config: DataAuditConfig,
) -> tuple[pd.DataFrame, bool]:
    if not len(data):
        row = {
            "pattern_key": "__NO_ROWS__",
            "pattern_bits": pd.NA,
            "aggregated": False,
            "source_pattern_count": 0,
            "missing_count": pd.NA,
            "row_count": 0,
            "row_rate": pd.NA,
            "missing_cell_count": 0,
            "min_missing_count": pd.NA,
            "max_missing_count": pd.NA,
            "sample_positions": (),
            "reference_row_count": pd.NA,
            "reference_row_rate": pd.NA,
            "absolute_rate_change": pd.NA,
            "count_status": "available",
            "count_reason": "computed",
            "rate_status": "undefined",
            "rate_reason": "no_rows",
            "reference_count_status": "undefined"
            if reference is not None
            else "unavailable",
            "reference_count_reason": "no_rows"
            if reference is not None
            else "reference_not_provided",
            "reference_rate_status": "undefined"
            if reference is not None
            else "unavailable",
            "reference_rate_reason": "no_rows"
            if reference is not None
            else "reference_not_provided",
            "comparison_status": "undefined"
            if reference is not None
            else "unavailable",
            "comparison_reason": "no_rows"
            if reference is not None
            else "reference_not_provided",
            "finding_key": pd.NA,
        }
        return _frame("missingness_patterns", [row]), False
    current = _pattern_counts(data, columns)
    schema_match = reference is None or all(
        column in reference.columns for column in columns
    )
    baseline = (
        _pattern_counts(reference, columns)
        if reference is not None and schema_match
        else {}
    )
    identities = set(current) | set(baseline)
    ranked = sorted(
        identities,
        key=lambda bits: (
            -(len(current.get(bits, ())) + len(baseline.get(bits, ()))),
            bits,
        ),
    )
    truncated = len(ranked) > config.max_missing_patterns
    kept = (
        ranked if not truncated else ranked[: max(0, config.max_missing_patterns - 1)]
    )
    rows = []
    for bits in kept:
        positions = current.get(bits, [])
        ref_count = len(baseline.get(bits, []))
        cr = len(positions) / len(data)
        rr = (
            ref_count / len(reference)
            if reference is not None and len(reference)
            else pd.NA
        )
        rows.append(
            {
                "pattern_key": "p:" + bits,
                "pattern_bits": bits,
                "aggregated": False,
                "source_pattern_count": 1,
                "missing_count": bits.count("1"),
                "row_count": len(positions),
                "row_rate": cr,
                "missing_cell_count": bits.count("1") * len(positions),
                "min_missing_count": bits.count("1"),
                "max_missing_count": bits.count("1"),
                "sample_positions": tuple(positions[: config.max_finding_samples]),
                "reference_row_count": ref_count
                if reference is not None and schema_match
                else pd.NA,
                "reference_row_rate": rr,
                "absolute_rate_change": cr - rr
                if reference is not None and schema_match and len(reference)
                else pd.NA,
                "count_status": "available",
                "count_reason": "computed",
                "rate_status": "available",
                "rate_reason": "computed",
                "reference_count_status": "available"
                if reference is not None and schema_match
                else ("not_verifiable" if reference is not None else "unavailable"),
                "reference_count_reason": "computed"
                if reference is not None and schema_match
                else (
                    "pattern_schema_mismatch"
                    if reference is not None
                    else "reference_not_provided"
                ),
                "reference_rate_status": "available"
                if reference is not None and schema_match and len(reference)
                else (
                    "undefined"
                    if reference is not None and schema_match
                    else "not_verifiable"
                    if reference is not None
                    else "unavailable"
                ),
                "reference_rate_reason": "computed"
                if reference is not None and schema_match and len(reference)
                else (
                    "no_rows"
                    if reference is not None and schema_match
                    else "pattern_schema_mismatch"
                    if reference is not None
                    else "reference_not_provided"
                ),
                "comparison_status": "available"
                if reference is not None and schema_match and len(reference)
                else (
                    "undefined"
                    if reference is not None and schema_match
                    else "not_verifiable"
                    if reference is not None
                    else "unavailable"
                ),
                "comparison_reason": "computed"
                if reference is not None and schema_match and len(reference)
                else (
                    "no_rows"
                    if reference is not None and schema_match
                    else "pattern_schema_mismatch"
                    if reference is not None
                    else "reference_not_provided"
                ),
                "finding_key": pd.NA,
            }
        )
    if truncated:
        other = ranked[len(kept) :]
        positions = sorted(i for bits in other for i in current.get(bits, []))
        ref_count = sum(len(baseline.get(bits, [])) for bits in other)
        counts = [bits.count("1") for bits in other]
        rows.append(
            {
                "pattern_key": "__OTHER__",
                "pattern_bits": pd.NA,
                "aggregated": True,
                "source_pattern_count": len(other),
                "missing_count": pd.NA,
                "row_count": len(positions),
                "row_rate": len(positions) / len(data),
                "missing_cell_count": sum(
                    bits.count("1") * len(current.get(bits, [])) for bits in other
                ),
                "min_missing_count": min(counts),
                "max_missing_count": max(counts),
                "sample_positions": tuple(positions[: config.max_finding_samples]),
                "reference_row_count": ref_count
                if reference is not None and schema_match
                else pd.NA,
                "reference_row_rate": ref_count / len(reference)
                if reference is not None and schema_match and len(reference)
                else pd.NA,
                "absolute_rate_change": len(positions) / len(data)
                - ref_count / len(reference)
                if reference is not None and schema_match and len(reference)
                else pd.NA,
                "count_status": "available",
                "count_reason": "computed",
                "rate_status": "available",
                "rate_reason": "computed",
                "reference_count_status": "available"
                if reference is not None and schema_match
                else "unavailable",
                "reference_count_reason": "computed"
                if reference is not None and schema_match
                else "reference_not_provided",
                "reference_rate_status": "available"
                if reference is not None and schema_match and len(reference)
                else "undefined",
                "reference_rate_reason": "computed"
                if reference is not None and schema_match and len(reference)
                else "no_rows",
                "comparison_status": "available"
                if reference is not None and schema_match and len(reference)
                else "undefined",
                "comparison_reason": "computed"
                if reference is not None and schema_match and len(reference)
                else "no_rows",
                "finding_key": pd.NA,
            }
        )
    return _frame("missingness_patterns", rows), truncated


def _drift_tables(
    data: pd.DataFrame,
    reference: pd.DataFrame | None,
    roles: DataAuditRoles,
    current_schema: object,
    reference_schema: object | None,
    config: DataAuditConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    if reference is None:
        return _frame("missingness_drift", []), _frame("schema_drift", []), []
    union = list(reference.columns) + [
        column for column in data.columns if column not in reference.columns
    ]
    current_map = {item.name: item for item in current_schema.columns}
    reference_map = {item.name: item for item in reference_schema.columns}
    missing_rows = []
    schema_rows = []
    findings = []
    for column in union:
        rp = column in reference.columns
        cp = column in data.columns
        rn = len(reference) if rp else pd.NA
        cn = len(data) if cp else pd.NA
        rmc = int(reference[column].isna().sum()) if rp else pd.NA
        cmc = int(data[column].isna().sum()) if cp else pd.NA
        rr = rmc / rn if rp and rn else pd.NA
        cr = cmc / cn if cp and cn else pd.NA
        both = rp and cp
        abs_change = cr - rr if both and rn and cn else pd.NA
        rel = (cr - rr) / rr if both and rn and cn and rr else pd.NA
        rel_status = (
            "available"
            if both and rn and cn and rr
            else "undefined"
            if both
            else "unavailable"
        )
        rel_reason = (
            "computed"
            if rel_status == "available"
            else (
                "zero_baseline_increase"
                if both and rn and cn and rr == 0 and cr > 0
                else "zero_baseline_no_change"
                if both and rn and cn and rr == 0
                else "no_rows"
                if both
                else "current_column_added"
                if cp
                else "current_column_missing"
            )
        )
        missing_rows.append(
            {
                "column": column,
                "reference_present": rp,
                "current_present": cp,
                "reference_n": rn,
                "current_n": cn,
                "reference_missing_count": rmc,
                "current_missing_count": cmc,
                "reference_missing_rate": rr,
                "current_missing_rate": cr,
                "absolute_rate_change": abs_change,
                "relative_rate_change": rel,
                "new_all_missing": bool(
                    both and cn and cmc == cn and (not rn or rmc != rn)
                ),
                "recovered": bool(both and rn and rmc == rn and cn and cmc != cn),
                "count_status": "available" if both else "unavailable",
                "count_reason": "computed"
                if both
                else "current_column_added"
                if cp
                else "current_column_missing",
                "rate_status": "available"
                if both and rn and cn
                else "undefined"
                if both
                else "unavailable",
                "rate_reason": "computed"
                if both and rn and cn
                else "no_rows"
                if both
                else "current_column_added"
                if cp
                else "current_column_missing",
                "reference_count_status": "available" if rp else "unavailable",
                "reference_count_reason": "computed" if rp else "current_column_added",
                "current_count_status": "available" if cp else "unavailable",
                "current_count_reason": "computed" if cp else "current_column_missing",
                "reference_rate_status": "available"
                if rp and rn
                else "undefined"
                if rp
                else "unavailable",
                "reference_rate_reason": "computed"
                if rp and rn
                else "no_rows"
                if rp
                else "current_column_added",
                "current_rate_status": "available"
                if cp and cn
                else "undefined"
                if cp
                else "unavailable",
                "current_rate_reason": "computed"
                if cp and cn
                else "no_rows"
                if cp
                else "current_column_missing",
                "absolute_change_status": "available"
                if both and rn and cn
                else "undefined"
                if both
                else "unavailable",
                "absolute_change_reason": "computed"
                if both and rn and cn
                else "no_rows"
                if both
                else "current_column_added"
                if cp
                else "current_column_missing",
                "relative_change_status": rel_status,
                "relative_change_reason": rel_reason,
                "finding_key": pd.NA,
            }
        )
        rd = str(reference[column].dtype) if rp else pd.NA
        cd = str(data[column].dtype) if cp else pd.NA
        rl = reference_map[column].logical_type if rp else pd.NA
        cl = current_map[column].logical_type if cp else pd.NA
        dtype_changed = bool(both and rd != cd)
        logical_changed = bool(both and rl != cl)
        added = cp and not rp
        removed = rp and not cp
        role = _role_for(column, roles)
        primary = (
            "removed"
            if removed
            else "added"
            if added
            else "dtype_and_logical_type_changed"
            if dtype_changed and logical_changed
            else "dtype_changed"
            if dtype_changed
            else "logical_type_changed"
            if logical_changed
            else "unchanged"
        )
        reason = (
            "current_column_added"
            if added
            else "current_column_missing"
            if removed
            else "same"
            if primary == "unchanged"
            else "computed"
        )
        status = "unavailable" if added or removed else "available"
        schema_rows.append(
            {
                "column": column,
                "reference_position": reference.columns.get_loc(column)
                if rp
                else pd.NA,
                "current_position": data.columns.get_loc(column) if cp else pd.NA,
                "reference_dtype": rd,
                "current_dtype": cd,
                "reference_logical_type": rl,
                "current_logical_type": cl,
                "reference_role": role if rp else pd.NA,
                "current_role": role if cp else pd.NA,
                "column_added": added,
                "column_removed": removed,
                "dtype_changed": dtype_changed,
                "logical_type_changed": logical_changed,
                "role_changed": False if both else pd.NA,
                "primary_change": primary,
                "status": status,
                "reason": reason,
                "finding_key": pd.NA,
            }
        )
        detail = len(schema_rows) - 1
        if added or removed or dtype_changed or logical_changed:
            finding_reason = (
                "current_column_added"
                if added
                else "current_column_missing"
                if removed
                else "dtype_changed"
                if dtype_changed
                else "logical_type_changed"
            )
            findings.append(
                _finding(
                    "schema_drift",
                    "comparison",
                    "comparison",
                    column,
                    data.columns.get_loc(column)
                    if cp
                    else reference.columns.get_loc(column),
                    finding_reason,
                    "primary_change",
                    None,
                    None,
                    None,
                    None,
                    (),
                    "schema_drift",
                    detail,
                    "reference_current_comparison",
                )
            )
        if (
            both
            and rn
            and cn
            and (
                abs(abs_change) >= config.missingness_drift_absolute_threshold
                or (
                    rel_status == "available"
                    and abs(rel) >= config.missingness_drift_relative_threshold
                )
            )
        ):
            findings.append(
                _finding(
                    "missingness_drift",
                    "comparison",
                    "comparison",
                    column,
                    data.columns.get_loc(column),
                    "missingness_rate_changed",
                    "absolute_rate_change",
                    float(abs_change),
                    config.missingness_drift_absolute_threshold,
                    None,
                    None,
                    (),
                    "missingness_drift",
                    len(missing_rows) - 1,
                    "reference_current_comparison",
                )
            )
    return (
        _frame("missingness_drift", missing_rows),
        _frame("schema_drift", schema_rows),
        findings,
    )


def _collinearity(
    data: pd.DataFrame, roles: DataAuditRoles, config: DataAuditConfig
) -> tuple[pd.DataFrame, bool]:
    features = list(roles.features or ())
    numeric = [
        column
        for column in features
        if is_numeric_dtype(data[column].dtype)
        and not is_bool_dtype(data[column].dtype)
    ]
    truncated = len(numeric) > config.max_collinearity_columns
    numeric = numeric[: config.max_collinearity_columns]
    rows = []
    for i, left in enumerate(numeric):
        for right in numeric[i + 1 :]:
            a = pd.to_numeric(data[left], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(data[right], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(a) & np.isfinite(b)
            valid = int(mask.sum())
            if (
                valid >= config.collinearity_min_periods
                and np.std(a[mask]) > 0
                and np.std(b[mask]) > 0
            ):
                value = float(np.corrcoef(a[mask], b[mask])[0, 1])
                if abs(value) >= config.collinearity_threshold:
                    rows.append(
                        {
                            "left_column": left,
                            "right_column": right,
                            "valid_n": valid,
                            "pearson_r": value,
                            "absolute_r": abs(value),
                            "threshold": config.collinearity_threshold,
                            "status": "available",
                            "reason": "computed",
                            "finding_key": pd.NA,
                        }
                    )
    return _frame("collinearity", rows), truncated


def _atomic_violations(
    data: pd.DataFrame, column: str, operator: str, right: _ConditionOperand
) -> tuple[list[int], list[int]]:
    try:
        result = _evaluate_atomic_condition(
            data,
            operator=operator,
            left=_ConditionOperand("column", column),
            right=right,
            root_version=_KERNEL_VERSION,
        )
    except ValueError as exc:
        key = str(exc).rsplit(": ", 1)[-1]
        if key in {"membership_budget_exceeded", "unsupported_scalar_type"}:
            raise _audit_config_error(key) from exc
        raise
    return [int(i) for i, v in enumerate(result.truth) if v == "false"], [
        int(i) for i, v in enumerate(result.truth) if v == "unknown"
    ]


def _rule_findings(
    data: pd.DataFrame, roles: DataAuditRoles, config: DataAuditConfig
) -> list[dict[str, object]]:
    findings = []
    for rule in config.column_rules:
        if rule.column not in data.columns:
            raise _audit_input_error("unknown_selector")
        specs = []
        if len(rule.not_after_columns) != len(set(rule.not_after_columns)):
            raise _audit_config_error("duplicate_selector")
        if rule.column in rule.not_after_columns:
            raise _audit_config_error("conflicting_roles")
        if (
            type(rule.minimum_inclusive) is not bool
            or type(rule.maximum_inclusive) is not bool
            or type(rule.nondecreasing) is not bool
        ):
            raise _audit_config_error("unsupported_rule_literal")
        if rule.minimum is not None and rule.maximum is not None:
            bound_frame = pd.DataFrame(
                {"lower": [rule.minimum], "upper": [rule.maximum]}
            )
            try:
                bound = _evaluate_atomic_condition(
                    bound_frame,
                    operator="le",
                    left=_ConditionOperand("column", "lower"),
                    right=_ConditionOperand("column", "upper"),
                    root_version=_KERNEL_VERSION,
                )
            except ValueError as exc:
                raise _audit_config_error("unsupported_rule_literal") from exc
            if bound.truth.iat[0] != "true":
                raise _audit_config_error("unsupported_rule_literal")
        if rule.minimum is not None:
            specs.append(
                (
                    "ge" if rule.minimum_inclusive else "gt",
                    _ConditionOperand("literal", rule.minimum),
                    "range_violation",
                )
            )
        if rule.maximum is not None:
            specs.append(
                (
                    "le" if rule.maximum_inclusive else "lt",
                    _ConditionOperand("literal", rule.maximum),
                    "range_violation",
                )
            )
        if rule.allowed_values:
            specs.append(
                (
                    "in",
                    _ConditionOperand("literal", rule.allowed_values),
                    "allowed_value_violation",
                )
            )
        if rule.special_values:
            violations, unknown = _atomic_violations(
                data,
                rule.column,
                "in",
                _ConditionOperand("literal", rule.special_values),
            )
            matches = [
                i for i in range(len(data)) if i not in violations and i not in unknown
            ]
            if matches:
                findings.append(
                    _finding(
                        "column_quality",
                        "row_set",
                        "current",
                        rule.column,
                        data.columns.get_loc(rule.column),
                        "special_value_present",
                        "special_in",
                        None,
                        None,
                        len(matches),
                        len(data),
                        tuple(matches[: config.max_finding_samples]),
                        "column_profile",
                        0,
                        "caller_column_rule|condition_kernel_atomic",
                    )
                )
        for operator, right, reason in specs:
            false, unknown = _atomic_violations(data, rule.column, operator, right)
            if false:
                findings.append(
                    _finding(
                        "constraint_violation",
                        "row_set",
                        "current",
                        rule.column,
                        data.columns.get_loc(rule.column),
                        reason,
                        operator,
                        None,
                        None,
                        len(false),
                        len(data),
                        tuple(false[: config.max_finding_samples]),
                        "column_profile",
                        0,
                        "caller_column_rule|condition_kernel_atomic",
                    )
                )
        for other in rule.not_after_columns:
            if other not in data.columns:
                raise _audit_input_error("unknown_selector")
            false, _ = _atomic_violations(
                data, rule.column, "le", _ConditionOperand("column", other)
            )
            if false:
                findings.append(
                    _finding(
                        "constraint_violation",
                        "row_set",
                        "current",
                        rule.column,
                        data.columns.get_loc(rule.column),
                        "cross_column_order_violation",
                        "not_after_le",
                        None,
                        None,
                        len(false),
                        len(data),
                        tuple(false[: config.max_finding_samples]),
                        "column_profile",
                        0,
                        "caller_column_rule|condition_kernel_atomic",
                    )
                )
        if rule.nondecreasing and len(data) > 1:
            prior_positions = list(range(len(data) - 1))
            current_positions = list(range(1, len(data)))
            if prior_positions:
                pair = pd.DataFrame(
                    {
                        "prior": [data[rule.column].iat[i] for i in prior_positions],
                        "current": [
                            data[rule.column].iat[i] for i in current_positions
                        ],
                    }
                )
                eligible = list(range(len(pair)))
                if roles.group is not None:
                    pair["prior_group"] = [
                        data[roles.group].iat[i] for i in prior_positions
                    ]
                    pair["current_group"] = [
                        data[roles.group].iat[i] for i in current_positions
                    ]
                    group_comparison = _evaluate_atomic_condition(
                        pair,
                        operator="eq",
                        left=_ConditionOperand("column", "prior_group"),
                        right=_ConditionOperand("column", "current_group"),
                        root_version=_KERNEL_VERSION,
                    )
                    eligible = [
                        i
                        for i, truth in enumerate(group_comparison.truth)
                        if truth == "true"
                    ]
                comparison = _evaluate_atomic_condition(
                    pair,
                    operator="le",
                    left=_ConditionOperand("column", "prior"),
                    right=_ConditionOperand("column", "current"),
                    root_version=_KERNEL_VERSION,
                )
                violating = [
                    current_positions[i]
                    for i, truth in enumerate(comparison.truth)
                    if i in eligible and truth == "false"
                ]
                if violating:
                    findings.append(
                        _finding(
                            "constraint_violation",
                            "row_set",
                            "current",
                            rule.column,
                            data.columns.get_loc(rule.column),
                            "monotonic_time_violation",
                            "nondecreasing_le",
                            None,
                            None,
                            len(violating),
                            len(pair),
                            tuple(violating[: config.max_finding_samples]),
                            "column_profile",
                            0,
                            "caller_column_rule|condition_kernel_atomic",
                        )
                    )
    return findings


def _roadmap_findings(
    data: pd.DataFrame, roles: DataAuditRoles, config: DataAuditConfig
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for role_name, columns in (
        ("score", roles.score_columns),
        ("cost", roles.cost_columns),
        ("exposure", roles.exposure_columns),
    ):
        for column in columns:
            position = data.columns.get_loc(column)
            if not is_numeric_dtype(data[column].dtype) or is_bool_dtype(
                data[column].dtype
            ):
                findings.append(
                    _finding(
                        "column_quality",
                        "column",
                        "current",
                        column,
                        position,
                        "audit_input_dtype_mismatch",
                        "pandas_dtype",
                        None,
                        None,
                        None,
                        len(data),
                        (),
                        "column_profile",
                        position,
                        "caller_roles",
                    )
                )
            if (
                role_name == "exposure"
                and is_numeric_dtype(data[column].dtype)
                and not is_bool_dtype(data[column].dtype)
            ):
                result = _evaluate_atomic_condition(
                    data,
                    operator="lt",
                    left=_ConditionOperand("column", column),
                    right=_ConditionOperand("literal", 0),
                    root_version=_KERNEL_VERSION,
                )
                positions = [
                    i for i, truth in enumerate(result.truth) if truth == "true"
                ]
                if positions:
                    findings.append(
                        _finding(
                            "column_quality",
                            "row_set",
                            "current",
                            column,
                            position,
                            "negative_exposure",
                            "negative_count",
                            float(len(positions)),
                            0.0,
                            len(positions),
                            len(data),
                            tuple(positions[: config.max_finding_samples]),
                            "numeric_profile",
                            0,
                            "caller_roles|condition_kernel_atomic",
                        )
                    )
    if roles.score_columns and roles.partition is None and roles.fold is None:
        findings.append(
            _finding(
                "column_quality",
                "dataset",
                "current",
                None,
                -1,
                "score_partition_provenance_missing",
                "score_columns",
                None,
                None,
                len(roles.score_columns),
                len(roles.score_columns),
                (),
                "dataset_profile",
                0,
                "caller_roles",
            )
        )
    for column in roles.constraint_input_columns:
        positions = [i for i in range(len(data)) if _is_missing(data[column].iat[i])]
        if positions:
            findings.append(
                _finding(
                    "column_quality",
                    "row_set",
                    "current",
                    column,
                    data.columns.get_loc(column),
                    "constraint_input_missing",
                    "missing_count",
                    float(len(positions)),
                    None,
                    len(positions),
                    len(data),
                    tuple(positions[: config.max_finding_samples]),
                    "column_profile",
                    0,
                    "caller_roles",
                )
            )
    time_columns = {
        value
        for name in (
            "observation_time",
            "event_time",
            "shared_feature_available_time",
            "label_available_time",
            "outcome_end_time",
            "partition_cutoff",
            "window_start",
            "window_end",
            "horizon_end",
            "analysis_as_of",
        )
        if (value := getattr(roles, name)) is not None
    } | {available for _, available in roles.feature_available_time_map}
    for column in sorted(time_columns, key=data.columns.get_loc):
        if not is_datetime64_any_dtype(data[column].dtype):
            findings.append(
                _finding(
                    "column_quality",
                    "column",
                    "current",
                    column,
                    data.columns.get_loc(column),
                    "datetime_role_mismatch",
                    "pandas_dtype",
                    None,
                    None,
                    None,
                    len(data),
                    (),
                    "column_profile",
                    0,
                    "caller_roles",
                )
            )
    return findings


def _overlap_findings(
    data: pd.DataFrame, roles: DataAuditRoles, config: DataAuditConfig
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for membership, category, reason_prefix in (
        (roles.partition, "partition_leakage", "partition"),
        (roles.fold, "partition_leakage", "fold"),
    ):
        if membership is None:
            continue
        membership_values = [data[membership].iat[i] for i in range(len(data))]
        for identity, reason in (
            (roles.row_identifier, f"identifier_{reason_prefix}_overlap"),
            (roles.group, f"group_{reason_prefix}_overlap"),
        ):
            if identity is None:
                continue
            memberships: dict[tuple[str, object], set[tuple[str, object]]] = {}
            samples: dict[tuple[str, object], list[int]] = {}
            for position in range(len(data)):
                left = data[identity].iat[position]
                right = membership_values[position]
                if _is_missing(left) or _is_missing(right):
                    continue
                key = _value_key(left)
                memberships.setdefault(key, set()).add(_value_key(right))
                samples.setdefault(key, []).append(position)
            affected = sorted(
                position
                for key, values in memberships.items()
                if len(values) > 1
                for position in samples[key]
            )
            if affected:
                findings.append(
                    _finding(
                        category,
                        reason_prefix,
                        "current",
                        identity,
                        data.columns.get_loc(identity),
                        reason,
                        "overlap_count",
                        float(len(affected)),
                        None,
                        len(affected),
                        len(data),
                        tuple(affected[: config.max_finding_samples]),
                        "slice_profile",
                        0,
                        "caller_roles|pandas_structure",
                    )
                )
    if roles.partition is not None and roles.fold is not None:
        mapping: dict[tuple[str, object], set[tuple[str, object]]] = {}
        affected: dict[tuple[str, object], list[int]] = {}
        for position in range(len(data)):
            fold_value = data[roles.fold].iat[position]
            partition_value = data[roles.partition].iat[position]
            if _is_missing(fold_value) or _is_missing(partition_value):
                continue
            key = _value_key(fold_value)
            mapping.setdefault(key, set()).add(_value_key(partition_value))
            affected.setdefault(key, []).append(position)
        positions = sorted(
            pos
            for key, values in mapping.items()
            if len(values) > 1
            for pos in affected[key]
        )
        if positions:
            findings.append(
                _finding(
                    "partition_leakage",
                    "fold",
                    "current",
                    roles.fold,
                    data.columns.get_loc(roles.fold),
                    "fold_partition_inconsistent",
                    "mapping_count",
                    float(len(positions)),
                    None,
                    len(positions),
                    len(data),
                    tuple(positions[: config.max_finding_samples]),
                    "slice_profile",
                    0,
                    "caller_roles|pandas_structure",
                )
            )
    if roles.partition is not None and len(data):
        payload_columns = [
            column for column in data.columns if column != roles.partition
        ]
        if payload_columns:
            payload_groups: dict[tuple[tuple[str, object], ...], list[int]] = {}
            for position in range(len(data)):
                key = tuple(
                    _value_key(data[column].iat[position]) for column in payload_columns
                )
                payload_groups.setdefault(key, []).append(position)
            overlap_positions: list[int] = []
            for positions in payload_groups.values():
                partitions = {
                    _value_key(data[roles.partition].iat[position])
                    for position in positions
                    if not _is_missing(data[roles.partition].iat[position])
                }
                if len(partitions) > 1:
                    overlap_positions.extend(positions)
            overlap_positions.sort()
            if overlap_positions:
                findings.append(
                    _finding(
                        "partition_leakage",
                        "partition",
                        "current",
                        None,
                        -1,
                        "duplicate_row_partition_overlap",
                        "overlap_count",
                        float(len(overlap_positions)),
                        None,
                        len(overlap_positions),
                        len(data),
                        tuple(overlap_positions[: config.max_finding_samples]),
                        "dataset_profile",
                        0,
                        "caller_roles|pandas_structure",
                    )
                )
    return findings


def _pit(
    data: pd.DataFrame,
    reference: pd.DataFrame | None,
    roles: DataAuditRoles,
    config: DataAuditConfig,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    def compare(
        frame: pd.DataFrame, left: str, operator: str, right: str
    ) -> tuple[list[int], list[int], str]:
        result = _evaluate_atomic_condition(
            frame,
            operator=operator,
            left=_ConditionOperand("column", left),
            right=_ConditionOperand("column", right),
            root_version=_KERNEL_VERSION,
        )
        violations = [i for i, truth in enumerate(result.truth) if truth == "false"]
        unknown = [i for i, truth in enumerate(result.truth) if truth == "unknown"]
        reason = (
            "timezone_mismatch"
            if any(result.reason.iat[i] == "timezone_mismatch" for i in unknown)
            else "missing_availability_metadata"
        )
        return violations, unknown, reason

    activation = (
        any(
            getattr(roles, name) is not None
            for name in (
                "observation_time",
                "event_time",
                "outcome_end_time",
                "label_available_time",
                "partition_cutoff",
                "window_start",
                "window_end",
                "horizon_end",
                "analysis_as_of",
            )
        )
        or roles.shared_feature_available_time is not None
        or bool(roles.feature_available_time_map)
    )
    if not activation:
        return _frame(
            "point_in_time_profile",
            [
                {
                    "side": "current",
                    "scope": "dataset",
                    "column": pd.NA,
                    "evaluated_count": pd.NA,
                    "violation_count": pd.NA,
                    "not_verifiable_count": pd.NA,
                    "status": "not_applicable",
                    "reason": "role_not_declared",
                    "finding_key": pd.NA,
                }
            ],
        ), []
    rows = []
    findings = []
    for side, frame in (("current", data), ("reference", reference)):
        if frame is None:
            continue
        feature_rows = []
        for feature in roles.features or ():
            available = roles.shared_feature_available_time or dict(
                roles.feature_available_time_map
            ).get(feature)
            if (
                available is None
                or roles.observation_time is None
                or available not in frame.columns
                or roles.observation_time not in frame.columns
            ):
                row = {
                    "side": side,
                    "scope": "feature",
                    "column": feature,
                    "evaluated_count": 0,
                    "violation_count": 0,
                    "not_verifiable_count": len(frame),
                    "status": "not_verifiable",
                    "reason": "partial_feature_availability_mapping"
                    if available is None
                    else "missing_availability_metadata",
                    "finding_key": pd.NA,
                }
                rows.append(row)
                feature_rows.append(row)
                continue
            false, unknown, unknown_reason = compare(
                frame, available, "le", roles.observation_time
            )
            cutoff_false: list[int] = []
            cutoff_unknown: list[int] = []
            cutoff_reason = "missing_availability_metadata"
            if roles.partition_cutoff is not None:
                if roles.partition_cutoff in frame.columns:
                    cutoff_false, cutoff_unknown, cutoff_reason = compare(
                        frame, available, "le", roles.partition_cutoff
                    )
                else:
                    cutoff_unknown = list(range(len(frame)))
            all_false = sorted(set(false) | set(cutoff_false))
            all_unknown = sorted(set(unknown) | set(cutoff_unknown))
            row = {
                "side": side,
                "scope": "feature",
                "column": feature,
                "evaluated_count": len(frame) - len(all_unknown),
                "violation_count": len(all_false),
                "not_verifiable_count": len(all_unknown),
                "status": "not_verifiable" if all_unknown else "available",
                "reason": unknown_reason
                if unknown
                else cutoff_reason
                if cutoff_unknown
                else "computed"
                if all_false
                else "safe",
                "finding_key": pd.NA,
            }
            rows.append(row)
            feature_rows.append(row)
            if false:
                findings.append(
                    _finding(
                        "point_in_time",
                        "time",
                        side,
                        feature,
                        frame.columns.get_loc(feature),
                        "feature_after_observation",
                        "violation_count",
                        float(len(false)),
                        None,
                        len(false),
                        len(frame),
                        tuple(false[: config.max_finding_samples]),
                        "point_in_time_profile",
                        len(rows) - 1,
                        "caller_roles|condition_kernel_atomic",
                    )
                )
            if cutoff_false:
                findings.append(
                    _finding(
                        "point_in_time",
                        "time",
                        side,
                        feature,
                        frame.columns.get_loc(feature),
                        "feature_after_cutoff",
                        "violation_count",
                        float(len(cutoff_false)),
                        None,
                        len(cutoff_false),
                        len(frame),
                        tuple(cutoff_false[: config.max_finding_samples]),
                        "point_in_time_profile",
                        len(rows) - 1,
                        "caller_roles|condition_kernel_atomic",
                    )
                )
        audit_inputs = [
            *roles.score_columns,
            *(
                (roles.historical_action,)
                if roles.historical_action is not None
                else ()
            ),
            *(
                (roles.historical_policy,)
                if roles.historical_policy is not None
                else ()
            ),
            *roles.cost_columns,
            *roles.exposure_columns,
            *roles.constraint_input_columns,
        ]
        metadata_columns = [
            column
            for column in (
                roles.partition,
                roles.fold,
                roles.observation_time,
                roles.window_start,
                roles.window_end,
                roles.horizon_end,
                roles.analysis_as_of,
            )
            if column is not None
        ]
        for column in audit_inputs:
            if column not in frame.columns:
                rows.append(
                    {
                        "side": side,
                        "scope": "audit_input",
                        "column": column,
                        "evaluated_count": 0,
                        "violation_count": 0,
                        "not_verifiable_count": len(frame),
                        "status": "unavailable",
                        "reason": "role_absent_in_reference",
                        "finding_key": pd.NA,
                    }
                )
                continue
            missing_positions = [
                position
                for position in range(len(frame))
                if _is_missing(frame[column].iat[position])
                or any(
                    metadata not in frame.columns
                    or _is_missing(frame[metadata].iat[position])
                    for metadata in metadata_columns
                )
            ]
            rows.append(
                {
                    "side": side,
                    "scope": "audit_input",
                    "column": column,
                    "evaluated_count": len(frame) - len(missing_positions),
                    "violation_count": 0,
                    "not_verifiable_count": len(missing_positions),
                    "status": "not_verifiable" if missing_positions else "available",
                    "reason": "missing_availability_metadata"
                    if missing_positions
                    else "safe",
                    "finding_key": pd.NA,
                }
            )
            if missing_positions:
                findings.append(
                    _finding(
                        "point_in_time",
                        "time",
                        side,
                        column,
                        frame.columns.get_loc(column),
                        "missing_availability_metadata",
                        "not_verifiable_count",
                        float(len(missing_positions)),
                        None,
                        len(missing_positions),
                        len(frame),
                        tuple(missing_positions[: config.max_finding_samples]),
                        "point_in_time_profile",
                        len(rows) - 1,
                        "caller_roles",
                    )
                )
        relations = (
            (
                roles.observation_time,
                roles.outcome_end_time,
                "le",
                "observation_after_outcome_end",
            ),
            (
                roles.observation_time,
                roles.event_time,
                "le",
                "event_before_observation",
            ),
            (roles.event_time, roles.outcome_end_time, "le", "event_after_outcome_end"),
            (
                roles.outcome_end_time,
                roles.label_available_time,
                "le",
                "label_before_outcome_end",
            ),
            (
                roles.observation_time,
                roles.partition_cutoff,
                "lt",
                "observation_at_or_after_cutoff",
            ),
            (
                roles.window_start,
                roles.window_end,
                "le",
                "window_start_after_window_end",
            ),
            (
                roles.window_end,
                roles.analysis_as_of,
                "le",
                "window_end_after_analysis_as_of",
            ),
            (
                roles.horizon_end,
                roles.analysis_as_of,
                "le",
                "horizon_end_after_analysis_as_of",
            ),
        )
        for left, right, operator, reason in relations:
            if left is None or right is None:
                continue
            if left not in frame.columns or right not in frame.columns:
                rows.append(
                    {
                        "side": side,
                        "scope": "chronology",
                        "column": pd.NA,
                        "evaluated_count": 0,
                        "violation_count": 0,
                        "not_verifiable_count": len(frame),
                        "status": "not_verifiable",
                        "reason": "missing_availability_metadata",
                        "finding_key": pd.NA,
                    }
                )
                continue
            false, unknown = _atomic_violations(
                frame, left, operator, _ConditionOperand("column", right)
            )
            rows.append(
                {
                    "side": side,
                    "scope": "chronology",
                    "column": pd.NA,
                    "evaluated_count": len(frame) - len(unknown),
                    "violation_count": len(false),
                    "not_verifiable_count": len(unknown),
                    "status": "not_verifiable" if unknown else "available",
                    "reason": "timezone_mismatch"
                    if unknown
                    else "computed"
                    if false
                    else "safe",
                    "finding_key": pd.NA,
                }
            )
            if false:
                findings.append(
                    _finding(
                        "time_leakage",
                        "time",
                        side,
                        None,
                        -1,
                        reason,
                        operator,
                        None,
                        None,
                        len(false),
                        len(frame),
                        tuple(false[: config.max_finding_samples]),
                        "point_in_time_profile",
                        len(rows) - 1,
                        "caller_roles|condition_kernel_atomic",
                    )
                )
        if (
            roles.observation_time is not None
            and roles.observation_time in frame.columns
        ):
            for identity_column in (roles.row_identifier, roles.group):
                if identity_column is None or identity_column not in frame.columns:
                    continue
                groups: dict[
                    tuple[tuple[str, object], tuple[str, object]], list[int]
                ] = {}
                for position in range(len(frame)):
                    identity = frame[identity_column].iat[position]
                    observed = frame[roles.observation_time].iat[position]
                    if _is_missing(identity) or _is_missing(observed):
                        continue
                    key = (_value_key(identity), _value_key(observed))
                    groups.setdefault(key, []).append(position)
                duplicates = sorted(
                    position
                    for positions in groups.values()
                    if len(positions) > 1
                    for position in positions
                )
                rows.append(
                    {
                        "side": side,
                        "scope": "chronology",
                        "column": pd.NA,
                        "evaluated_count": sum(len(value) for value in groups.values()),
                        "violation_count": len(duplicates),
                        "not_verifiable_count": len(frame)
                        - sum(len(value) for value in groups.values()),
                        "status": "available",
                        "reason": "computed" if duplicates else "safe",
                        "finding_key": pd.NA,
                    }
                )
                if duplicates:
                    findings.append(
                        _finding(
                            "time_leakage",
                            "time",
                            side,
                            identity_column,
                            frame.columns.get_loc(identity_column),
                            "duplicate_entity_time",
                            "duplicate_count",
                            float(len(duplicates)),
                            None,
                            len(duplicates),
                            len(frame),
                            tuple(duplicates[: config.max_finding_samples]),
                            "point_in_time_profile",
                            len(rows) - 1,
                            "caller_roles|pandas_structure",
                        )
                    )
            if len(frame) > 1:
                prior_positions = list(range(len(frame) - 1))
                current_positions = list(range(1, len(frame)))
                pair = pd.DataFrame(
                    {
                        "prior": [
                            frame[roles.observation_time].iat[position]
                            for position in prior_positions
                        ],
                        "current": [
                            frame[roles.observation_time].iat[position]
                            for position in current_positions
                        ],
                    }
                )
                ordering = _evaluate_atomic_condition(
                    pair,
                    operator="le",
                    left=_ConditionOperand("column", "prior"),
                    right=_ConditionOperand("column", "current"),
                    root_version=_KERNEL_VERSION,
                )
                reverse_positions = [
                    current_positions[index]
                    for index, truth in enumerate(ordering.truth)
                    if truth == "false"
                ]
                unknown_positions = [
                    current_positions[index]
                    for index, truth in enumerate(ordering.truth)
                    if truth == "unknown"
                ]
                rows.append(
                    {
                        "side": side,
                        "scope": "chronology",
                        "column": pd.NA,
                        "evaluated_count": len(pair) - len(unknown_positions),
                        "violation_count": len(reverse_positions),
                        "not_verifiable_count": len(unknown_positions),
                        "status": "not_verifiable"
                        if unknown_positions
                        else "available",
                        "reason": "timezone_mismatch"
                        if any(
                            ordering.reason.iat[index] == "timezone_mismatch"
                            for index, truth in enumerate(ordering.truth)
                            if truth == "unknown"
                        )
                        else "missing_availability_metadata"
                        if unknown_positions
                        else "computed"
                        if reverse_positions
                        else "safe",
                        "finding_key": pd.NA,
                    }
                )
                if reverse_positions:
                    findings.append(
                        _finding(
                            "time_leakage",
                            "time",
                            side,
                            roles.observation_time,
                            frame.columns.get_loc(roles.observation_time),
                            "time_order_violation",
                            "violation_count",
                            float(len(reverse_positions)),
                            None,
                            len(reverse_positions),
                            len(pair),
                            tuple(reverse_positions[: config.max_finding_samples]),
                            "point_in_time_profile",
                            len(rows) - 1,
                            "caller_roles|condition_kernel_atomic",
                        )
                    )
        chronology_rows = [
            row for row in rows if row["side"] == side and row["scope"] == "chronology"
        ]
        audit_input_rows = [
            row for row in rows if row["side"] == side and row["scope"] == "audit_input"
        ]
        all_rows = feature_rows + audit_input_rows + chronology_rows
        missing_feature_universe = roles.features is None
        any_unknown = missing_feature_universe or any(
            row["status"] in {"not_verifiable", "unavailable"} for row in all_rows
        )
        violations = sum(int(row["violation_count"] or 0) for row in all_rows)
        evaluated = sum(int(row["evaluated_count"] or 0) for row in all_rows)
        no_feature_evidence = roles.features == () and not all_rows
        dataset_detail_ordinal = len(rows)
        rows.append(
            {
                "side": side,
                "scope": "dataset",
                "column": pd.NA,
                "evaluated_count": evaluated,
                "violation_count": violations,
                "not_verifiable_count": sum(
                    int(row["not_verifiable_count"] or 0) for row in all_rows
                ),
                "status": "not_applicable"
                if no_feature_evidence
                else "not_verifiable"
                if any_unknown
                else "available",
                "reason": "missing_availability_metadata"
                if any_unknown
                else "no_features"
                if no_feature_evidence
                else "computed"
                if violations
                else "safe",
                "finding_key": pd.NA,
            }
        )
        if missing_feature_universe:
            findings.append(
                _finding(
                    "point_in_time",
                    "dataset",
                    side,
                    None,
                    -1,
                    "missing_availability_metadata",
                    "feature_universe",
                    None,
                    None,
                    len(frame),
                    len(frame),
                    (),
                    "point_in_time_profile",
                    dataset_detail_ordinal,
                    "caller_roles",
                )
            )
    return _frame("point_in_time_profile", rows), findings


def _sanitized(
    roles: DataAuditRoles, config: DataAuditConfig
) -> tuple[str, pd.DataFrame]:
    policy = {
        field.name: getattr(config, field.name)
        for field in fields(config)
        if field.name not in {"positive_label", "column_rules"}
    }
    role_payload = {field.name: getattr(roles, field.name) for field in fields(roles)}
    rule_payload = []
    for ordinal, rule in enumerate(config.column_rules):
        ops = []
        for key, value in (
            ("minimum_ge" if rule.minimum_inclusive else "minimum_gt", rule.minimum),
            ("maximum_le" if rule.maximum_inclusive else "maximum_lt", rule.maximum),
            ("allowed_in", rule.allowed_values),
            ("special_in", rule.special_values),
            ("not_after_le", rule.not_after_columns),
            ("nondecreasing_le", rule.nondecreasing),
        ):
            if value is not None and value != () and value is not False:
                literals = value if type(value) is tuple else (value,)
                families = {
                    _label_family(item)
                    if _label_family(item) != "not_declared"
                    else "none"
                    for item in literals
                }
                ops.append(
                    {
                        "operator_key": key,
                        "literal_declared": True,
                        "literal_count": len(literals),
                        "literal_type_family": next(iter(families))
                        if len(families) == 1
                        else "mixed",
                    }
                )
        rule_payload.append(
            {
                "rule_key": f"column_rule:{ordinal}",
                "rule_column_name": rule.column,
                "operators": ops,
            }
        )
    payload = {
        "config_schema_version": _CONFIG_VERSION,
        "policy": policy,
        "positive_label_declared": config.positive_label is not None,
        "positive_label_type": _label_family(config.positive_label),
        "roles": role_payload,
        "rules": rule_payload,
        "rule_count": len(config.column_rules),
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            default=list,
        ).encode()
    ).hexdigest()
    rows = [
        {
            "provenance_key": "config_schema_version",
            "value_type": "text",
            "numeric_value": pd.NA,
            "text_value": _CONFIG_VERSION,
            "count_value": pd.NA,
            "boolean_value": pd.NA,
            "status": "available",
            "reason": "computed",
        },
        {
            "provenance_key": "atomic_kernel_version",
            "value_type": "text",
            "numeric_value": pd.NA,
            "text_value": _KERNEL_VERSION,
            "count_value": pd.NA,
            "boolean_value": pd.NA,
            "status": "available",
            "reason": "computed",
        },
    ]
    for field in fields(config):
        if field.name in {"positive_label", "column_rules"}:
            continue
        value = getattr(config, field.name)
        rows.append(
            {
                "provenance_key": field.name,
                "value_type": "count"
                if type(value) is int
                else "numeric"
                if value is not None
                else "none",
                "numeric_value": float(value)
                if value is not None and type(value) is not int
                else pd.NA,
                "text_value": pd.NA,
                "count_value": value if type(value) is int else pd.NA,
                "boolean_value": pd.NA,
                "status": "available",
                "reason": "computed",
            }
        )
    rows.extend(
        [
            {
                "provenance_key": "positive_label_declared",
                "value_type": "boolean",
                "numeric_value": pd.NA,
                "text_value": pd.NA,
                "count_value": pd.NA,
                "boolean_value": config.positive_label is not None,
                "status": "available",
                "reason": "computed",
            },
            {
                "provenance_key": "positive_label_type",
                "value_type": "text",
                "numeric_value": pd.NA,
                "text_value": _label_family(config.positive_label),
                "count_value": pd.NA,
                "boolean_value": pd.NA,
                "status": "available",
                "reason": "computed",
            },
        ]
    )
    for role_field in fields(roles):
        value = getattr(roles, role_field.name)
        values: tuple[str, ...]
        if type(value) is str:
            values = (value,)
        elif role_field.name == "feature_available_time_map":
            values = tuple(item for pair in value for item in pair)
        elif type(value) is tuple:
            values = value
        else:
            values = ()
        for ordinal, column in enumerate(values):
            rows.append(
                {
                    "provenance_key": f"role:{role_field.name}:{ordinal}",
                    "value_type": "text",
                    "numeric_value": pd.NA,
                    "text_value": column,
                    "count_value": pd.NA,
                    "boolean_value": pd.NA,
                    "status": "available",
                    "reason": "computed",
                }
            )
    for rule in rule_payload:
        rows.append(
            {
                "provenance_key": rule["rule_key"],
                "value_type": "text",
                "numeric_value": pd.NA,
                "text_value": rule["rule_column_name"],
                "count_value": pd.NA,
                "boolean_value": pd.NA,
                "status": "available",
                "reason": "computed",
            }
        )
        for operator in rule["operators"]:
            rows.extend(
                {
                    "provenance_key": f"{rule['rule_key']}:{key}",
                    "value_type": value_type,
                    "numeric_value": pd.NA,
                    "text_value": value if value_type == "text" else pd.NA,
                    "count_value": value if value_type == "count" else pd.NA,
                    "boolean_value": value if value_type == "boolean" else pd.NA,
                    "status": "available",
                    "reason": "computed",
                }
                for key, value, value_type in (
                    ("operator_key", operator["operator_key"], "text"),
                    ("literal_declared", operator["literal_declared"], "boolean"),
                    ("literal_count", operator["literal_count"], "count"),
                    ("literal_type_family", operator["literal_type_family"], "text"),
                )
            )
    rows.extend(
        [
            {
                "provenance_key": "rule_count",
                "value_type": "count",
                "numeric_value": pd.NA,
                "text_value": pd.NA,
                "count_value": len(config.column_rules),
                "boolean_value": pd.NA,
                "status": "available",
                "reason": "computed",
            },
            {
                "provenance_key": "config_fingerprint",
                "value_type": "text",
                "numeric_value": pd.NA,
                "text_value": fingerprint,
                "count_value": pd.NA,
                "boolean_value": pd.NA,
                "status": "available",
                "reason": "computed",
            },
        ]
    )
    return fingerprint, _frame("provenance", rows)


def _leakage_findings(
    data: pd.DataFrame, roles: DataAuditRoles, config: DataAuditConfig
) -> list[dict[str, object]]:
    findings = []
    if roles.features is None:
        return findings
    target = roles.target
    role_columns = set(
        x
        for name in _SCALAR_ROLES
        if name != "target"
        for x in (getattr(roles, name),)
        if x is not None
    ) | set(
        roles.score_columns
        + roles.cost_columns
        + roles.exposure_columns
        + roles.constraint_input_columns
    )
    for feature in roles.features:
        position = data.columns.get_loc(feature)
        if feature == target:
            findings.append(
                _finding(
                    "direct_target_leakage",
                    "column",
                    "current",
                    feature,
                    position,
                    "target_in_features",
                    "role",
                    None,
                    None,
                    None,
                    None,
                    (),
                    "column_profile",
                    position,
                    "caller_roles",
                )
            )
            continue
        if feature in roles.post_outcome_columns:
            findings.append(
                _finding(
                    "direct_target_leakage",
                    "column",
                    "current",
                    feature,
                    position,
                    "post_outcome_feature",
                    "role",
                    None,
                    None,
                    None,
                    None,
                    (),
                    "column_profile",
                    position,
                    "caller_roles",
                )
            )
        if feature in role_columns:
            findings.append(
                _finding(
                    "direct_target_leakage",
                    "column",
                    "current",
                    feature,
                    position,
                    "role_column_in_features",
                    "role",
                    None,
                    None,
                    None,
                    None,
                    (),
                    "column_profile",
                    position,
                    "caller_roles",
                )
            )
        if target is None or target not in data.columns:
            continue
        result = _evaluate_atomic_condition(
            data,
            operator="eq",
            left=_ConditionOperand("column", feature),
            right=_ConditionOperand("column", target),
            root_version=_KERNEL_VERSION,
        )
        evaluable = [i for i, v in enumerate(result.truth) if v != "unknown"]
        matches = [i for i in evaluable if result.truth.iat[i] == "true"]
        if len(evaluable) >= 2 and len(matches) == len(evaluable):
            findings.append(
                _finding(
                    "direct_target_leakage",
                    "row_set",
                    "current",
                    feature,
                    position,
                    "exact_target_copy",
                    "exact_match_rate",
                    1.0,
                    1.0,
                    len(matches),
                    len(evaluable),
                    tuple(matches[: config.max_finding_samples]),
                    "column_profile",
                    position,
                    "condition_kernel_atomic",
                )
            )
        elif (
            len(evaluable) >= config.proxy_min_support
            and len(matches) / len(evaluable) >= config.near_copy_rate
        ):
            findings.append(
                _finding(
                    "target_proxy",
                    "row_set",
                    "current",
                    feature,
                    position,
                    "near_target_copy",
                    "exact_match_rate",
                    len(matches) / len(evaluable),
                    config.near_copy_rate,
                    len(matches),
                    len(evaluable),
                    tuple(matches[: config.max_finding_samples]),
                    "column_profile",
                    position,
                    "condition_kernel_atomic",
                )
            )
        elif not is_numeric_dtype(data[feature].dtype):
            pairs = [
                (_value_key(data[feature].iat[i]), _value_key(data[target].iat[i]))
                for i in range(len(data))
                if not _is_missing(data[feature].iat[i])
                and not _is_missing(data[target].iat[i])
            ]
            feature_counts: dict[tuple[str, object], int] = {}
            target_by_feature: dict[tuple[str, object], set[tuple[str, object]]] = {}
            for feature_key, target_key in pairs:
                feature_counts[feature_key] = feature_counts.get(feature_key, 0) + 1
                target_by_feature.setdefault(feature_key, set()).add(target_key)
            unique_rate = len(feature_counts) / len(pairs) if pairs else 0.0
            suspected_identifier = (
                len(pairs) >= config.identifier_min_non_missing
                and unique_rate >= config.identifier_rate
            )
            deterministic = (
                len(pairs) >= config.proxy_min_support
                and 1 < len(feature_counts) <= config.max_category_levels
                and len({target_key for _, target_key in pairs}) >= 2
                and all(len(values) == 1 for values in target_by_feature.values())
                and all(count >= 2 for count in feature_counts.values())
            )
            if deterministic and not suspected_identifier:
                findings.append(
                    _finding(
                        "target_proxy",
                        "row_set",
                        "current",
                        feature,
                        position,
                        "deterministic_categorical_proxy",
                        "level_mapping",
                        float(len(feature_counts)),
                        None,
                        len(pairs),
                        len(pairs),
                        (),
                        "categorical_profile",
                        0,
                        "pandas_structure|task15_label_semantics",
                    )
                )
    return findings


def _resources(
    data: pd.DataFrame,
    reference: pd.DataFrame | None,
    roles: DataAuditRoles,
    config: DataAuditConfig,
    pattern_truncated: bool,
    collinear_truncated: bool,
) -> pd.DataFrame:
    rows = []
    for side, frame in (("current", data), ("reference", reference)):
        if frame is None:
            continue
        values = (
            ("columns", frame.shape[1], frame.shape[1], frame.shape[1], False),
            (
                "column_rules",
                len(config.column_rules),
                config.max_column_rules,
                len(config.column_rules),
                False,
            ),
            (
                "duplicate_scan_rows",
                len(frame),
                config.duplicate_scan_row_limit,
                min(len(frame), config.duplicate_scan_row_limit),
                len(frame) > config.duplicate_scan_row_limit,
            ),
            (
                "unique_inspection_rows",
                len(frame),
                config.max_unique_inspection_rows,
                min(len(frame), config.max_unique_inspection_rows),
                len(frame) > config.max_unique_inspection_rows,
            ),
            (
                "category_levels",
                config.max_category_levels,
                config.max_category_levels,
                config.max_category_levels,
                False,
            ),
            (
                "missing_patterns",
                config.max_missing_patterns,
                config.max_missing_patterns,
                config.max_missing_patterns
                if pattern_truncated
                else min(config.max_missing_patterns, max(1, len(frame))),
                pattern_truncated,
            ),
            (
                "collinearity_columns",
                len(roles.features or ()),
                config.max_collinearity_columns,
                min(len(roles.features or ()), config.max_collinearity_columns),
                collinear_truncated,
            ),
            (
                "finding_samples",
                config.max_finding_samples,
                config.max_finding_samples,
                config.max_finding_samples,
                False,
            ),
        )
        for resource, requested, available, actual, truncated in values:
            rows.append(
                {
                    "side": side,
                    "resource": resource,
                    "requested": requested,
                    "available": available,
                    "actual": actual,
                    "truncated": truncated,
                    "status": "not_verifiable" if truncated else "available",
                    "reason": "duplicate_scan_budget"
                    if resource == "duplicate_scan_rows" and truncated
                    else "budget_exceeded"
                    if truncated
                    else "computed",
                    "finding_key": pd.NA,
                }
            )
    return _frame("resource_usage", rows)


def _attach_findings(tables: dict[str, pd.DataFrame], findings: pd.DataFrame) -> None:
    if findings.empty:
        return
    severity = {"error": 0, "warning": 1, "info": 2}
    for table_name, table in tables.items():
        if "finding_key" not in table.columns:
            continue
        subset = findings.loc[findings["detail_table"] == table_name]
        for ordinal in range(len(table)):
            choices = subset.loc[subset["detail_row_ordinal"] == ordinal]
            if not choices.empty:
                chosen = sorted(
                    choices.to_dict("records"),
                    key=lambda row: (severity[row["severity"]], row["finding_key"]),
                )[0]
                table.at[ordinal, "finding_key"] = chosen["finding_key"]


def audit_data_quality(
    data: pd.DataFrame,
    *,
    reference: pd.DataFrame | None = None,
    roles: DataAuditRoles | None = None,
    config: DataAuditConfig | None = None,
) -> DataAuditResult:
    """Audit quality, missingness, drift, and leakage evidence without mutation.

    Parameters
    ----------
    data, reference
        Current data and an optional independent reference DataFrame.
    roles, config
        Explicit role declarations and bounded audit policy.

    Returns
    -------
    DataAuditResult
        Fourteen newly allocated, typed evidence tables.

    Raises
    ------
    ValueError
        If input schema, selectors, literals, or resource budgets are invalid.

    Notes
    -----
    Missing evidence remains structured as unavailable, undefined, not applicable,
    or not verifiable. Inputs are never modified and raw values are never returned.

    Examples
    --------
    >>> result = audit_data_quality(pd.DataFrame({"x": [1, None]}))
    >>> result.n_rows
    2
    """
    if type(data) is not pd.DataFrame:
        raise _audit_input_error("data_not_dataframe")
    if reference is not None and type(reference) is not pd.DataFrame:
        raise _audit_input_error("reference_not_dataframe")
    for frame in (data, reference):
        if frame is None:
            continue
        if frame.columns.has_duplicates:
            raise _audit_input_error("duplicate_columns")
        if any(type(column) is not str for column in frame.columns):
            raise _audit_input_error("non_string_columns")
        _safe_scalar_scan(frame)
    roles = DataAuditRoles() if roles is None else roles
    config = DataAuditConfig() if config is None else config
    if type(roles) is not DataAuditRoles:
        raise _audit_config_error("invalid_selector")
    if type(config) is not DataAuditConfig:
        raise _audit_config_error("invalid_selector")
    _validate_roles(data, roles)
    _validate_config(config, roles)
    union_width = len(
        set(data.columns) | (set(reference.columns) if reference is not None else set())
    )
    if (
        data.shape[1] > config.max_columns
        or (reference is not None and reference.shape[1] > config.max_columns)
        or union_width > config.max_columns
    ):
        raise _audit_config_error("max_columns_exceeded")
    current_schema = infer_schema(data)
    reference_schema = infer_schema(reference) if reference is not None else None
    current_rows, current_findings, current_warnings = _profile_side(
        data, "current", roles, config, current_schema
    )
    reference_rows = {name: [] for name in current_rows}
    reference_findings = []
    reference_warnings = []
    if reference is not None:
        reference_rows, reference_findings, reference_warnings = _profile_side(
            reference, "reference", roles, config, reference_schema
        )
    combined = {
        name: _frame(name, current_rows[name] + reference_rows[name])
        for name in current_rows
    }
    pattern_columns = (
        list(roles.features)
        if roles.features is not None
        else _profile_columns(data, roles)
    )
    missingness_patterns, pattern_truncated = _missing_patterns(
        data, reference, pattern_columns, config
    )
    missingness_drift, schema_drift, drift_findings = _drift_tables(
        data, reference, roles, current_schema, reference_schema, config
    )
    collinearity, collinear_truncated = _collinearity(data, roles, config)
    pit, pit_findings = _pit(data, reference, roles, config)
    findings_list = (
        current_findings
        + reference_findings
        + drift_findings
        + _rule_findings(data, roles, config)
        + _leakage_findings(data, roles, config)
        + _roadmap_findings(data, roles, config)
        + _overlap_findings(data, roles, config)
        + pit_findings
    )
    resources = _resources(
        data, reference, roles, config, pattern_truncated, collinear_truncated
    )
    fingerprint, provenance = _sanitized(roles, config)
    tables = {
        **combined,
        "missingness_patterns": missingness_patterns,
        "missingness_drift": missingness_drift,
        "schema_drift": schema_drift,
        "collinearity": collinearity,
        "point_in_time_profile": pit,
        "resource_usage": resources,
    }
    _relink_findings(findings_list, tables)
    category_order = (
        "dataset_structure",
        "column_quality",
        "missingness",
        "schema_drift",
        "missingness_drift",
        "target_quality",
        "direct_target_leakage",
        "target_proxy",
        "partition_leakage",
        "group_leakage",
        "time_leakage",
        "point_in_time",
        "constraint_violation",
        "resource_limitation",
    )
    severity_order = {"error": 0, "warning": 1, "info": 2}
    dataset_order = {"current": 0, "reference": 1, "comparison": 2}
    findings_list.sort(
        key=lambda row: (
            severity_order[row["severity"]],
            category_order.index(row["category"]),
            dataset_order[row["dataset_role"]],
            -1 if pd.isna(row["column_position"]) else int(row["column_position"]),
            min(row["sample_positions"]) if row["sample_positions"] else -1,
            int(row["detail_row_ordinal"]),
            row["finding_key"],
        )
    )
    findings = _frame("findings", findings_list)
    _attach_findings(tables, findings)
    warning_set = []
    for warning in (
        "large_input" if len(data) > 1_000_000 else None,
        "duplicate_scan_skipped"
        if len(data) > config.duplicate_scan_row_limit
        else None,
        "unique_inspection_skipped"
        if len(data) > config.max_unique_inspection_rows
        else None,
        "missing_patterns_truncated" if pattern_truncated else None,
        "collinearity_columns_truncated" if collinear_truncated else None,
        "insufficient_drift_rows"
        if reference is not None
        and len(data) < config.minimum_drift_rows
        and len(reference) < config.minimum_drift_rows
        else None,
        "point_in_time_not_verifiable"
        if any(pit["status"] == "not_verifiable")
        else None,
    ):
        if warning is not None:
            warning_set.append(warning)
    limitations = [
        "in_memory_single_process",
        "structural_identifier_evidence_only",
        "association_not_causation",
        "target_proxy_false_positive_possible",
        "caller_declared_time_provenance",
        "no_automatic_leakage_repair",
    ]
    if warning_set:
        limitations.append("budget_limited_evidence")
    return DataAuditResult(
        fingerprint,
        len(data),
        data.shape[1],
        None if reference is None else len(reference),
        None if reference is None else reference.shape[1],
        tables["dataset_profile"],
        tables["column_profile"],
        tables["numeric_profile"],
        tables["categorical_profile"],
        tables["target_profile"],
        tables["slice_profile"],
        missingness_patterns,
        missingness_drift,
        schema_drift,
        collinearity,
        pit,
        resources,
        provenance,
        findings,
        tuple(warning_set),
        tuple(limitations),
    )
