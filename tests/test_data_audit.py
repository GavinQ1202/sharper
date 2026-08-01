"""Task 16 public data-audit contract tests."""

import hashlib
import json
from dataclasses import FrozenInstanceError, fields
from datetime import timezone

import numpy as np
import pandas as pd
import pytest

from sharper import (
    ColumnAuditRule,
    DataAuditConfig,
    DataAuditResult,
    DataAuditRoles,
    audit_data_quality,
)
from sharper._condition_kernel import (
    _ConditionOperand,
    _evaluate_atomic_condition,
)
from sharper.data_audit import _relink_findings


def test_public_dataclasses_are_frozen_and_ordered() -> None:
    assert [field.name for field in fields(DataAuditRoles)][:4] == [
        "target",
        "features",
        "score_columns",
        "excluded_columns",
    ]
    assert [field.name for field in fields(ColumnAuditRule)] == [
        "column",
        "minimum",
        "maximum",
        "minimum_inclusive",
        "maximum_inclusive",
        "allowed_values",
        "special_values",
        "not_after_columns",
        "nondecreasing",
    ]
    assert [field.name for field in fields(DataAuditResult)][5:19] == [
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
    ]
    with pytest.raises(FrozenInstanceError):
        DataAuditRoles().target = "y"  # type: ignore[misc]


def test_profiles_are_typed_deterministic_and_input_is_unchanged() -> None:
    frame = pd.DataFrame(
        {"x": [1.0, None, 3.0], "cat": ["a", "", "a"]}, index=[2, 2, 1]
    )
    before = frame.copy(deep=True)
    first = audit_data_quality(frame)
    second = audit_data_quality(frame)
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(first.column_profile, second.column_profile)
    assert first.config_fingerprint == second.config_fingerprint
    assert first.column_profile["missing_rate"].dtype == "Float64"
    assert first.findings["finding_key"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_zero_rows_zero_features_and_missing_patterns() -> None:
    empty = audit_data_quality(pd.DataFrame({"x": pd.Series(dtype="float64")}))
    assert empty.missingness_patterns.loc[0, "pattern_key"] == "__NO_ROWS__"
    zero = audit_data_quality(
        pd.DataFrame({"x": [1, 2]}), roles=DataAuditRoles(features=())
    )
    assert zero.missingness_patterns.loc[0, "pattern_key"] == "p:"
    assert "zero_feature_dataset" in zero.findings["reason"].tolist()


def test_reference_drift_and_schema_precedence() -> None:
    reference = pd.DataFrame({"x": [1.0, None], "old": [1, 2]})
    current = pd.DataFrame(
        {"x": pd.Series([None, None], dtype="float64"), "new": [1, 2]}
    )
    result = audit_data_quality(current, reference=reference)
    assert result.schema_drift["primary_change"].tolist() == [
        "logical_type_changed",
        "removed",
        "added",
    ]
    x = result.missingness_drift.loc[result.missingness_drift["column"] == "x"].iloc[0]
    assert x["absolute_rate_change"] == pytest.approx(0.5)


def test_rule_audit_uses_closed_semantics() -> None:
    frame = pd.DataFrame({"x": [0, 1, 3], "end": [0, 2, 2]})
    config = DataAuditConfig(
        column_rules=(
            ColumnAuditRule("x", minimum=1, maximum=2, not_after_columns=("end",)),
        )
    )
    result = audit_data_quality(frame, config=config)
    assert {"range_violation", "cross_column_order_violation"} <= set(
        result.findings["reason"]
    )


def test_monotonic_roadmap_and_overlap_evidence() -> None:
    frame = pd.DataFrame(
        {
            "entity": [1, 1, 2],
            "partition": ["train", "test", "train"],
            "fold": [0, 1, 0],
            "score": ["bad", "bad", "bad"],
            "exposure": [1.0, -2.0, 3.0],
            "value": [2, 1, 3],
        }
    )
    result = audit_data_quality(
        frame,
        roles=DataAuditRoles(
            row_identifier="entity",
            partition="partition",
            fold="fold",
            score_columns=("score",),
            exposure_columns=("exposure",),
        ),
        config=DataAuditConfig(
            column_rules=(ColumnAuditRule("value", nondecreasing=True),)
        ),
    )
    assert {
        "identifier_partition_overlap",
        "audit_input_dtype_mismatch",
        "negative_exposure",
        "monotonic_time_violation",
    } <= set(result.findings["reason"])


def test_direct_leakage_and_point_in_time_evidence() -> None:
    frame = pd.DataFrame(
        {
            "x": [0, 1],
            "y": [0, 1],
            "available": pd.to_datetime(["2025-01-01", "2025-01-03"]),
            "observed": pd.to_datetime(["2025-01-02", "2025-01-02"]),
        }
    )
    roles = DataAuditRoles(
        target="y",
        features=("x",),
        observation_time="observed",
        shared_feature_available_time="available",
    )
    result = audit_data_quality(frame, roles=roles)
    assert "exact_target_copy" in result.findings["reason"].tolist()
    assert "feature_after_observation" in result.findings["reason"].tolist()


@pytest.mark.parametrize(
    ("data", "key"),
    [
        (pd.DataFrame([[1, 2]], columns=["x", "x"]), "duplicate_columns"),
        (pd.DataFrame({1: [1]}), "non_string_columns"),
    ],
)
def test_stable_input_errors(data: pd.DataFrame, key: str) -> None:
    with pytest.raises(ValueError, match=f"data audit input is invalid: {key}"):
        audit_data_quality(data)


def test_pit_feature_universe_none_empty_complete_and_partial() -> None:
    observed = pd.DataFrame(
        {
            "observed": pd.to_datetime(["2025-01-01"]),
            "available": pd.to_datetime(["2025-01-01"]),
            "x": [1],
            "y": [2],
        }
    )
    undeclared = audit_data_quality(
        observed, roles=DataAuditRoles(observation_time="observed")
    )
    dataset = undeclared.point_in_time_profile.iloc[-1]
    assert (dataset["status"], dataset["reason"]) == (
        "not_verifiable",
        "missing_availability_metadata",
    )
    assert "missing_availability_metadata" in undeclared.findings["reason"].tolist()

    with pytest.raises(ValueError, match="invalid_feature_availability_mapping"):
        audit_data_quality(
            observed,
            roles=DataAuditRoles(
                observation_time="observed",
                shared_feature_available_time="available",
            ),
        )
    with pytest.raises(ValueError, match="invalid_feature_availability_mapping"):
        audit_data_quality(
            observed,
            roles=DataAuditRoles(
                observation_time="observed",
                feature_available_time_map=(("x", "available"),),
            ),
        )
    explicit_empty = audit_data_quality(
        observed,
        roles=DataAuditRoles(features=(), observation_time="observed"),
    )
    empty_dataset = explicit_empty.point_in_time_profile.iloc[-1]
    assert (empty_dataset["status"], empty_dataset["reason"]) == (
        "not_applicable",
        "no_features",
    )
    complete = audit_data_quality(
        observed,
        roles=DataAuditRoles(
            features=("x", "y"),
            observation_time="observed",
            shared_feature_available_time="available",
        ),
    )
    assert complete.point_in_time_profile.iloc[-1]["reason"] == "safe"
    partial = audit_data_quality(
        observed,
        roles=DataAuditRoles(
            features=("x", "y"),
            observation_time="observed",
            feature_available_time_map=(("x", "available"),),
        ),
    )
    y_row = partial.point_in_time_profile.loc[
        partial.point_in_time_profile["column"] == "y"
    ].iloc[0]
    assert (y_row["status"], y_row["reason"]) == (
        "not_verifiable",
        "partial_feature_availability_mapping",
    )


def test_feature_cutoff_audit_inputs_duplicate_entity_and_time_order() -> None:
    frame = pd.DataFrame(
        {
            "entity": [1, 1, 2],
            "x": [1, 2, 3],
            "score": [0.1, 0.2, 0.3],
            "action": ["a", "a", "b"],
            "policy": ["p", "p", "p"],
            "cost": [1.0, 2.0, 3.0],
            "exposure": [4.0, 5.0, 6.0],
            "constraint": [1, 1, 1],
            "available": pd.to_datetime(["2025-01-02", "2025-01-02", "2025-01-04"]),
            "observed": pd.to_datetime(["2025-01-03", "2025-01-03", "2025-01-02"]),
            "cutoff": pd.to_datetime(["2025-01-02", "2025-01-01", "2025-01-04"]),
            "window_start": pd.to_datetime(["2025-01-01"] * 3),
            "window_end": pd.to_datetime(["2025-01-03"] * 3),
            "horizon": pd.to_datetime(["2025-01-03"] * 3),
            "as_of": pd.to_datetime(["2025-01-04"] * 3),
        },
        index=[7, 7, 3],
    )
    before = frame.copy(deep=True)
    result = audit_data_quality(
        frame,
        roles=DataAuditRoles(
            features=("x",),
            row_identifier="entity",
            score_columns=("score",),
            historical_action="action",
            historical_policy="policy",
            cost_columns=("cost",),
            exposure_columns=("exposure",),
            constraint_input_columns=("constraint",),
            observation_time="observed",
            shared_feature_available_time="available",
            partition_cutoff="cutoff",
            window_start="window_start",
            window_end="window_end",
            horizon_end="horizon",
            analysis_as_of="as_of",
        ),
    )
    assert {
        "feature_after_cutoff",
        "duplicate_entity_time",
        "time_order_violation",
    } <= set(result.findings["reason"])
    audit_columns = result.point_in_time_profile.loc[
        result.point_in_time_profile["scope"] == "audit_input", "column"
    ].tolist()
    assert audit_columns == [
        "score",
        "action",
        "policy",
        "cost",
        "exposure",
        "constraint",
    ]
    assert result.findings.loc[
        result.findings["reason"] == "duplicate_entity_time", "sample_positions"
    ].iloc[0] == (0, 1)
    pd.testing.assert_frame_equal(frame, before)


def test_rare_selection_gap_and_partition_shift_boundaries() -> None:
    rare = audit_data_quality(
        pd.DataFrame({"target": [0] * 21 + [1]}),
        roles=DataAuditRoles(target="target"),
        config=DataAuditConfig(positive_label=1),
    )
    rare_row = rare.findings.loc[rare.findings["reason"] == "rare_target_class"].iloc[0]
    assert (rare_row["count"], rare_row["denominator"]) == (1, 22)

    selection = audit_data_quality(
        pd.DataFrame({"target": [1, None, 1, 1], "selection": ["a", "a", "b", "b"]}),
        roles=DataAuditRoles(target="target", selection="selection"),
        config=DataAuditConfig(positive_label=1),
    )
    assert "selection_outcome_support_gap" in selection.findings["reason"].tolist()

    partition_frame = pd.DataFrame(
        {"target": [0, 0, 1, 1], "partition": ["a", "a", "b", "b"]},
        index=[4, 4, 2, 2],
    )
    no_threshold = audit_data_quality(
        partition_frame,
        roles=DataAuditRoles(target="target", partition="partition"),
        config=DataAuditConfig(positive_label=1, partition_target_min_support=1),
    )
    assert "target_distribution_shift" not in no_threshold.findings["reason"].tolist()
    equality = audit_data_quality(
        partition_frame,
        roles=DataAuditRoles(target="target", partition="partition"),
        config=DataAuditConfig(
            positive_label=1,
            partition_target_rate_shift_threshold=0.5,
            partition_target_min_support=1,
        ),
    )
    assert equality.findings["reason"].tolist().count("target_distribution_shift") == 2
    insufficient = audit_data_quality(
        partition_frame,
        roles=DataAuditRoles(target="target", partition="partition"),
        config=DataAuditConfig(
            positive_label=1,
            partition_target_rate_shift_threshold=0.0,
            partition_target_min_support=3,
        ),
    )
    partition_rows = insufficient.slice_profile.loc[
        (insufficient.slice_profile["slice_role"] == "partition")
        & (insufficient.slice_profile["row_kind"] == "value")
    ]
    assert set(
        zip(partition_rows["quality_status"], partition_rows["quality_reason"])
    ) == {("undefined", "insufficient_support")}


def test_resource_closed_pairs_and_finding_detail_linkage() -> None:
    patterns = audit_data_quality(
        pd.DataFrame({"a": [None, 1], "b": [1, None]}),
        roles=DataAuditRoles(features=("a", "b")),
        config=DataAuditConfig(max_missing_patterns=1),
    )
    resource = patterns.resource_usage.loc[
        patterns.resource_usage["resource"] == "missing_patterns"
    ].iloc[0]
    assert (resource["status"], resource["reason"]) == (
        "not_verifiable",
        "budget_exceeded",
    )

    frame = pd.DataFrame(
        {"excluded": ["a", "b"], "target": [0, 1], "x": [0, 1], "y": [1, 0]},
        index=[3, 3],
    )
    roles = DataAuditRoles(
        target="target", features=("y", "x"), excluded_columns=("excluded",)
    )
    first = audit_data_quality(frame, roles=roles)
    second = audit_data_quality(frame, roles=roles)
    finding = first.findings.loc[first.findings["reason"] == "exact_target_copy"].iloc[
        0
    ]
    detail = first.column_profile.iloc[int(finding["detail_row_ordinal"])]
    assert finding["column"] == detail["column"] == "x"
    assert detail["finding_key"] == finding["finding_key"]
    y = first.column_profile.loc[first.column_profile["column"] == "y"].iloc[0]
    assert pd.isna(y["finding_key"])
    pd.testing.assert_frame_equal(first.findings, second.findings)


@pytest.mark.parametrize(
    ("current_empty", "finding_side"),
    [(False, "reference"), (True, "current")],
)
def test_empty_dataset_finding_links_to_final_side_row(
    current_empty: bool, finding_side: str
) -> None:
    empty = pd.DataFrame({"x": pd.Series(dtype="int64")})
    populated = pd.DataFrame({"x": [1, 2]}, index=[8, 8])
    current = empty if current_empty else populated
    reference = populated if current_empty else empty
    current_before = current.copy(deep=True)
    reference_before = reference.copy(deep=True)
    roles = DataAuditRoles()
    config = DataAuditConfig()

    first = audit_data_quality(current, reference=reference, roles=roles, config=config)
    second = audit_data_quality(
        current, reference=reference, roles=roles, config=config
    )
    finding = first.findings.loc[
        (first.findings["reason"] == "empty_dataset")
        & (first.findings["dataset_role"] == finding_side)
    ].iloc[0]
    expected_ordinal = int(
        np.flatnonzero(first.dataset_profile["side"].eq(finding_side).to_numpy())[0]
    )
    other_ordinal = int(
        np.flatnonzero(first.dataset_profile["side"].ne(finding_side).to_numpy())[0]
    )

    assert finding["detail_table"] == "dataset_profile"
    assert int(finding["detail_row_ordinal"]) == expected_ordinal
    assert expected_ordinal != other_ordinal
    assert (
        first.dataset_profile.iloc[expected_ordinal]["finding_key"]
        == finding["finding_key"]
    )
    assert (
        first.dataset_profile.iloc[other_ordinal]["finding_key"]
        != finding["finding_key"]
    )
    payload = [
        finding["category"],
        finding["scope"],
        finding["dataset_role"],
        -1,
        finding["metric_key"],
        finding["reason"],
        finding["detail_table"],
        expected_ordinal,
        -1,
    ]
    expected_key = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert finding["finding_key"] == expected_key
    pd.testing.assert_frame_equal(first.findings, second.findings)
    pd.testing.assert_frame_equal(first.dataset_profile, second.dataset_profile)
    pd.testing.assert_frame_equal(current, current_before)
    pd.testing.assert_frame_equal(reference, reference_before)
    assert roles == DataAuditRoles()
    assert config == DataAuditConfig()


def test_two_sided_dataset_findings_keep_side_specific_linkage() -> None:
    current = pd.DataFrame({"x": pd.Series(dtype="int64")})
    reference = pd.DataFrame({"x": [1, 2]})
    result = audit_data_quality(
        current,
        reference=reference,
        roles=DataAuditRoles(features=()),
    )
    dataset_findings = result.findings.loc[
        result.findings["detail_table"] == "dataset_profile"
    ]
    assert set(dataset_findings["dataset_role"]) == {"current", "reference"}
    for side in ("current", "reference"):
        expected_ordinal = int(
            np.flatnonzero(result.dataset_profile["side"].eq(side).to_numpy())[0]
        )
        side_findings = dataset_findings.loc[dataset_findings["dataset_role"] == side]
        assert set(side_findings["detail_row_ordinal"]) == {expected_ordinal}
        assert result.dataset_profile.iloc[expected_ordinal]["finding_key"] in set(
            side_findings["finding_key"]
        )
    current_keys = set(
        dataset_findings.loc[
            dataset_findings["dataset_role"] == "current", "finding_key"
        ]
    )
    reference_keys = set(
        dataset_findings.loc[
            dataset_findings["dataset_role"] == "reference", "finding_key"
        ]
    )
    assert current_keys.isdisjoint(reference_keys)


def test_dataset_relink_uses_final_side_identity_after_row_reorder() -> None:
    result = audit_data_quality(
        pd.DataFrame({"x": [1]}),
        reference=pd.DataFrame({"x": pd.Series(dtype="int64")}),
    )
    finding = (
        result.findings.loc[
            (result.findings["reason"] == "empty_dataset")
            & (result.findings["dataset_role"] == "reference")
        ]
        .iloc[0]
        .to_dict()
    )
    reordered = result.dataset_profile.iloc[::-1].reset_index(drop=True)
    finding["detail_row_ordinal"] = 0
    original_key = finding["finding_key"]

    _relink_findings([finding], {"dataset_profile": reordered})

    expected_ordinal = int(
        np.flatnonzero(reordered["side"].eq("reference").to_numpy())[0]
    )
    assert int(finding["detail_row_ordinal"]) == expected_ordinal == 0
    assert finding["finding_key"] != original_key


def test_feature_cutoff_boundaries_unknown_timezone_and_safe_time_order() -> None:
    frame = pd.DataFrame(
        {
            "entity": [1, 2, 3, 4],
            "x": [1, 2, 3, 4],
            "available": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-04", None]
            ),
            "observed": pd.to_datetime(["2025-01-05"] * 4),
            "cutoff": pd.to_datetime(["2025-01-02"] * 4),
        }
    )
    result = audit_data_quality(
        frame,
        roles=DataAuditRoles(
            features=("x",),
            row_identifier="entity",
            observation_time="observed",
            shared_feature_available_time="available",
            partition_cutoff="cutoff",
        ),
    )
    feature = result.point_in_time_profile.loc[
        result.point_in_time_profile["scope"] == "feature"
    ].iloc[0]
    assert (
        feature["evaluated_count"],
        feature["violation_count"],
        feature["not_verifiable_count"],
        feature["status"],
        feature["reason"],
    ) == (3, 1, 1, "not_verifiable", "missing_availability_metadata")
    cutoff = result.findings.loc[
        result.findings["reason"] == "feature_after_cutoff"
    ].iloc[0]
    assert cutoff["sample_positions"] == (2,)
    assert "duplicate_entity_time" not in result.findings["reason"].tolist()
    assert "time_order_violation" not in result.findings["reason"].tolist()

    timezone_frame = pd.DataFrame(
        {
            "x": [1],
            "available": [pd.Timestamp("2025-01-01")],
            "observed": [pd.Timestamp("2025-01-02")],
            "cutoff": [pd.Timestamp("2025-01-02", tz=timezone.utc)],
        }
    )
    timezone_result = audit_data_quality(
        timezone_frame,
        roles=DataAuditRoles(
            features=("x",),
            observation_time="observed",
            shared_feature_available_time="available",
            partition_cutoff="cutoff",
        ),
    )
    timezone_feature = timezone_result.point_in_time_profile.loc[
        timezone_result.point_in_time_profile["scope"] == "feature"
    ].iloc[0]
    assert (timezone_feature["status"], timezone_feature["reason"]) == (
        "not_verifiable",
        "timezone_mismatch",
    )


def test_audit_input_missing_metadata_is_structured_without_raw_values() -> None:
    frame = pd.DataFrame(
        {
            "score": [0.1, 0.2],
            "fold": [0, None],
            "observed": pd.to_datetime(["2025-01-01", None]),
            "window_start": pd.to_datetime(["2024-12-01", "2024-12-01"]),
            "window_end": pd.to_datetime(["2024-12-31", "2024-12-31"]),
            "horizon": pd.to_datetime(["2025-01-31", "2025-01-31"]),
            "as_of": pd.to_datetime(["2025-02-01", "2025-02-01"]),
        },
        index=[5, 5],
    )
    result = audit_data_quality(
        frame,
        roles=DataAuditRoles(
            features=(),
            score_columns=("score",),
            fold="fold",
            observation_time="observed",
            window_start="window_start",
            window_end="window_end",
            horizon_end="horizon",
            analysis_as_of="as_of",
        ),
    )
    row = result.point_in_time_profile.loc[
        result.point_in_time_profile["scope"] == "audit_input"
    ].iloc[0]
    assert (
        row["column"],
        row["evaluated_count"],
        row["not_verifiable_count"],
        row["status"],
        row["reason"],
    ) == ("score", 1, 1, "not_verifiable", "missing_availability_metadata")
    assert tuple(result.point_in_time_profile.columns) == (
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


def test_rare_target_exact_boundaries_and_partition_pooled_support() -> None:
    below = audit_data_quality(
        pd.DataFrame({"target": [1] * 19 + [0] * 381}),
        roles=DataAuditRoles(target="target"),
        config=DataAuditConfig(positive_label=1),
    )
    equality = audit_data_quality(
        pd.DataFrame({"target": [1] * 20 + [0] * 380}),
        roles=DataAuditRoles(target="target"),
        config=DataAuditConfig(positive_label=1),
    )
    above = audit_data_quality(
        pd.DataFrame({"target": [1] * 21 + [0] * 379 + [None]}),
        roles=DataAuditRoles(target="target"),
        config=DataAuditConfig(positive_label=1),
    )
    assert below.findings["reason"].tolist().count("rare_target_class") == 1
    assert "rare_target_class" not in equality.findings["reason"].tolist()
    assert "rare_target_class" not in above.findings["reason"].tolist()
    assert "target_missing" in above.findings["reason"].tolist()

    one_selection_bucket = audit_data_quality(
        pd.DataFrame({"target": [1, None], "selection": ["a", "a"]}),
        roles=DataAuditRoles(target="target", selection="selection"),
        config=DataAuditConfig(positive_label=1),
    )
    assert (
        "selection_outcome_support_gap"
        not in one_selection_bucket.findings["reason"].tolist()
    )

    pooled_too_small = audit_data_quality(
        pd.DataFrame({"target": [0, 1, 0, 1], "partition": ["a", "a", "b", "b"]}),
        roles=DataAuditRoles(target="target", partition="partition"),
        config=DataAuditConfig(
            positive_label=1,
            partition_target_rate_shift_threshold=0.0,
            partition_target_min_support=5,
        ),
    )
    values = pooled_too_small.slice_profile.loc[
        (pooled_too_small.slice_profile["slice_role"] == "partition")
        & (pooled_too_small.slice_profile["row_kind"] == "value")
    ]
    assert set(values["quality_reason"]) == {"insufficient_support"}
    assert (
        "target_distribution_shift" not in pooled_too_small.findings["reason"].tolist()
    )


def test_resource_status_reason_matrix_for_truncation_and_hard_limit() -> None:
    frame = pd.DataFrame(
        {
            "a": np.arange(25, dtype=float),
            "b": np.arange(25, dtype=float) * 2,
            "c": np.arange(25, dtype=float) * 3,
        }
    )
    result = audit_data_quality(
        frame,
        roles=DataAuditRoles(features=("a", "b", "c")),
        config=DataAuditConfig(
            max_collinearity_columns=2,
            max_unique_inspection_rows=10,
            duplicate_scan_row_limit=10,
        ),
    )
    pairs = set(zip(result.resource_usage["status"], result.resource_usage["reason"]))
    assert pairs <= {
        ("available", "computed"),
        ("not_verifiable", "budget_exceeded"),
        ("not_verifiable", "duplicate_scan_budget"),
    }
    by_resource = result.resource_usage.set_index("resource")
    assert tuple(by_resource.loc["collinearity_columns", ["status", "reason"]]) == (
        "not_verifiable",
        "budget_exceeded",
    )
    assert tuple(by_resource.loc["unique_inspection_rows", ["status", "reason"]]) == (
        "not_verifiable",
        "budget_exceeded",
    )
    assert tuple(by_resource.loc["duplicate_scan_rows", ["status", "reason"]]) == (
        "not_verifiable",
        "duplicate_scan_budget",
    )
    with pytest.raises(ValueError, match="max_columns_exceeded"):
        audit_data_quality(frame, config=DataAuditConfig(max_columns=2))


def test_atomic_audit_rule_parity_for_allowed_range_cross_and_monotonic() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 3.0, None, np.inf],
            "end": [2.0, 2.0, 4.0, 5.0],
            "ordered": [1.0, 3.0, 2.0, None],
        },
        index=[9, 9, 1, 1],
    )
    operands = {
        "allowed_value_violation": (
            "x",
            "in",
            _ConditionOperand("literal", (1, 2.0)),
            ["true", "false", "unknown", "unknown"],
            (1,),
        ),
        "range_violation": (
            "x",
            "le",
            _ConditionOperand("literal", 2.0),
            ["true", "false", "unknown", "unknown"],
            (1,),
        ),
        "cross_column_order_violation": (
            "x",
            "le",
            _ConditionOperand("column", "end"),
            ["true", "false", "unknown", "unknown"],
            (1,),
        ),
    }
    for _, (column, operator, right, truth, _) in operands.items():
        atomic = _evaluate_atomic_condition(
            frame,
            operator=operator,
            left=_ConditionOperand("column", column),
            right=right,
            root_version="v1",
        )
        assert atomic.truth.tolist() == truth
        assert atomic.truth.tolist().count("unknown") == 2

    result = audit_data_quality(
        frame,
        config=DataAuditConfig(
            column_rules=(
                ColumnAuditRule(
                    "x",
                    maximum=2.0,
                    allowed_values=(1, 2.0),
                    not_after_columns=("end",),
                ),
                ColumnAuditRule("ordered", nondecreasing=True),
            )
        ),
    )
    for reason, (_, _, _, _, samples) in operands.items():
        finding = result.findings.loc[result.findings["reason"] == reason].iloc[0]
        assert (finding["count"], finding["sample_positions"]) == (1, samples)

    pair = pd.DataFrame({"prior": [1.0, 3.0, 2.0], "current": [3.0, 2.0, None]})
    monotonic = _evaluate_atomic_condition(
        pair,
        operator="le",
        left=_ConditionOperand("column", "prior"),
        right=_ConditionOperand("column", "current"),
        root_version="v1",
    )
    assert monotonic.truth.tolist() == ["true", "false", "unknown"]
    finding = result.findings.loc[
        result.findings["reason"] == "monotonic_time_violation"
    ].iloc[0]
    assert (finding["count"], finding["denominator"], finding["sample_positions"]) == (
        1,
        3,
        (2,),
    )
