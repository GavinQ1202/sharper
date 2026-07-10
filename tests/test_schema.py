"""Contract tests for deterministic schema inference."""

import numpy as np
import pandas as pd
import pytest

from sharper import ColumnSchema, SchemaReport, TargetCandidate, infer_schema

_COLUMN_REASON_CODES = {
    "pandas_numeric_dtype",
    "pandas_boolean_dtype",
    "boolean_values_only",
    "pandas_datetime_dtype",
    "pandas_category_dtype",
    "string_datetime_parse_rate_met",
    "identifier_name_pattern",
    "identifier_high_unique_rate",
    "categorical_unique_threshold_met",
    "text_high_unique_rate",
    "all_missing",
    "empty_dataframe",
    "mixed_object_unknown",
    "fallback_unknown",
}
_TARGET_REASON_CODES = {
    "explicit_target",
    "target_name_exact",
    "target_name_contains_target",
    "target_name_contains_label",
    "target_name_contains_outcome",
    "target_name_is_y",
    "classification_low_cardinality",
    "classification_boolean",
    "classification_categorical",
    "regression_numeric_high_cardinality",
    "excluded_identifier",
}


def _by_name(report: SchemaReport) -> dict[str, ColumnSchema]:
    return {column.name: column for column in report.columns}


def test_infer_schema_mixed_dataframe_contract() -> None:
    """Physical dtypes and value rules produce the frozen logical types."""
    text_values = [f"sentence {index}" for index in range(60)] + ["sentence 0"]
    frame = pd.DataFrame(
        {
            "amount": [1.5, 2.5, 3.5] + [1.5] * 58,
            "segment": ["a", "b", "a"] + ["a"] * 58,
            "event_date": pd.date_range("2024-01-01", periods=61),
            "active": pd.Series(([True, False, True] * 21)[:61], dtype="boolean"),
            "notes": text_values,
            "customer_id": list(range(61)),
            "unknown": pd.Series([None] * 61, dtype="object"),
        }
    )

    report = infer_schema(frame)
    columns = _by_name(report)

    assert isinstance(report, SchemaReport)
    assert all(isinstance(column, ColumnSchema) for column in report.columns)
    assert [column.name for column in report.columns] == list(frame.columns)
    assert columns["amount"].logical_type == "numeric"
    assert columns["amount"].reasons == ["pandas_numeric_dtype"]
    assert columns["segment"].logical_type == "categorical"
    assert columns["event_date"].logical_type == "datetime"
    assert columns["active"].logical_type == "boolean"
    assert columns["notes"].logical_type == "text"
    assert columns["customer_id"].logical_type == "identifier"
    assert columns["customer_id"].is_id_like is True
    assert columns["unknown"].logical_type == "unknown"
    assert columns["unknown"].reasons == ["all_missing"]
    assert report.logical_type_counts == {
        "numeric": 1,
        "categorical": 1,
        "datetime": 1,
        "boolean": 1,
        "text": 1,
        "identifier": 1,
        "unknown": 1,
    }


def test_nullable_counts_rates_and_constant_semantics_match_pandas() -> None:
    """Missing and unique fields use their frozen denominators."""
    frame = pd.DataFrame(
        {
            "constant": pd.Series([7.0, 7.0, np.nan], dtype="Float64"),
            "all_missing": pd.Series([pd.NA, pd.NA, pd.NA], dtype="string"),
        }
    )

    columns = _by_name(infer_schema(frame))

    constant = columns["constant"]
    assert constant.nullable is True
    assert constant.missing_count == int(frame["constant"].isna().sum())
    assert constant.missing_rate == pytest.approx(1 / 3)
    assert constant.unique_count == int(frame["constant"].nunique(dropna=True))
    assert constant.unique_rate == pytest.approx(1 / 2)
    assert constant.is_constant is True

    all_missing = columns["all_missing"]
    assert all_missing.missing_rate == 1.0
    assert all_missing.unique_count == 0
    assert all_missing.unique_rate == 0.0
    assert all_missing.is_constant is False
    assert all_missing.confidence == 0.5


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ([True, False, True], "object"),
        (["true", " FALSE ", "True"], "object"),
        (["true", "false", "TRUE"], "category"),
        ([" true ", "FALSE", "True"], "string"),
    ],
)
def test_non_bool_dtype_boolean_values_use_frozen_rule(
    values: list[object],
    dtype: str,
) -> None:
    """Normalized true/false values in supported dtypes use their own reason."""
    frame = pd.DataFrame({"flag": pd.Series(values, dtype=dtype)})

    column = infer_schema(frame).columns[0]

    assert column.logical_type == "boolean"
    assert column.confidence == 0.8
    assert column.reasons == ["boolean_values_only"]


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        (["0", "1", "0"], "object"),
        (["0", "1", "0"], "category"),
        (["0", "1", "0"], "string"),
    ],
)
def test_zero_one_string_tokens_are_not_boolean(
    values: list[str],
    dtype: str,
) -> None:
    """String 0/1 tokens remain categorical rather than boolean."""
    frame = pd.DataFrame({"flag": pd.Series(values, dtype=dtype)})

    column = infer_schema(frame).columns[0]

    assert column.logical_type == "categorical"
    expected_reason = (
        "pandas_category_dtype"
        if dtype == "category"
        else "categorical_unique_threshold_met"
    )
    assert column.reasons == [expected_reason]


def test_numeric_zero_one_values_are_not_boolean() -> None:
    """Repeated numeric 0/1 values retain the numeric logical type."""
    frame = pd.DataFrame({"flag": [0, 1, 0, 1]})

    column = infer_schema(frame).columns[0]

    assert column.logical_type == "numeric"
    assert column.reasons == ["pandas_numeric_dtype"]


def test_date_string_detection_is_read_only_and_deterministic() -> None:
    """At least 80 percent parseable strings produce inferred datetime."""
    frame = pd.DataFrame(
        {
            "when": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "not-a-date",
            ]
        }
    )
    before = frame.copy(deep=True)

    column = infer_schema(frame).columns[0]

    assert column.logical_type == "datetime"
    assert column.confidence == 0.85
    assert column.reasons == ["string_datetime_parse_rate_met"]
    pd.testing.assert_frame_equal(frame, before)
    assert frame["when"].dtype == before["when"].dtype


@pytest.mark.parametrize("name", ["mixed", "customer_id", "event_uuid", "lookup_key"])
def test_all_unique_mixed_object_is_unknown_before_identifier(name: str) -> None:
    """Mixed objects remain unknown despite uniqueness or identifier-like names."""
    frame = pd.DataFrame({name: pd.Series([1, "two", 3.5], dtype="object")})

    column = infer_schema(frame).columns[0]

    assert column.unique_rate == 1.0
    assert column.logical_type == "unknown"
    assert column.is_id_like is False
    assert column.confidence == 0.5
    assert column.reasons == ["mixed_object_unknown"]


def test_identifier_threshold_boundary_and_reasons() -> None:
    """The configured inclusive threshold controls identifier inference."""
    frame = pd.DataFrame({"account_id": [1, 2, 3, 4, 4]})

    at_boundary = infer_schema(frame, id_threshold=0.80).columns[0]
    above_boundary = infer_schema(frame, id_threshold=0.81).columns[0]

    assert at_boundary.logical_type == "identifier"
    assert at_boundary.confidence == 0.9
    assert at_boundary.reasons == [
        "identifier_name_pattern",
        "identifier_high_unique_rate",
    ]
    assert above_boundary.logical_type == "numeric"


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.01])
def test_invalid_identifier_threshold_raises(threshold: float) -> None:
    """Thresholds outside the frozen interval are rejected."""
    with pytest.raises(ValueError, match="id_threshold"):
        infer_schema(pd.DataFrame({"value": [1.0]}), id_threshold=threshold)


def test_empty_dataframe_contracts() -> None:
    """Both zero-column and typed zero-row frames return stable unknown schemas."""
    zero_columns = infer_schema(pd.DataFrame())
    assert zero_columns.n_rows == 0
    assert zero_columns.n_columns == 0
    assert zero_columns.columns == []
    assert zero_columns.logical_type_counts == dict.fromkeys(
        [
            "numeric",
            "categorical",
            "datetime",
            "boolean",
            "text",
            "identifier",
            "unknown",
        ],
        0,
    )

    typed = pd.DataFrame(
        {
            "number": pd.Series(dtype="float64"),
            "flag": pd.Series(dtype="boolean"),
        }
    )
    report = infer_schema(typed)
    assert [column.logical_type for column in report.columns] == [
        "unknown",
        "unknown",
    ]
    assert all(column.confidence == 0.5 for column in report.columns)
    assert all(column.reasons == ["empty_dataframe"] for column in report.columns)
    assert all(column.nullable is False for column in report.columns)


def test_all_missing_string_column_is_unknown_not_boolean() -> None:
    """The all-missing rule takes precedence over boolean-token inference."""
    frame = pd.DataFrame({"flag": pd.Series([pd.NA, pd.NA], dtype="string")})

    column = infer_schema(frame).columns[0]

    assert column.logical_type == "unknown"
    assert column.confidence == 0.5
    assert column.reasons == ["all_missing"]


def test_non_string_and_duplicate_column_names_raise() -> None:
    """Unsupported or ambiguous column labels fail without conversion."""
    with pytest.raises(
        ValueError,
        match="DataFrame column names must all be strings",
    ):
        infer_schema(pd.DataFrame([[1]], columns=[1]))

    with pytest.raises(ValueError, match="must be unique"):
        infer_schema(pd.DataFrame([[1, 2]], columns=["value", "value"]))


def test_explicit_target_is_first_and_missing_target_raises() -> None:
    """An explicit target is suggested first and validated."""
    frame = pd.DataFrame(
        {
            "feature_label": ["a", "b", "a", "b"],
            "chosen": [0.0, 1.0, 0.0, 1.0],
        }
    )

    report = infer_schema(frame, target="chosen")

    assert report.target_candidates[0] == TargetCandidate(
        name="chosen",
        suggested_task_type="classification",
        confidence=0.9,
        reasons=["explicit_target", "classification_low_cardinality"],
    )
    assert isinstance(report.target_candidates[0], TargetCandidate)
    with pytest.raises(ValueError, match="target column not found"):
        infer_schema(frame, target="absent")


def test_target_name_signals_are_case_insensitive_and_ordered() -> None:
    """All frozen automatic name signals retain original names and reason codes."""
    frame = pd.DataFrame(
        {
            "model_target": ["a", "b", "a", "b"],
            "Risk_Label": ["a", "b", "a", "b"],
            "OUTCOME": ["a", "b", "a", "b"],
            "y": [True, False, True, False],
        }
    )

    candidates = infer_schema(frame).target_candidates

    assert [candidate.name for candidate in candidates] == list(frame.columns)
    assert candidates[0].reasons == [
        "target_name_contains_target",
        "classification_categorical",
    ]
    assert candidates[0].confidence == 0.75
    assert candidates[1].reasons == [
        "target_name_contains_label",
        "classification_categorical",
    ]
    assert candidates[1].confidence == 0.75
    assert candidates[2].reasons == [
        "target_name_contains_outcome",
        "classification_categorical",
    ]
    assert candidates[2].confidence == 0.9
    assert candidates[3].reasons == [
        "target_name_is_y",
        "classification_boolean",
    ]
    assert candidates[3].confidence == 0.9
    assert all(candidate.confidence != 0.6 for candidate in candidates)


def test_numeric_target_task_rules_and_identifier_exclusion() -> None:
    """Numeric cardinality determines task type and automatic IDs are excluded."""
    frame = pd.DataFrame(
        {
            "target": [float(index) for index in range(25)],
            "low_target": [float(index % 3) for index in range(25)],
            "record_id": list(range(25)),
        }
    )

    candidates = infer_schema(frame).target_candidates
    by_name = {candidate.name: candidate for candidate in candidates}

    assert by_name["target"].suggested_task_type == "regression"
    assert "regression_numeric_high_cardinality" in by_name["target"].reasons
    assert by_name["low_target"].suggested_task_type == "classification"
    assert "classification_low_cardinality" in by_name["low_target"].reasons
    assert "record_id" not in by_name

    explicit_id = infer_schema(frame, target="record_id").target_candidates[0]
    assert explicit_id.suggested_task_type == "unknown"
    assert explicit_id.confidence == 0.75
    assert explicit_id.reasons == ["explicit_target"]


def test_reason_codes_and_confidences_are_closed_and_deterministic() -> None:
    """Repeated inference emits only frozen values and codes."""
    frame = pd.DataFrame(
        {
            "value": [1.5, 2.5, 1.5],
            "target_label": ["yes", "no", "yes"],
            "missing": [None, None, None],
        }
    )

    first = infer_schema(frame)
    second = infer_schema(frame)

    assert first == second
    assert all(
        column.confidence in {1.0, 0.9, 0.85, 0.8, 0.5} for column in first.columns
    )
    assert all(set(column.reasons) <= _COLUMN_REASON_CODES for column in first.columns)
    assert all(
        candidate.confidence in {0.9, 0.75} for candidate in first.target_candidates
    )
    assert all(
        set(candidate.reasons) <= _TARGET_REASON_CODES
        for candidate in first.target_candidates
    )


def test_infer_schema_does_not_mutate_input() -> None:
    """Inference leaves values, dtypes, and labels untouched."""
    frame = pd.DataFrame(
        {
            "mixed_id": pd.Series([1, "two"], dtype="object"),
            "boolean_tokens": pd.Series([" TRUE ", "false"], dtype="string"),
            "value": [1.0, np.nan],
        }
    )
    before = frame.copy(deep=True)

    infer_schema(frame)

    pd.testing.assert_frame_equal(frame, before)
