"""Contract tests for Task 09 feature suggestions and stateless derivation."""

from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import sharper.features as features_module
from sharper import (
    FeatureSuggestion,
    derive_features,
    infer_schema,
    suggest_feature_derivations,
)


def _arithmetic_suggestion(
    feature_type: str, left: str = "a", right: str = "b"
) -> FeatureSuggestion:
    operators = {
        "ratio": ("div", "/", 2),
        "difference": ("minus", "-", 3),
        "product": ("times", "*", 4),
    }
    name_operator, formula_operator, priority = operators[feature_type]
    return FeatureSuggestion(
        name=f"{left}__{name_operator}__{right}",
        feature_type=feature_type,
        source_columns=(left, right),
        formula=f"{left} {formula_operator} {right}",
        parameters=(),
        reason="numeric_pair_arithmetic",
        risk="low",
        requires_fit=False,
        priority=priority,
    )


def _datetime_suggestion(
    feature_type: str, column: str = "when", reference: str = "2024-01-10"
) -> FeatureSuggestion:
    suffixes = {
        "datetime_year": ("year", f"{column}.dt.year", "datetime_component", "low"),
        "datetime_month": (
            "month",
            f"{column}.dt.month",
            "datetime_component",
            "low",
        ),
        "datetime_quarter": (
            "quarter",
            f"{column}.dt.quarter",
            "datetime_component",
            "low",
        ),
        "datetime_dayofweek": (
            "dayofweek",
            f"{column}.dt.dayofweek",
            "datetime_component",
            "low",
        ),
        "datetime_is_weekend": (
            "is_weekend",
            f"{column}.dt.is_weekend",
            "datetime_component",
            "low",
        ),
    }
    if feature_type == "datetime_days_since_reference":
        return FeatureSuggestion(
            name=f"{column}__days_since__{reference.replace('-', '_')}",
            feature_type=feature_type,
            source_columns=(column,),
            formula=f"{reference} - {column}",
            parameters=(("reference_date", reference),),
            reason="explicit_reference_date",
            risk="medium",
            requires_fit=False,
            priority=1,
        )
    suffix, formula, reason, risk = suffixes[feature_type]
    return FeatureSuggestion(
        name=f"{column}__{suffix}",
        feature_type=feature_type,
        source_columns=(column,),
        formula=formula,
        parameters=(),
        reason=reason,
        risk=risk,
        requires_fit=False,
        priority=1,
    )


def _mixed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": ["yes", "no", "yes", "no", "yes", "no"],
            "excluded": [1, 1, 2, 2, 3, 3],
            "customer_id": [101, 102, 103, 104, 105, 106],
            "all_missing": pd.Series([pd.NA] * 6, dtype="Float64"),
            "constant": [7, 7, 7, 7, 7, 7],
            "complex": [1 + 1j, 2 + 2j, 1 + 1j, 2 + 2j, 1 + 1j, 2 + 2j],
            "number": [1, 1, 2, 2, 3, 3],
            "duplicate_number": [1, 1, 2, 2, 3, 3],
            "when": pd.to_datetime(["2024-01-01", "2024-01-02"] * 3),
            "segment": pd.Series(["a", "b", "a", "b", "a", "b"], dtype="string"),
        }
    )


def _budget_frame() -> pd.DataFrame:
    rows = np.arange(8)
    numeric_values = (
        rows % 2,
        (rows // 2) % 2,
        (rows // 4) % 2,
        (rows + 1) % 3,
        (rows + 2) % 3,
        (rows + 1) % 4,
        (rows + 2) % 4,
    )
    data: dict[str, object] = {
        f"n{index}": values for index, values in enumerate(numeric_values)
    }
    base = pd.Timestamp("2024-01-01")
    for index in range(4):
        data[f"d{index}"] = pd.Series(
            [base + pd.Timedelta(days=(row + index) % (index + 2)) for row in rows]
        )
    for index in range(6):
        data[f"c{index}"] = pd.Series(
            [f"g{(row + index) % (index + 2)}" for row in rows], dtype="string"
        )
    data["target"] = pd.Series(["yes", "no"] * 4, dtype="string")
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    "function",
    [suggest_feature_derivations, derive_features],
)
def test_shared_validation_rejects_non_dataframe(function: object) -> None:
    """Both entry points enforce the pandas boundary."""
    arguments = ({"x": [1]},) if function is suggest_feature_derivations else ({}, [])
    with pytest.raises(ValueError, match="df must be a pandas DataFrame"):
        function(*arguments)


@pytest.mark.parametrize("column", [1, ("x",)])
def test_shared_validation_rejects_non_string_columns(column: object) -> None:
    """Both entry points reject non-string DataFrame labels."""
    frame = pd.DataFrame([[1]], columns=[column])
    for function, args in (
        (suggest_feature_derivations, (frame,)),
        (derive_features, (frame, [])),
    ):
        with pytest.raises(ValueError, match="column names must all be strings"):
            function(*args)


def test_shared_validation_rejects_duplicate_columns() -> None:
    """Duplicate DataFrame labels use the frozen message."""
    frame = pd.DataFrame([[1, 2]], columns=["x", "x"])
    for function, args in (
        (suggest_feature_derivations, (frame,)),
        (derive_features, (frame, [])),
    ):
        with pytest.raises(ValueError, match="duplicate DataFrame column names"):
            function(*args)


def test_column_partition_precedence_and_duplicate_domain() -> None:
    """Every column has one state and unsupported columns precede duplicates."""
    frame = _mixed_frame()
    report = suggest_feature_derivations(
        frame, target="target", exclude_columns=("excluded",)
    )

    assert report.excluded_columns == ("target", "excluded")
    assert report.skipped_columns == (
        "customer_id",
        "all_missing",
        "constant",
        "complex",
        "duplicate_number",
    )
    assert report.eligible_columns == ("number", "when", "segment")
    assert report.skipped_reasons == {
        "target": "target_column",
        "excluded": "explicitly_excluded",
        "customer_id": "identifier_like",
        "all_missing": "all_missing",
        "constant": "constant",
        "complex": "unsupported_dtype",
        "duplicate_number": "duplicate_content",
    }
    states = report.eligible_columns + report.excluded_columns + report.skipped_columns
    assert set(states) == set(frame.columns)
    assert len(states) == len(set(states)) == len(frame.columns)


def test_target_exclusion_overlap_uses_target_reason_once() -> None:
    """Target precedence wins without duplicate state entries."""
    frame = pd.DataFrame({"target": ["a", "b", "a"], "x": [1, 1, 2]})
    report = suggest_feature_derivations(
        frame, target="target", exclude_columns=("target",)
    )
    assert report.requested_exclusions == ("target",)
    assert report.excluded_columns == ("target",)
    assert report.skipped_reasons["target"] == "target_column"


def test_unsupported_columns_do_not_participate_in_duplicate_detection() -> None:
    """Identical unsupported columns cannot hide an otherwise eligible source."""
    values = [1, 2, 1, 2]
    frame = pd.DataFrame(
        {
            "complex_a": np.asarray(values, dtype="complex128"),
            "complex_b": np.asarray(values, dtype="complex128"),
            "numeric": values,
        }
    )
    report = suggest_feature_derivations(frame)
    assert report.eligible_columns == ("numeric",)
    assert report.skipped_reasons == {
        "complex_a": "unsupported_dtype",
        "complex_b": "unsupported_dtype",
    }


def test_duplicate_content_uses_series_equals_despite_names_and_missing() -> None:
    """Series names differ, while equal values, dtype, and missing masks still match."""
    frame = pd.DataFrame(
        {
            "first": pd.Series([1, pd.NA, 2, 1], dtype="Int64"),
            "second": pd.Series([1, pd.NA, 2, 1], dtype="Int64"),
        }
    )
    report = suggest_feature_derivations(frame)
    assert frame["first"].name != frame["second"].name
    assert frame["first"].equals(frame["second"])
    assert report.eligible_columns == ("first",)
    assert report.skipped_reasons == {"second": "duplicate_content"}


def test_bool_is_categorical_and_timedelta_is_unsupported() -> None:
    """Bool review candidates are allowed while timedeltas stay unsupported."""
    frame = pd.DataFrame(
        {
            "flag": pd.Series([True, False, True, False], dtype="boolean"),
            "duration": pd.to_timedelta([1, 2, 1, 2], unit="D"),
        }
    )
    report = suggest_feature_derivations(frame)
    assert report.eligible_columns == ("flag",)
    assert report.skipped_reasons == {"duration": "unsupported_dtype"}
    candidate = report.suggestions[0]
    assert candidate.feature_type == "group_aggregate_candidate"
    assert candidate.source_columns == ("flag",)


def test_schema_resolution_and_mismatch_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schema inference is the only implicit prerequisite and mismatches fail."""
    frame = pd.DataFrame({"x": [1, 1, 2]})
    schema = infer_schema(frame)
    calls: list[pd.DataFrame] = []

    def fake_infer(value: pd.DataFrame) -> object:
        calls.append(value)
        return schema

    monkeypatch.setattr(features_module, "infer_schema", fake_infer)
    suggest_feature_derivations(frame)
    assert calls == [frame]

    with pytest.raises(ValueError, match="schema must be a SchemaReport"):
        suggest_feature_derivations(frame, schema="bad")
    mismatch = infer_schema(pd.DataFrame({"y": [1, 1, 2]}))
    with pytest.raises(ValueError, match="schema does not match DataFrame"):
        suggest_feature_derivations(frame, schema=mismatch)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target": 1}, "target must be a string"),
        ({"target": "missing"}, "target column not found"),
        ({"exclude_columns": [1]}, "columns must contain only strings"),
        ({"exclude_columns": ["x", "x"]}, "duplicate column parameter"),
        ({"exclude_columns": ["missing"]}, "column not found"),
        ({"max_suggestions": True}, "max_suggestions must be a positive integer"),
        ({"max_suggestions": 0}, "max_suggestions must be a positive integer"),
    ],
)
def test_suggestion_parameter_validation(
    kwargs: dict[str, object], message: str
) -> None:
    """Target, exclusions, and budget use stable contract messages."""
    with pytest.raises(ValueError, match=message):
        suggest_feature_derivations(pd.DataFrame({"x": [1, 1, 2]}), **kwargs)


@pytest.mark.parametrize(
    "reference",
    [
        "2024-02-03",
        date(2024, 2, 3),
        datetime(2024, 2, 3, 15, 30),
        pd.Timestamp("2024-02-03 15:30"),
    ],
)
def test_reference_date_dispatch_and_normalization(reference: object) -> None:
    """Every approved reference input normalizes to the same calendar date."""
    frame = pd.DataFrame({"x": [1, 1, 2]})
    report = suggest_feature_derivations(frame, reference_date=reference)
    assert report.reference_date == "2024-02-03"
    assert all(
        item.feature_type != "datetime_days_since_reference"
        for item in report.suggestions
    )


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        (pd.NaT, "reference_date must be a valid date"),
        ("2024-2-03", "reference_date must be a valid date"),
        ("2024-02-30", "reference_date must be a valid date"),
        (
            datetime(2024, 2, 3, tzinfo=timezone.utc),
            "reference_date must be timezone-naive",
        ),
        (
            pd.Timestamp("2024-02-03", tz="UTC"),
            "reference_date must be timezone-naive",
        ),
    ],
)
def test_invalid_reference_dates(reference: object, message: str) -> None:
    """Invalid and timezone-aware reference dates fail deterministically."""
    with pytest.raises(ValueError, match=message):
        suggest_feature_derivations(
            pd.DataFrame({"x": [1, 1, 2]}), reference_date=reference
        )


def test_candidate_generation_order_fields_and_target_independence() -> None:
    """Approved candidates have canonical fields and ignore target values."""
    frame = pd.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [2, 3, 2, 3],
            "when": pd.to_datetime(["2024-01-01", "2024-01-06"] * 2),
            "segment": pd.Series(["x", "y", "x", "y"], dtype="string"),
            "target": pd.Series(["yes", "no", "yes", "no"], dtype="string"),
        }
    )
    report = suggest_feature_derivations(
        frame, target="target", reference_date="2024-01-10"
    )
    types = [item.feature_type for item in report.suggestions]
    assert types[:6] == [
        "datetime_year",
        "datetime_month",
        "datetime_quarter",
        "datetime_dayofweek",
        "datetime_is_weekend",
        "datetime_days_since_reference",
    ]
    assert types[6:9] == ["ratio", "difference", "product"]
    assert all(item.priority == 1 for item in report.suggestions[:6])
    assert report.suggestions[5].parameters == (("reference_date", "2024-01-10"),)
    assert report.suggestions[6] == _arithmetic_suggestion("ratio")
    assert any(item.feature_type == "binning_candidate" for item in report.suggestions)
    binning = next(
        item for item in report.suggestions if item.feature_type == "binning_candidate"
    )
    assert binning == FeatureSuggestion(
        name="a__binning_candidate",
        feature_type="binning_candidate",
        source_columns=("a",),
        formula=None,
        parameters=(("strategy", "learned"),),
        reason="numeric_binning_review",
        risk="medium",
        requires_fit=True,
        priority=5,
    )
    group = next(
        item
        for item in report.suggestions
        if item.feature_type == "group_aggregate_candidate"
    )
    assert group.source_columns == ("segment",)
    assert group.formula is None
    assert group.parameters == (("strategy", "fit_on_train_only"),)
    assert group.reason == "categorical_group_aggregate_review"
    assert group.risk == "high"
    assert group.requires_fit is True
    assert group.priority == 6
    target_candidate = next(
        item
        for item in report.suggestions
        if item.feature_type == "target_encoding_candidate"
    )
    assert target_candidate.source_columns == ("segment",)
    assert "target" not in target_candidate.source_columns
    assert target_candidate.parameters == (("strategy", "fit_on_train_only"),)
    assert target_candidate.reason == "target_aware_encoding_review"
    assert target_candidate.risk == "high"
    assert target_candidate.requires_fit is True
    assert target_candidate.priority == 7

    changed = frame.copy()
    changed["target"] = ["different", "values", "different", "values"]
    changed_report = suggest_feature_derivations(
        changed, target="target", reference_date="2024-01-10"
    )
    assert changed_report.suggestions == report.suggestions


def test_name_conflicts_are_filtered_before_available_counts() -> None:
    """Existing derived names do not consume a type budget."""
    frame = pd.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [2, 3, 2, 3],
            "a__div__b": pd.Series(["x", "y", "x", "y"], dtype="string"),
        }
    )
    report = suggest_feature_derivations(frame)
    assert report.available_counts["ratio"] == 0
    assert all(item.name != "a__div__b" for item in report.suggestions)


def test_generated_same_name_candidates_keep_first_before_budget() -> None:
    """Literal special-character names can collide and deterministically dedup."""
    frame = pd.DataFrame(
        {
            "a__div__b": [0, 0, 1, 1, 0, 1],
            "c": [0, 1, 0, 1, 1, 0],
            "a": [1, 0, 0, 1, 0, 1],
            "b__div__c": [1, 1, 0, 0, 1, 0],
        }
    )
    report = suggest_feature_derivations(frame)
    colliding_name = "a__div__b__div__c"
    ratio_names = [
        item.name for item in report.suggestions if item.feature_type == "ratio"
    ]
    assert report.available_counts["ratio"] == 5
    assert ratio_names.count(colliding_name) == 1


@pytest.mark.parametrize("max_suggestions", [49, 50, 51])
def test_type_and_global_budgets(max_suggestions: int) -> None:
    """Per-type budgets precede the exact global 49/50/51 boundary."""
    report = suggest_feature_derivations(
        _budget_frame(),
        target="target",
        reference_date="2024-02-01",
        max_suggestions=max_suggestions,
    )
    assert report.type_budgets == {
        "datetime": 20,
        "ratio": 10,
        "difference": 10,
        "product": 10,
        "binning_candidate": 5,
        "group_aggregate_candidate": 5,
        "target_encoding_candidate": 5,
    }
    assert report.available_counts["datetime"] == 24
    assert report.available_counts["ratio"] == 21
    assert report.available_suggestion_count == 65
    assert len(report.suggestions) == max_suggestions
    assert report.truncated is True
    assert report.truncation_reason == "max_suggestions"


def test_per_type_truncation_does_not_set_global_truncation() -> None:
    """Only max_suggestions affects result-level truncation metadata."""
    report = suggest_feature_derivations(
        _budget_frame(),
        target="target",
        reference_date="2024-02-01",
        max_suggestions=100,
    )
    assert report.available_counts["datetime"] > report.type_budgets["datetime"]
    assert report.truncated is False
    assert report.truncation_reason is None
    assert len(report.suggestions) == report.available_suggestion_count == 65


def test_timezone_aware_datetime_is_unsupported_for_generation() -> None:
    """Timezone-aware sources are skipped rather than converted."""
    frame = pd.DataFrame(
        {"when": pd.date_range("2024-01-01", periods=4, tz="UTC").repeat(2)[:4]}
    )
    report = suggest_feature_derivations(frame)
    assert report.skipped_columns == ("when",)
    assert report.skipped_reasons == {"when": "unsupported_dtype"}


@pytest.mark.parametrize("container", ["bad", b"bad", bytearray(b"bad")])
def test_derive_rejects_string_like_containers(container: object) -> None:
    """String-like containers are not suggestion sequences."""
    with pytest.raises(ValueError, match="suggestions must be a sequence"):
        derive_features(pd.DataFrame({"a": [1]}), container)


def test_derive_rejects_generator_and_invalid_elements() -> None:
    """Generators and non-suggestion members fail at their frozen steps."""
    frame = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="suggestions must be a sequence"):
        derive_features(frame, (item for item in []))
    with pytest.raises(ValueError, match="contain only FeatureSuggestion"):
        derive_features(frame, ["bad"])


def test_derive_rejects_non_boolean_copy() -> None:
    """Copy validation precedes suggestion-container validation."""
    with pytest.raises(ValueError, match="copy must be a boolean"):
        derive_features(pd.DataFrame({"a": [1]}), "bad", copy=1)


def test_derive_validation_precedence() -> None:
    """Requires-fit, supported type, source, canonical, and collision are ordered."""
    frame = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [2, 3],
            "flag": [True, False],
            "a__div__b": [0.0, 0.0],
        }
    )
    base = _arithmetic_suggestion("ratio")

    with pytest.raises(ValueError, match="duplicate suggestion name"):
        derive_features(frame, [base, base])
    with pytest.raises(ValueError, match="requires_fit suggestions"):
        derive_features(
            frame, [replace(base, requires_fit=True, source_columns=("x",))]
        )
    with pytest.raises(ValueError, match="unsupported feature type"):
        derive_features(frame, [replace(base, feature_type="binning_candidate")])
    with pytest.raises(ValueError, match="suggestion fields do not match"):
        derive_features(frame, [replace(base, source_columns=("a",))])
    missing = _arithmetic_suggestion("ratio", "missing", "b")
    with pytest.raises(ValueError, match="source column not found"):
        derive_features(frame, [missing])
    bool_source = _arithmetic_suggestion("ratio", "flag", "b")
    with pytest.raises(
        ValueError, match="arithmetic source columns must be real numeric"
    ):
        derive_features(frame, [replace(bool_source, formula="wrong")])
    with pytest.raises(ValueError, match="suggestion fields do not match"):
        derive_features(frame, [replace(base, formula="wrong")])
    with pytest.raises(ValueError, match="derived feature name already exists"):
        derive_features(frame, [base])


def test_malformed_feature_type_uses_frozen_unsupported_error() -> None:
    """Array-valued external fields cannot leak NumPy ambiguous-truth errors."""
    malformed = replace(
        _arithmetic_suggestion("ratio"),
        feature_type=np.array(["ratio", "product"]),
    )
    with pytest.raises(ValueError, match="unsupported feature type for Task 09"):
        derive_features(pd.DataFrame({"a": [1], "b": [2]}), [malformed])


def test_requires_fit_candidates_and_falsified_suggestion_only_type_are_rejected() -> (
    None
):
    """Suggestion-only types can never be materialized in Task 09."""
    frame = pd.DataFrame({"a": [1, 1, 2]})
    candidate = next(
        item
        for item in suggest_feature_derivations(frame).suggestions
        if item.feature_type == "binning_candidate"
    )
    with pytest.raises(ValueError, match="requires_fit suggestions"):
        derive_features(frame, [candidate])
    with pytest.raises(ValueError, match="unsupported feature type"):
        derive_features(frame, [replace(candidate, requires_fit=False)])


@pytest.mark.parametrize(
    "suggestion",
    [
        _arithmetic_suggestion("ratio"),
        _arithmetic_suggestion("difference"),
        _arithmetic_suggestion("product"),
        _datetime_suggestion("datetime_year"),
        _datetime_suggestion("datetime_month"),
        _datetime_suggestion("datetime_quarter"),
        _datetime_suggestion("datetime_dayofweek"),
        _datetime_suggestion("datetime_is_weekend"),
        _datetime_suggestion("datetime_days_since_reference"),
    ],
)
def test_every_materializable_type_enforces_canonical_fields(
    suggestion: FeatureSuggestion,
) -> None:
    """All nine materializable types reject an altered canonical field."""
    frame = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [2, 3],
            "when": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    with pytest.raises(ValueError, match="suggestion fields do not match"):
        derive_features(frame, [replace(suggestion, risk="high")])


def test_malformed_days_since_parameters_use_canonical_error() -> None:
    """Malformed immutable parameter shapes cannot leak internal exceptions."""
    frame = pd.DataFrame({"when": pd.to_datetime(["2024-01-01"])})
    suggestion = _datetime_suggestion("datetime_days_since_reference")
    malformed = replace(suggestion, parameters=(("reference_date",),))
    with pytest.raises(ValueError, match="suggestion fields do not match"):
        derive_features(frame, [malformed])


def test_arithmetic_materialization_float64_missing_zero_and_nonfinite() -> None:
    """Arithmetic converts before operation and normalizes invalid results."""
    index = pd.Index(["r", "missing", "zero", "inf"], name="row")
    frame = pd.DataFrame(
        {
            "a": pd.Series([2**62, pd.NA, 4, 1], dtype="Int64", index=index),
            "b": pd.Series([4, 2, 0, np.inf], dtype="Float64", index=index),
        },
        index=index,
    )
    suggestions = [
        _arithmetic_suggestion("ratio"),
        _arithmetic_suggestion("difference"),
        _arithmetic_suggestion("product"),
    ]
    result = derive_features(frame, suggestions)
    assert all(str(result.data[item.name].dtype) == "float64" for item in suggestions)
    assert result.data.loc["r", "a__times__b"] == pytest.approx(float(2**62) * 4.0)
    assert result.data.loc["r", "a__minus__b"] == pytest.approx(float(2**62) - 4.0)
    assert np.isnan(result.data.loc["missing", "a__minus__b"])
    assert np.isnan(result.data.loc["zero", "a__div__b"])
    assert np.isnan(result.data.loc["inf", "a__times__b"])
    assert result.data.index.equals(frame.index)


def test_datetime_materialization_values_dtypes_and_missing() -> None:
    """Datetime components and signed days-since use nullable extension dtypes."""
    frame = pd.DataFrame(
        {
            "when": pd.to_datetime(
                ["2024-01-01", "2024-01-06", "2024-01-07", None, "2024-01-11"]
            )
        }
    )
    types = (
        "datetime_year",
        "datetime_month",
        "datetime_quarter",
        "datetime_dayofweek",
        "datetime_is_weekend",
        "datetime_days_since_reference",
    )
    suggestions = [_datetime_suggestion(item) for item in types]
    result = derive_features(frame, suggestions)
    assert str(result.data["when__year"].dtype) == "Int64"
    assert str(result.data["when__month"].dtype) == "Int64"
    assert str(result.data["when__quarter"].dtype) == "Int64"
    assert str(result.data["when__dayofweek"].dtype) == "Int64"
    assert str(result.data["when__is_weekend"].dtype) == "boolean"
    assert str(result.data["when__days_since__2024_01_10"].dtype) == "Int64"
    pdt.assert_series_equal(
        result.data["when__dayofweek"],
        pd.Series([0, 5, 6, pd.NA, 3], dtype="Int64", name="when__dayofweek"),
    )
    pdt.assert_series_equal(
        result.data["when__is_weekend"],
        pd.Series(
            [False, True, True, pd.NA, False],
            dtype="boolean",
            name="when__is_weekend",
        ),
    )
    pdt.assert_series_equal(
        result.data["when__days_since__2024_01_10"],
        pd.Series(
            [9, 4, 3, pd.NA, -1],
            dtype="Int64",
            name="when__days_since__2024_01_10",
        ),
    )


def test_timezone_aware_datetime_materialization_uses_stable_error() -> None:
    """External datetime suggestions cannot strip source timezones."""
    frame = pd.DataFrame({"when": pd.date_range("2024-01-01", periods=2, tz="UTC")})
    with pytest.raises(
        ValueError,
        match="datetime source column must have timezone-naive datetime dtype",
    ):
        derive_features(frame, [_datetime_suggestion("datetime_year")])


@pytest.mark.parametrize("unit", ["s", "ms", "us"])
def test_non_nanosecond_datetime_resolutions_are_stable(unit: str) -> None:
    """Timezone-naive datetime resolutions materialize to the same extension dtypes."""
    values = np.array(
        ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
        dtype=f"datetime64[{unit}]",
    )
    frame = pd.DataFrame({"when": values})
    report = suggest_feature_derivations(frame, reference_date="2024-01-03")
    suggestions = [item for item in report.suggestions if not item.requires_fit]
    result = derive_features(frame, suggestions)
    assert [str(result.data[item.name].dtype) for item in suggestions] == [
        "Int64",
        "Int64",
        "Int64",
        "Int64",
        "boolean",
        "Int64",
    ]


def test_copy_modes_empty_suggestions_and_object_cell_semantics() -> None:
    """Copy identity and pandas object-cell sharing match the frozen contract."""
    nested = {"value": 1}
    frame = pd.DataFrame({"obj": [nested]})
    copied = derive_features(frame, [], copy=True)
    assert copied.data is not frame
    assert copied.data.at[0, "obj"] is frame.at[0, "obj"]
    assert copied.applied_suggestions == ()
    assert copied.skipped_suggestions == ()
    assert copied.skipped_reasons == {}
    in_place = derive_features(frame, (), copy=False)
    assert in_place.data is frame


def test_copy_false_success_and_validation_failure_atomicity() -> None:
    """In-place calls commit only successful, fully validated derivations."""
    frame = pd.DataFrame({"a": [1, 2], "b": [2, 4]})
    before = frame.copy(deep=True)
    with pytest.raises(ValueError, match="source column not found"):
        derive_features(
            frame, [_arithmetic_suggestion("ratio", "missing", "b")], copy=False
        )
    pdt.assert_frame_equal(frame, before)

    result = derive_features(frame, [_arithmetic_suggestion("ratio")], copy=False)
    assert result.data is frame
    assert frame.columns.tolist() == ["a", "b", "a__div__b"]
    assert result.applied_suggestions == ("a__div__b",)
    assert result.skipped_suggestions == ()
    assert result.skipped_reasons == {}


def test_duplicate_index_is_preserved_during_materialization() -> None:
    """Derived Series align without changing duplicate index labels or order."""
    frame = pd.DataFrame(
        {"a": [1, 2, 3], "b": [2, 4, 6]},
        index=pd.Index(["same", "same", "other"], name="row"),
    )
    result = derive_features(frame, [_arithmetic_suggestion("ratio")])
    assert result.data.index.equals(frame.index)
    assert result.data["a__div__b"].tolist() == [0.5, 0.5, 0.5]


def test_unexpected_computation_failure_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later temporary-computation failure leaves both copy modes untouched."""
    frame = pd.DataFrame({"a": [1, 2], "b": [2, 4]})
    before = frame.copy(deep=True)
    suggestions = [
        _arithmetic_suggestion("ratio"),
        _arithmetic_suggestion("difference"),
    ]
    original = features_module._materialize_suggestion
    calls = 0

    def fail_second(value: pd.DataFrame, suggestion: FeatureSuggestion) -> pd.Series:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("computation failed")
        return original(value, suggestion)

    monkeypatch.setattr(features_module, "_materialize_suggestion", fail_second)
    with pytest.raises(RuntimeError, match="computation failed"):
        derive_features(frame, suggestions, copy=False)
    pdt.assert_frame_equal(frame, before)

    calls = 0
    with pytest.raises(RuntimeError, match="computation failed"):
        derive_features(frame, suggestions, copy=True)
    pdt.assert_frame_equal(frame, before)


def test_copy_false_commit_failure_rolls_back_all_new_columns() -> None:
    """An exception during unified commit cannot leave an earlier new column."""

    class FailingCommitFrame(pd.DataFrame):
        _metadata = ["new_column_writes"]

        @property
        def _constructor(self) -> type[pd.DataFrame]:
            return FailingCommitFrame

        def __setitem__(self, key: object, value: object) -> None:
            if key not in self.columns:
                self.new_column_writes += 1
                if self.new_column_writes == 2:
                    raise RuntimeError("commit failed")
            super().__setitem__(key, value)

    frame = FailingCommitFrame({"a": [1, 2], "b": [2, 4]})
    frame.new_column_writes = 0
    before = pd.DataFrame(frame.copy(deep=True))
    suggestions = [
        _arithmetic_suggestion("ratio"),
        _arithmetic_suggestion("difference"),
    ]
    with pytest.raises(RuntimeError, match="commit failed"):
        derive_features(frame, suggestions, copy=False)
    pdt.assert_frame_equal(pd.DataFrame(frame), before)


def test_suggestion_and_derivation_do_not_mutate_by_default() -> None:
    """Suggestion generation and default derivation preserve the caller frame."""
    frame = pd.DataFrame({"a": [1, 1, 2], "b": [2, 3, 3]})
    before = frame.copy(deep=True)
    report = suggest_feature_derivations(frame)
    suggestion = next(
        item for item in report.suggestions if item.feature_type == "ratio"
    )
    result = derive_features(frame, [suggestion])
    pdt.assert_frame_equal(frame, before)
    assert result.data.columns.tolist() == ["a", "b", "a__div__b"]


def test_result_dataclasses_are_frozen() -> None:
    """Task 09 result fields cannot be reassigned."""
    report = suggest_feature_derivations(pd.DataFrame({"x": [1, 1, 2]}))
    with pytest.raises(FrozenInstanceError):
        report.n_rows = 10
