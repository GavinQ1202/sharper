"""Deterministic feature suggestions and safe stateless derivation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from itertools import combinations

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_datetime64_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
)

from sharper.schema import SchemaReport, infer_schema

_TYPE_BUDGETS = {
    "datetime": 20,
    "ratio": 10,
    "difference": 10,
    "product": 10,
    "binning_candidate": 5,
    "group_aggregate_candidate": 5,
    "target_encoding_candidate": 5,
}

_MATERIALIZABLE_TYPES = (
    "ratio",
    "difference",
    "product",
    "datetime_year",
    "datetime_month",
    "datetime_quarter",
    "datetime_dayofweek",
    "datetime_is_weekend",
    "datetime_days_since_reference",
)
_ARITHMETIC_TYPES = ("ratio", "difference", "product")
_DATETIME_TYPES = (
    "datetime_year",
    "datetime_month",
    "datetime_quarter",
    "datetime_dayofweek",
    "datetime_is_weekend",
    "datetime_days_since_reference",
)
_ISO_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class FeatureSuggestion:
    """Describe one deterministic candidate feature.

    Attributes
    ----------
    name
        Stable proposed output-column name.
    feature_type
        Feature type from the closed Task 09 vocabulary.
    source_columns
        Source names in formula order.
    formula
        Canonical human-readable formula, or ``None`` for review candidates.
    parameters
        Ordered immutable string parameters.
    reason
        Reason code from the closed Task 09 vocabulary.
    risk
        ``"low"``, ``"medium"``, or ``"high"``.
    requires_fit
        Whether safe materialization would require learned state.
    priority
        Fixed feature-type priority; smaller numbers come first.
    """

    name: str
    feature_type: str
    source_columns: tuple[str, ...]
    formula: str | None
    parameters: tuple[tuple[str, str], ...]
    reason: str
    risk: str
    requires_fit: bool
    priority: int


@dataclass(frozen=True)
class FeatureSuggestionReport:
    """Contain bounded deterministic feature suggestions and column states.

    The three column-state tuples form a complete partition in DataFrame order.
    Per-type counts are measured before per-type budgets; result-level truncation
    reports only the global budget.
    """

    n_rows: int
    requested_target: str | None
    requested_exclusions: tuple[str, ...]
    reference_date: str | None
    eligible_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    skipped_columns: tuple[str, ...]
    skipped_reasons: dict[str, str]
    max_suggestions: int
    type_budgets: dict[str, int]
    available_counts: dict[str, int]
    available_suggestion_count: int
    truncated: bool
    truncation_reason: str | None
    suggestions: tuple[FeatureSuggestion, ...]


@dataclass(frozen=True)
class FeatureDerivationResult:
    """Contain a derived DataFrame and applied-suggestion metadata.

    Task 09 applies every valid suggestion or raises before returning. Therefore
    the skipped fields are always empty in successful results.
    """

    data: pd.DataFrame
    applied_suggestions: tuple[str, ...]
    skipped_suggestions: tuple[str, ...]
    skipped_reasons: dict[str, str]
    copy: bool


def suggest_feature_derivations(
    df: pd.DataFrame,
    *,
    schema: SchemaReport | None = None,
    target: str | None = None,
    exclude_columns: Sequence[str] = (),
    reference_date: str | date | datetime | pd.Timestamp | None = None,
    max_suggestions: int = 50,
) -> FeatureSuggestionReport:
    """Suggest bounded deterministic feature derivations without mutation.

    Parameters
    ----------
    df
        DataFrame with unique string column names.
    schema
        Matching schema report. When omitted, ``infer_schema(df)`` is called.
    target
        Optional target column. It is excluded from all feature sources; its
        values are never read for candidate ranking or statistics.
    exclude_columns
        Existing columns to exclude explicitly, in caller order.
    reference_date
        Optional strict ISO date, date, naive datetime, or naive Timestamp used
        only to suggest deterministic days-since features.
    max_suggestions
        Positive, non-boolean global suggestion budget.

    Returns
    -------
    FeatureSuggestionReport
        Column partition, budget metadata, and ordered suggestions.

    Raises
    ------
    ValueError
        If the DataFrame boundary, schema, target, exclusions, reference date,
        or budget violates the frozen Task 09 contract.

    Notes
    -----
    Missing and constant columns are skipped. Object/string values are not parsed
    as datetimes. This function never modifies ``df`` and performs no fitted or
    target-aware computation.

    Examples
    --------
    >>> import pandas as pd
    >>> from sharper import suggest_feature_derivations
    >>> frame = pd.DataFrame({"a": [1, 1, 2], "b": [2, 3, 3]})
    >>> report = suggest_feature_derivations(frame)
    >>> report.suggestions[0].feature_type
    'ratio'
    """
    _validate_dataframe(df)
    resolved_schema = _resolve_schema(df, schema)
    _validate_target(df, target)
    requested_exclusions = _validate_exclusions(df, exclude_columns)
    normalized_reference = _normalize_reference_date(reference_date)
    _validate_max_suggestions(max_suggestions)

    eligible, excluded, skipped, skipped_reasons = _partition_columns(
        df,
        schema=resolved_schema,
        target=target,
        exclusions=requested_exclusions,
    )
    candidate_groups = _generate_candidates(
        df,
        eligible=eligible,
        target=target,
        reference_date=normalized_reference,
    )
    filtered_groups = _filter_candidate_conflicts(df, candidate_groups)
    available_counts = {key: len(filtered_groups[key]) for key in _TYPE_BUDGETS}
    budgeted_groups = {
        key: filtered_groups[key][: _TYPE_BUDGETS[key]] for key in _TYPE_BUDGETS
    }
    combined = tuple(
        suggestion for key in _TYPE_BUDGETS for suggestion in budgeted_groups[key]
    )
    available_suggestion_count = len(combined)
    truncated = available_suggestion_count > max_suggestions
    suggestions = combined[:max_suggestions]

    return FeatureSuggestionReport(
        n_rows=len(df),
        requested_target=target,
        requested_exclusions=requested_exclusions,
        reference_date=normalized_reference,
        eligible_columns=eligible,
        excluded_columns=excluded,
        skipped_columns=skipped,
        skipped_reasons=skipped_reasons,
        max_suggestions=max_suggestions,
        type_budgets=dict(_TYPE_BUDGETS),
        available_counts=available_counts,
        available_suggestion_count=available_suggestion_count,
        truncated=truncated,
        truncation_reason="max_suggestions" if truncated else None,
        suggestions=suggestions,
    )


def derive_features(
    df: pd.DataFrame,
    suggestions: Sequence[FeatureSuggestion],
    *,
    copy: bool = True,
) -> FeatureDerivationResult:
    """Materialize approved stateless feature suggestions atomically.

    Parameters
    ----------
    df
        DataFrame with unique string column names.
    suggestions
        A list, tuple, or other non-string Sequence of approved suggestions.
    copy
        If ``True``, return a pandas ``deep=True`` copy. If ``False``, commit all
        new columns to ``df`` only after validation and temporary computation
        succeed.

    Returns
    -------
    FeatureDerivationResult
        Derived data, applied names, empty skipped metadata, and copy mode.

    Raises
    ------
    ValueError
        If validation fails, a suggestion requires fitting, a source is invalid,
        canonical fields differ, or an output name already exists.

    Notes
    -----
    Arithmetic missing and non-finite results become ``NaN``. Datetime missing
    values become ``pd.NA``. With ``copy=False``, successful calls modify ``df``;
    failures do not leave partial columns. Pandas ``deep=True`` does not recursively
    clone mutable Python objects stored in object-dtype cells.

    Examples
    --------
    >>> import pandas as pd
    >>> from sharper import derive_features, suggest_feature_derivations
    >>> frame = pd.DataFrame({"a": [1, 1, 2], "b": [2, 3, 3]})
    >>> suggestion = suggest_feature_derivations(frame).suggestions[0]
    >>> derive_features(frame, [suggestion]).data.columns[-1]
    'a__div__b'
    """
    _validate_dataframe(df)
    if not isinstance(copy, bool):
        raise ValueError("copy must be a boolean")
    if not isinstance(suggestions, Sequence) or isinstance(
        suggestions, (str, bytes, bytearray)
    ):
        raise ValueError("suggestions must be a sequence of FeatureSuggestion")

    suggestion_tuple = tuple(suggestions)
    if not all(isinstance(item, FeatureSuggestion) for item in suggestion_tuple):
        raise ValueError("suggestions must contain only FeatureSuggestion values")
    _validate_unique_suggestion_names(suggestion_tuple)
    _validate_derivation_suggestions(df, suggestion_tuple)

    computed = [
        (suggestion.name, _materialize_suggestion(df, suggestion))
        for suggestion in suggestion_tuple
    ]
    result_data = df.copy(deep=True) if copy else df
    _commit_computed_features(result_data, computed)

    return FeatureDerivationResult(
        data=result_data,
        applied_suggestions=tuple(item.name for item in suggestion_tuple),
        skipped_suggestions=(),
        skipped_reasons={},
        copy=copy,
    )


def _validate_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if not all(isinstance(name, str) for name in df.columns):
        raise ValueError("DataFrame column names must all be strings")
    if df.columns.has_duplicates:
        raise ValueError("duplicate DataFrame column names are not supported")


def _resolve_schema(df: pd.DataFrame, schema: SchemaReport | None) -> SchemaReport:
    if schema is None:
        return infer_schema(df)
    if not isinstance(schema, SchemaReport):
        raise ValueError("schema must be a SchemaReport")
    schema_names = [column.name for column in schema.columns]
    if (
        schema.n_rows != len(df)
        or schema.n_columns != len(df.columns)
        or schema_names != list(df.columns)
    ):
        raise ValueError("schema does not match DataFrame")
    return schema


def _validate_target(df: pd.DataFrame, target: str | None) -> None:
    if target is not None and not isinstance(target, str):
        raise ValueError("target must be a string")
    if target is not None and target not in df.columns:
        raise ValueError("target column not found")


def _validate_exclusions(
    df: pd.DataFrame, exclude_columns: Sequence[str]
) -> tuple[str, ...]:
    if isinstance(exclude_columns, (str, bytes, bytearray)) or not isinstance(
        exclude_columns, Sequence
    ):
        raise ValueError("columns must contain only strings")
    exclusions = tuple(exclude_columns)
    if not all(isinstance(column, str) for column in exclusions):
        raise ValueError("columns must contain only strings")
    if len(set(exclusions)) != len(exclusions):
        raise ValueError("duplicate column parameter")
    if any(column not in df.columns for column in exclusions):
        raise ValueError("column not found")
    return exclusions


def _normalize_reference_date(
    value: str | date | datetime | pd.Timestamp | None,
) -> str | None:
    if value is None:
        return None
    if value is pd.NaT:
        raise ValueError("reference_date must be a valid date")
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            raise ValueError("reference_date must be a valid date")
        if value.tzinfo is not None:
            raise ValueError("reference_date must be timezone-naive")
        normalized = value.normalize().date()
    elif isinstance(value, datetime):
        if value.tzinfo is not None and value.utcoffset() is not None:
            raise ValueError("reference_date must be timezone-naive")
        normalized = value.date()
    elif isinstance(value, date):
        normalized = value
    elif isinstance(value, str):
        if _ISO_DATE_PATTERN.fullmatch(value) is None:
            raise ValueError("reference_date must be a valid date")
        try:
            normalized = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError("reference_date must be a valid date") from error
    else:
        raise ValueError("reference_date must be a valid date")
    return normalized.isoformat()


def _validate_max_suggestions(max_suggestions: int) -> None:
    if (
        isinstance(max_suggestions, bool)
        or not isinstance(max_suggestions, int)
        or max_suggestions < 1
    ):
        raise ValueError("max_suggestions must be a positive integer")


def _partition_columns(
    df: pd.DataFrame,
    *,
    schema: SchemaReport,
    target: str | None,
    exclusions: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], dict[str, str]]:
    schemas = {column.name: column for column in schema.columns}
    exclusion_set = set(exclusions)
    eligible: list[str] = []
    excluded: list[str] = []
    skipped: list[str] = []
    reasons: dict[str, str] = {}
    retained: list[str] = []

    for name in df.columns:
        column_schema = schemas[name]
        reason: str | None = None
        state = "eligible"
        if name == target:
            reason = "target_column"
            state = "excluded"
        elif name in exclusion_set:
            reason = "explicitly_excluded"
            state = "excluded"
        elif column_schema.logical_type == "identifier" or column_schema.is_id_like:
            reason = "identifier_like"
            state = "skipped"
        elif column_schema.missing_count == len(df):
            reason = "all_missing"
            state = "skipped"
        elif column_schema.is_constant:
            reason = "constant"
            state = "skipped"
        elif not _is_supported_source(df[name]):
            reason = "unsupported_dtype"
            state = "skipped"
        elif any(
            str(df[prior].dtype) == str(df[name].dtype) and df[prior].equals(df[name])
            for prior in retained
        ):
            reason = "duplicate_content"
            state = "skipped"

        if state == "excluded":
            excluded.append(name)
            reasons[name] = reason or ""
        elif state == "skipped":
            skipped.append(name)
            reasons[name] = reason or ""
        else:
            eligible.append(name)
            retained.append(name)

    return tuple(eligible), tuple(excluded), tuple(skipped), reasons


def _is_real_numeric(series: pd.Series) -> bool:
    dtype = series.dtype
    return (
        (is_integer_dtype(dtype) or is_float_dtype(dtype))
        and not is_bool_dtype(dtype)
        and not is_complex_dtype(dtype)
    )


def _is_datetime(series: pd.Series) -> bool:
    return is_datetime64_dtype(series.dtype) and not isinstance(
        series.dtype, pd.DatetimeTZDtype
    )


def _is_categorical(series: pd.Series) -> bool:
    dtype = series.dtype
    return (
        is_bool_dtype(dtype)
        or is_object_dtype(dtype)
        or is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def _is_supported_source(series: pd.Series) -> bool:
    return _is_real_numeric(series) or _is_datetime(series) or _is_categorical(series)


def _generate_candidates(
    df: pd.DataFrame,
    *,
    eligible: tuple[str, ...],
    target: str | None,
    reference_date: str | None,
) -> dict[str, list[FeatureSuggestion]]:
    numeric = [name for name in eligible if _is_real_numeric(df[name])]
    datetimes = [name for name in eligible if _is_datetime(df[name])]
    categoricals = [name for name in eligible if _is_categorical(df[name])]
    groups = {key: [] for key in _TYPE_BUDGETS}

    for name in datetimes:
        groups["datetime"].extend(_datetime_suggestions(name, reference_date))
    pairs = list(combinations(numeric, 2))
    groups["ratio"] = [_arithmetic_suggestion("ratio", *pair) for pair in pairs]
    groups["difference"] = [
        _arithmetic_suggestion("difference", *pair) for pair in pairs
    ]
    groups["product"] = [_arithmetic_suggestion("product", *pair) for pair in pairs]
    groups["binning_candidate"] = [
        FeatureSuggestion(
            name=f"{name}__binning_candidate",
            feature_type="binning_candidate",
            source_columns=(name,),
            formula=None,
            parameters=(("strategy", "learned"),),
            reason="numeric_binning_review",
            risk="medium",
            requires_fit=True,
            priority=5,
        )
        for name in numeric
    ]
    groups["group_aggregate_candidate"] = [
        FeatureSuggestion(
            name=f"{name}__group_aggregate_candidate",
            feature_type="group_aggregate_candidate",
            source_columns=(name,),
            formula=None,
            parameters=(("strategy", "fit_on_train_only"),),
            reason="categorical_group_aggregate_review",
            risk="high",
            requires_fit=True,
            priority=6,
        )
        for name in categoricals
    ]
    if target is not None:
        groups["target_encoding_candidate"] = [
            FeatureSuggestion(
                name=f"{name}__target_encoding_candidate",
                feature_type="target_encoding_candidate",
                source_columns=(name,),
                formula=None,
                parameters=(("strategy", "fit_on_train_only"),),
                reason="target_aware_encoding_review",
                risk="high",
                requires_fit=True,
                priority=7,
            )
            for name in categoricals
        ]
    return groups


def _datetime_suggestions(
    column: str, reference_date: str | None
) -> list[FeatureSuggestion]:
    specifications = (
        ("datetime_year", "year", f"{column}.dt.year", "datetime_component", "low"),
        (
            "datetime_month",
            "month",
            f"{column}.dt.month",
            "datetime_component",
            "low",
        ),
        (
            "datetime_quarter",
            "quarter",
            f"{column}.dt.quarter",
            "datetime_component",
            "low",
        ),
        (
            "datetime_dayofweek",
            "dayofweek",
            f"{column}.dt.dayofweek",
            "datetime_component",
            "low",
        ),
        (
            "datetime_is_weekend",
            "is_weekend",
            f"{column}.dt.is_weekend",
            "datetime_component",
            "low",
        ),
    )
    suggestions = [
        FeatureSuggestion(
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
        for feature_type, suffix, formula, reason, risk in specifications
    ]
    if reference_date is not None:
        suggestions.append(
            FeatureSuggestion(
                name=f"{column}__days_since__{reference_date.replace('-', '_')}",
                feature_type="datetime_days_since_reference",
                source_columns=(column,),
                formula=f"{reference_date} - {column}",
                parameters=(("reference_date", reference_date),),
                reason="explicit_reference_date",
                risk="medium",
                requires_fit=False,
                priority=1,
            )
        )
    return suggestions


def _arithmetic_suggestion(
    feature_type: str, left: str, right: str
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


def _filter_candidate_conflicts(
    df: pd.DataFrame,
    groups: dict[str, list[FeatureSuggestion]],
) -> dict[str, list[FeatureSuggestion]]:
    existing_names = set(df.columns)
    seen_names: set[str] = set()
    seen_identities: set[tuple[object, ...]] = set()
    filtered = {key: [] for key in _TYPE_BUDGETS}
    for key in _TYPE_BUDGETS:
        for suggestion in groups[key]:
            identity = (
                suggestion.name,
                suggestion.feature_type,
                suggestion.source_columns,
                suggestion.parameters,
            )
            if suggestion.name in existing_names:
                continue
            if suggestion.name in seen_names or identity in seen_identities:
                continue
            seen_names.add(suggestion.name)
            seen_identities.add(identity)
            filtered[key].append(suggestion)
    return filtered


def _validate_unique_suggestion_names(
    suggestions: tuple[FeatureSuggestion, ...],
) -> None:
    seen: list[str] = []
    for suggestion in suggestions:
        if isinstance(suggestion.name, str) and suggestion.name in seen:
            raise ValueError("duplicate suggestion name")
        if isinstance(suggestion.name, str):
            seen.append(suggestion.name)


def _validate_derivation_suggestions(
    df: pd.DataFrame, suggestions: tuple[FeatureSuggestion, ...]
) -> None:
    for suggestion in suggestions:
        if suggestion.requires_fit is True:
            raise ValueError(
                "requires_fit suggestions cannot be materialized in Task 09"
            )
    for suggestion in suggestions:
        if (
            not isinstance(suggestion.feature_type, str)
            or suggestion.feature_type not in _MATERIALIZABLE_TYPES
        ):
            raise ValueError("unsupported feature type for Task 09 materialization")
    for suggestion in suggestions:
        _validate_source_columns(df, suggestion)
    for suggestion in suggestions:
        if not _matches_canonical_contract(suggestion):
            raise ValueError("suggestion fields do not match the Task 09 contract")
    for suggestion in suggestions:
        if suggestion.name in df.columns:
            raise ValueError("derived feature name already exists")


def _validate_source_columns(df: pd.DataFrame, suggestion: FeatureSuggestion) -> None:
    sources = suggestion.source_columns
    if not isinstance(sources, tuple):
        raise ValueError("suggestion fields do not match the Task 09 contract")
    if suggestion.feature_type in _ARITHMETIC_TYPES:
        valid_count = len(sources) == 2
        if valid_count and all(isinstance(name, str) for name in sources):
            valid_count = sources[0] != sources[1]
    else:
        valid_count = len(sources) == 1
    if not valid_count:
        raise ValueError("suggestion fields do not match the Task 09 contract")
    if any(not isinstance(name, str) or name not in df.columns for name in sources):
        raise ValueError("source column not found")
    if suggestion.feature_type in _ARITHMETIC_TYPES:
        if not all(_is_real_numeric(df[name]) for name in sources):
            raise ValueError("arithmetic source columns must be real numeric")
    elif not _is_datetime(df[sources[0]]):
        raise ValueError(
            "datetime source column must have timezone-naive datetime dtype"
        )


def _matches_canonical_contract(suggestion: FeatureSuggestion) -> bool:
    sources = suggestion.source_columns
    feature_type = suggestion.feature_type
    if feature_type in _ARITHMETIC_TYPES:
        expected = _arithmetic_suggestion(feature_type, sources[0], sources[1])
        return _canonical_fields_equal(suggestion, expected)
    column = sources[0]
    if feature_type == "datetime_days_since_reference":
        if (
            not isinstance(suggestion.parameters, tuple)
            or len(suggestion.parameters) != 1
            or not isinstance(suggestion.parameters[0], tuple)
            or len(suggestion.parameters[0]) != 2
            or suggestion.parameters[0][0] != "reference_date"
        ):
            return False
        reference_date = suggestion.parameters[0][1]
        if not isinstance(reference_date, str):
            return False
        try:
            normalized = _normalize_reference_date(reference_date)
        except ValueError:
            return False
        return _canonical_fields_equal(
            suggestion, _datetime_suggestions(column, normalized)[-1]
        )
    expected_by_type = {
        item.feature_type: item
        for item in _datetime_suggestions(column, reference_date=None)
    }
    return _canonical_fields_equal(suggestion, expected_by_type[feature_type])


def _canonical_fields_equal(
    actual: FeatureSuggestion, expected: FeatureSuggestion
) -> bool:
    parameters_are_valid = isinstance(actual.parameters, tuple) and all(
        isinstance(parameter, tuple)
        and len(parameter) == 2
        and all(isinstance(value, str) for value in parameter)
        for parameter in actual.parameters
    )
    return (
        isinstance(actual.name, str)
        and actual.name == expected.name
        and isinstance(actual.feature_type, str)
        and actual.feature_type == expected.feature_type
        and isinstance(actual.source_columns, tuple)
        and all(isinstance(source, str) for source in actual.source_columns)
        and actual.source_columns == expected.source_columns
        and (actual.formula is None or isinstance(actual.formula, str))
        and actual.formula == expected.formula
        and parameters_are_valid
        and actual.parameters == expected.parameters
        and isinstance(actual.reason, str)
        and actual.reason == expected.reason
        and isinstance(actual.risk, str)
        and actual.risk == expected.risk
        and isinstance(actual.requires_fit, bool)
        and actual.requires_fit is expected.requires_fit
        and isinstance(actual.priority, int)
        and not isinstance(actual.priority, bool)
        and actual.priority == expected.priority
    )


def _commit_computed_features(
    target: pd.DataFrame, computed: list[tuple[str, pd.Series]]
) -> None:
    try:
        for name, values in computed:
            target[name] = values
    except BaseException:
        for name, _ in reversed(computed):
            if name in target.columns:
                pd.DataFrame.__delitem__(target, name)
        raise


def _materialize_suggestion(
    df: pd.DataFrame, suggestion: FeatureSuggestion
) -> pd.Series:
    feature_type = suggestion.feature_type
    source = suggestion.source_columns
    if feature_type in _ARITHMETIC_TYPES:
        left = df[source[0]].to_numpy(dtype=np.float64, na_value=np.nan)
        right = df[source[1]].to_numpy(dtype=np.float64, na_value=np.nan)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            if feature_type == "ratio":
                result = left / right
            elif feature_type == "difference":
                result = left - right
            else:
                result = left * right
        result[~np.isfinite(result)] = np.nan
        return pd.Series(result, index=df.index, dtype="float64")

    values = df[source[0]]
    if feature_type == "datetime_year":
        return values.dt.year.astype("Int64")
    if feature_type == "datetime_month":
        return values.dt.month.astype("Int64")
    if feature_type == "datetime_quarter":
        return values.dt.quarter.astype("Int64")
    if feature_type == "datetime_dayofweek":
        return values.dt.dayofweek.astype("Int64")
    if feature_type == "datetime_is_weekend":
        dayofweek = values.dt.dayofweek.astype("Int64")
        return (dayofweek >= 5).astype("boolean")
    reference_date = pd.Timestamp(dict(suggestion.parameters)["reference_date"])
    return (reference_date.normalize() - values.dt.normalize()).dt.days.astype("Int64")
