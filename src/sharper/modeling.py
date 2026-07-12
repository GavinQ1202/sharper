"""Split-first classification baselines for structured pandas data."""
# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
    is_timedelta64_dtype,
)
from sklearn.base import ClassifierMixin, RegressorMixin, clone, is_regressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sharper.schema import SchemaReport, infer_schema

Label = str | int | bool


@dataclass(frozen=True)
class TrainingResult:
    """A fitted, train-only classification pipeline and immutable holdout snapshot."""

    task: Literal["classification"]
    target: str
    feature_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    time_column: str | None
    schema: SchemaReport
    pipeline: Pipeline
    estimator: ClassifierMixin
    classes: tuple[str | int | bool, ...]
    train_row_positions: tuple[int, ...]
    test_row_positions: tuple[int, ...]
    X_test: pd.DataFrame
    y_test: tuple[str | int | bool, ...]
    test_size: float
    random_state: int | None
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class RegressionTrainingResult:
    """A fitted, train-only regression pipeline and immutable holdout snapshot."""

    task: Literal["regression"]
    target: str
    feature_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    time_column: str | None
    schema: SchemaReport
    pipeline: Pipeline
    estimator: RegressorMixin
    train_row_positions: tuple[int, ...]
    test_row_positions: tuple[int, ...]
    X_test: pd.DataFrame
    y_test: tuple[float, ...]
    test_size: float
    random_state: int | None
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]


def _label_kind(value: object) -> str | None:
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, (str, np.str_)):
        return "str"
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return "int"
    return None


def _normalise_label(value: object) -> Label:
    kind = _label_kind(value)
    if kind == "bool":
        return bool(value)
    if kind == "int":
        return int(value)
    if kind == "str":
        return str(value)
    raise ValueError(
        "classification target labels must be complete homogeneous scalar values"
    )


def _label_key(value: Label) -> tuple[type[object], object]:
    """Keep bool and int labels distinct even though they compare equal in Python."""
    return (type(value), value)


def _labels_are_unique(values: Sequence[Label]) -> bool:
    return len({_label_key(value) for value in values}) == len(values)


def _same_label_set(left: Sequence[Label], right: Sequence[Label]) -> bool:
    return {_label_key(value) for value in left} == {
        _label_key(value) for value in right
    }


def _validate_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")


def _validate_dataframe_names(df: pd.DataFrame) -> None:
    if (
        not all(isinstance(column, str) for column in df.columns)
        or df.columns.has_duplicates
    ):
        raise ValueError("DataFrame column names must be unique strings")


def _validate_names(value: object, *, name: str, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(
            f"{name} must be a {'non-empty ' if not allow_empty else ''}sequence of unique column names"
        )
    names = tuple(value)
    if (
        (not allow_empty and not names)
        or not all(isinstance(column, str) for column in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(
            f"{name} must be a {'non-empty ' if not allow_empty else ''}sequence of unique column names"
        )
    return names


def _validate_labels(series: pd.Series) -> tuple[Label, ...]:
    labels: list[Label] = []
    kind: str | None = None
    for value in series.tolist():
        missing = pd.isna(value)
        if (
            not isinstance(missing, (bool, np.bool_))
            or missing
            or (current_kind := _label_kind(value)) is None
        ):
            raise ValueError(
                "classification target labels must be complete homogeneous scalar values"
            )
        if kind is None:
            kind = current_kind
        elif current_kind != kind:
            raise ValueError(
                "classification target labels must be complete homogeneous scalar values"
            )
        labels.append(_normalise_label(value))
    ordered: list[Label] = []
    seen: set[tuple[type[object], object]] = set()
    for value in labels:
        key = _label_key(value)
        if key not in seen:
            seen.add(key)
            ordered.append(value)
    if len(ordered) < 2 or any(labels.count(value) < 2 for value in ordered):
        raise ValueError(
            "classification target must contain at least two classes with two rows each"
        )
    return tuple(labels)


def _is_categorical_dtype(dtype: object) -> bool:
    return (
        is_object_dtype(dtype)
        or is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
        or is_bool_dtype(dtype)
    )


def _feature_kind(series: pd.Series) -> str | None:
    dtype = series.dtype
    if (
        is_numeric_dtype(dtype)
        and not is_bool_dtype(dtype)
        and not is_complex_dtype(dtype)
    ):
        return "numeric"
    if _is_categorical_dtype(dtype):
        return "categorical"
    return None


def _validate_fitted_classes(estimator: object, classes: tuple[Label, ...]) -> None:
    raw_classes = getattr(estimator, "classes_", None)
    if raw_classes is None:
        raise ValueError("classifier estimator has invalid output")
    try:
        normalised = tuple(_normalise_label(value) for value in list(raw_classes))
    except (TypeError, ValueError) as error:
        raise ValueError("classifier estimator has invalid output") from error
    if (
        not normalised
        or not _labels_are_unique(normalised)
        or not _same_label_set(normalised, classes)
    ):
        raise ValueError("classifier estimator has invalid output")


def _validate_regression_target(series: pd.Series) -> tuple[float, ...]:
    """Validate a finite, non-constant real-numeric regression target."""
    if (
        not is_numeric_dtype(series.dtype)
        or is_bool_dtype(series.dtype)
        or is_complex_dtype(series.dtype)
    ):
        raise ValueError(
            "regression target must be complete finite real numeric values"
        )
    try:
        values = series.astype("float64").to_numpy(copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "regression target must be complete finite real numeric values"
        ) from error
    if not np.isfinite(values).all():
        raise ValueError(
            "regression target must be complete finite real numeric values"
        )
    if len(np.unique(values)) < 2:
        raise ValueError("regression target must contain at least two distinct values")
    return tuple(float(value) for value in values)


def _regression_result_warnings(
    df: pd.DataFrame, *, random_state: int | None, custom_estimator: bool
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Construct the frozen warning and limitation vocabularies in order."""
    warning_values: list[str] = []
    if df.index.has_duplicates:
        warning_values.append("duplicate_index")
    try:
        if bool(df.duplicated().any()):
            warning_values.append("duplicate_rows")
    except Exception:
        pass
    limitation_values: list[str] = []
    if random_state is None:
        limitation_values.append("random_state_none")
    if custom_estimator:
        warning_values.append("custom_estimator_random_state_not_managed")
        limitation_values.append("custom_estimator_determinism_not_guaranteed")
    return tuple(warning_values), tuple(limitation_values)


def train_classifier(
    df: pd.DataFrame,
    target: str,
    *,
    features: Sequence[str] | None = None,
    exclude_columns: Sequence[str] = (),
    time_column: str | None = None,
    estimator: ClassifierMixin | None = None,
    test_size: float = 0.20,
    random_state: int | None = 42,
) -> TrainingResult:
    """Fit a leakage-safe classification baseline using only training rows.

    Parameters are intentionally narrow: ``exclude_columns`` must name known
    post-outcome, future, or entity-risk fields, and ``time_column`` rejects the
    random-holdout path. Missing numeric/categorical feature values are imputed
    inside the train-only pipeline. The input frame is never changed.

    Returns a fitted ``TrainingResult`` with only the holdout feature table and
    labels retained for evaluation. Raises ``ValueError`` for invalid inputs,
    unsupported temporal data, estimator failures, or malformed fitted output.

    Examples
    --------
    >>> import pandas as pd
    >>> from sharper import train_classifier
    >>> result = train_classifier(pd.DataFrame({"x": [0, 1, 2, 3], "y": [0, 0, 1, 1]}), "y", test_size=0.5)
    >>> result.task
    'classification'
    """
    _validate_dataframe(df)
    if not isinstance(target, str):
        raise ValueError("target must be a column name string")
    if time_column is not None and not isinstance(time_column, str):
        raise ValueError("time_column must be a column name string or None")
    if (
        isinstance(test_size, (bool, np.bool_))
        or not isinstance(test_size, Real)
        or not np.isfinite(float(test_size))
        or not 0.0 < float(test_size) < 1.0
    ):
        raise ValueError(
            "test_size must permit a stratified split strictly between 0 and 1"
        )
    if random_state is not None and (
        isinstance(random_state, (bool, np.bool_))
        or not isinstance(random_state, Integral)
        or random_state < 0
    ):
        raise ValueError("random_state must be a non-negative integer or None")
    _validate_dataframe_names(df)
    if target not in df.columns:
        raise ValueError(f"target column not found: {target!r}")

    exclusions = _validate_names(
        exclude_columns, name="exclude_columns", allow_empty=True
    )
    if target in exclusions:
        raise ValueError("target must not appear in exclude_columns")
    for column in exclusions:
        if column not in df.columns:
            raise ValueError(f"excluded column not found: {column!r}")
    if time_column is not None:
        if time_column not in df.columns:
            raise ValueError(f"time column not found: {time_column!r}")
        if time_column == target:
            raise ValueError("time_column must be a column name string or None")
        raise ValueError("time-ordered classification is not supported")
    if any(
        is_datetime64_any_dtype(series.dtype) or is_timedelta64_dtype(series.dtype)
        for _, series in df.items()
    ):
        raise ValueError("time-ordered classification is not supported")
    requested_features = (
        None
        if features is None
        else _validate_names(features, name="features", allow_empty=False)
    )
    if requested_features is not None:
        if target in requested_features:
            raise ValueError("target must not appear in features")
        for column in requested_features:
            if column not in df.columns:
                raise ValueError(f"feature column not found: {column!r}")
        if set(requested_features).intersection(exclusions):
            raise ValueError("features and exclude_columns must not overlap")

    labels = _validate_labels(df[target])

    model_columns = [
        column for column in df.columns if column not in {target, *exclusions}
    ]
    candidates = (
        model_columns if requested_features is None else list(requested_features)
    )
    for column in candidates:
        if is_complex_dtype(df[column].dtype):
            raise ValueError(f"unsupported model feature dtype: {column!r}")

    if estimator is not None:
        if (
            getattr(estimator, "_estimator_type", None) != "classifier"
            or not callable(getattr(estimator, "fit", None))
            or not callable(getattr(estimator, "predict", None))
        ):
            raise ValueError("estimator must be a classifier")
        try:
            base_estimator = clone(estimator)
        except Exception as error:
            raise ValueError("classifier estimator could not be cloned") from error
    else:
        base_estimator = LogisticRegression(max_iter=1000, random_state=random_state)

    positions = np.arange(len(df), dtype=int)
    try:
        train_positions, test_positions = train_test_split(
            positions,
            test_size=float(test_size),
            random_state=random_state,
            stratify=list(labels),
        )
    except ValueError as error:
        raise ValueError(
            "test_size must permit a stratified split strictly between 0 and 1"
        ) from error

    X = df.loc[:, model_columns]
    X_train = X.iloc[train_positions].copy(deep=True)
    X_test = X.iloc[test_positions].copy(deep=True)
    schema = infer_schema(X_train)
    schema_by_name = {column.name: column for column in schema.columns}
    selected: list[str] = []
    numeric: list[str] = []
    categorical: list[str] = []
    for column in model_columns:
        if column not in candidates:
            continue
        kind = _feature_kind(X_train[column])
        if kind is None:
            if requested_features is not None:
                raise ValueError(f"unsupported model feature dtype: {column!r}")
            continue
        report = schema_by_name[column]
        if report.is_id_like:
            if requested_features is not None:
                raise ValueError(f"unsupported model feature dtype: {column!r}")
            continue
        if report.is_constant:
            continue
        selected.append(column)
        (numeric if kind == "numeric" else categorical).append(column)
    if not selected:
        raise ValueError("no eligible model features")

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    pipeline = Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers, remainder="drop")),
            ("estimator", base_estimator),
        ]
    )
    y_train = [labels[position] for position in train_positions]
    try:
        pipeline.fit(X_train.loc[:, selected], y_train)
    except Exception as error:
        raise ValueError("classifier estimator fit failed") from error
    fitted_estimator = pipeline.named_steps["estimator"]
    _validate_fitted_classes(fitted_estimator, tuple(dict.fromkeys(labels)))

    warnings: list[str] = []
    if df.index.has_duplicates:
        warnings.append("duplicate_index")
    try:
        if bool(df.duplicated().any()):
            warnings.append("duplicate_rows")
    except Exception:
        pass
    limitations: list[str] = []
    if random_state is None:
        limitations.append("random_state_none")
    if estimator is not None:
        warnings.append("custom_estimator_random_state_not_managed")
        limitations.append("custom_estimator_determinism_not_guaranteed")
    return TrainingResult(
        task="classification",
        target=target,
        feature_columns=tuple(selected),
        excluded_columns=tuple(exclusions),
        time_column=None,
        schema=schema,
        pipeline=pipeline,
        estimator=fitted_estimator,
        classes=tuple(dict.fromkeys(labels)),
        train_row_positions=tuple(int(position) for position in train_positions),
        test_row_positions=tuple(int(position) for position in test_positions),
        X_test=X_test.loc[:, selected].copy(deep=True),
        y_test=tuple(labels[position] for position in test_positions),
        test_size=float(test_size),
        random_state=None if random_state is None else int(random_state),
        warnings=tuple(warnings),
        limitations=tuple(limitations),
    )


def train_regressor(
    df: pd.DataFrame,
    target: str,
    *,
    features: Sequence[str] | None = None,
    exclude_columns: Sequence[str] = (),
    time_column: str | None = None,
    estimator: RegressorMixin | None = None,
    test_size: float = 0.20,
    random_state: int | None = 42,
) -> RegressionTrainingResult:
    """Fit a leakage-safe Ridge baseline using only training rows.

    Parameters
    ----------
    df
        Source DataFrame. It is never modified.
    target
        Present finite real-numeric target column.
    features, exclude_columns, time_column
        Optional feature selection and known risk columns. Time risk is rejected
        because this API supports random holdout only.
    estimator, test_size, random_state
        Optional regressor, holdout fraction, and split/default-estimator seed.

    Returns
    -------
    RegressionTrainingResult
        A fitted train-only pipeline and copied holdout snapshot.

    Raises
    ------
    ValueError
        If inputs, target, split, feature eligibility, or estimator output are
        outside the frozen Task 12 contract.

    Notes
    -----
    Schema inference, ID-like detection, final feature selection, preprocessing,
    and fitting use training rows only. Missing feature values are handled inside
    the train-only pipeline.

    Examples
    --------
    >>> import pandas as pd
    >>> from sharper import train_regressor
    >>> result = train_regressor(pd.DataFrame({"x": [0, 1, 2, 3], "y": [0., 1., 2., 3.]}), "y", test_size=0.5)
    >>> result.task
    'regression'
    """
    _validate_dataframe(df)
    if not isinstance(target, str):
        raise ValueError("target must be a column name string")
    if time_column is not None and not isinstance(time_column, str):
        raise ValueError("time_column must be a column name string or None")
    if (
        isinstance(test_size, (bool, np.bool_))
        or not isinstance(test_size, Real)
        or not np.isfinite(float(test_size))
        or not 0.0 < float(test_size) < 1.0
    ):
        raise ValueError("test_size must be strictly between 0 and 1")
    if random_state is not None and (
        isinstance(random_state, (bool, np.bool_))
        or not isinstance(random_state, Integral)
        or random_state < 0
    ):
        raise ValueError("random_state must be a non-negative integer or None")
    _validate_dataframe_names(df)
    if target not in df.columns:
        raise ValueError(f"target column not found: {target!r}")

    exclusions = _validate_names(
        exclude_columns, name="exclude_columns", allow_empty=True
    )
    if target in exclusions:
        raise ValueError("target must not appear in exclude_columns")
    for column in exclusions:
        if column not in df.columns:
            raise ValueError(f"excluded column not found: {column!r}")
    if time_column is not None:
        if time_column not in df.columns:
            raise ValueError(f"time column not found: {time_column!r}")
        if time_column == target:
            raise ValueError("time_column must be a column name string or None")
        raise ValueError("time-ordered regression is not supported")
    if any(
        is_datetime64_any_dtype(series.dtype) or is_timedelta64_dtype(series.dtype)
        for _, series in df.items()
    ):
        raise ValueError("time-ordered regression is not supported")
    requested_features = (
        None
        if features is None
        else _validate_names(features, name="features", allow_empty=False)
    )
    if requested_features is not None:
        if target in requested_features:
            raise ValueError("target must not appear in features")
        for column in requested_features:
            if column not in df.columns:
                raise ValueError(f"feature column not found: {column!r}")
        if set(requested_features).intersection(exclusions):
            raise ValueError("features and exclude_columns must not overlap")

    values = _validate_regression_target(df[target])
    model_columns = [
        column for column in df.columns if column not in {target, *exclusions}
    ]
    candidates = (
        model_columns if requested_features is None else list(requested_features)
    )
    for column in candidates:
        if is_complex_dtype(df[column].dtype):
            raise ValueError(f"unsupported model feature dtype: {column!r}")

    if estimator is not None:
        if (
            getattr(estimator, "_estimator_type", None) != "regressor"
            or not callable(getattr(estimator, "fit", None))
            or not callable(getattr(estimator, "predict", None))
        ):
            raise ValueError("estimator must be a regressor")
        try:
            base_estimator = clone(estimator)
        except Exception as error:
            raise ValueError("regressor estimator could not be cloned") from error
    else:
        base_estimator = Ridge(random_state=random_state)

    positions = np.arange(len(df), dtype=int)
    try:
        train_positions, test_positions = train_test_split(
            positions,
            test_size=float(test_size),
            random_state=random_state,
        )
    except ValueError as error:
        raise ValueError(
            "test_size must produce at least two train and holdout rows"
        ) from error
    if len(train_positions) < 2 or len(test_positions) < 2:
        raise ValueError("test_size must produce at least two train and holdout rows")

    X = df.loc[:, model_columns]
    X_train = X.iloc[train_positions].copy(deep=True)
    X_test = X.iloc[test_positions].copy(deep=True)
    schema = infer_schema(X_train)
    schema_by_name = {column.name: column for column in schema.columns}
    selected: list[str] = []
    numeric: list[str] = []
    categorical: list[str] = []
    for column in model_columns:
        if column not in candidates:
            continue
        kind = _feature_kind(X_train[column])
        if kind is None:
            if requested_features is not None:
                raise ValueError(f"unsupported model feature dtype: {column!r}")
            continue
        report = schema_by_name[column]
        if report.is_id_like:
            if requested_features is not None:
                raise ValueError(f"unsupported model feature dtype: {column!r}")
            continue
        if report.is_constant:
            continue
        selected.append(column)
        (numeric if kind == "numeric" else categorical).append(column)
    if not selected:
        raise ValueError("no eligible model features")

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    pipeline = Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers, remainder="drop")),
            ("estimator", base_estimator),
        ]
    )
    y_train = [values[position] for position in train_positions]
    try:
        pipeline.fit(X_train.loc[:, selected], y_train)
    except Exception as error:
        raise ValueError("regressor estimator fit failed") from error
    fitted_estimator = pipeline.named_steps["estimator"]
    if not (
        is_regressor(fitted_estimator)
        or getattr(fitted_estimator, "_estimator_type", None) == "regressor"
    ) or not callable(getattr(fitted_estimator, "predict", None)):
        raise ValueError("regressor estimator has invalid output")
    warning_values, limitation_values = _regression_result_warnings(
        df, random_state=random_state, custom_estimator=estimator is not None
    )
    return RegressionTrainingResult(
        task="regression",
        target=target,
        feature_columns=tuple(selected),
        excluded_columns=tuple(exclusions),
        time_column=None,
        schema=schema,
        pipeline=pipeline,
        estimator=fitted_estimator,
        train_row_positions=tuple(int(position) for position in train_positions),
        test_row_positions=tuple(int(position) for position in test_positions),
        X_test=X_test.loc[:, selected].copy(deep=True),
        y_test=tuple(values[position] for position in test_positions),
        test_size=float(test_size),
        random_state=None if random_state is None else int(random_state),
        warnings=warning_values,
        limitations=limitation_values,
    )
