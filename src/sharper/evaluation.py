"""Holdout-only classification evaluation for Sharper training results."""
# ruff: noqa: E501

from __future__ import annotations

import warnings
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from sklearn.base import is_regressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.metrics import (
    confusion_matrix as sklearn_confusion_matrix,
)
from sklearn.metrics import (
    roc_curve as sklearn_roc_curve,
)
from sklearn.pipeline import Pipeline

from sharper.modeling import (
    Label,
    RegressionTrainingResult,
    TrainingResult,
    _label_key,
    _labels_are_unique,
    _normalise_label,
    _same_label_set,
)
from sharper.schema import SchemaReport


@dataclass(frozen=True)
class ClassificationEvaluation:
    """Frozen holdout predictions, metrics, and optional binary ROC detail."""

    task: Literal["classification"]
    target: str
    holdout_positions: tuple[int, ...]
    classes: tuple[str | int | bool, ...]
    y_true: tuple[str | int | bool, ...]
    y_pred: tuple[str | int | bool, ...]
    score_kind: Literal["predict_proba", "decision_function"] | None
    positive_label: str | int | bool | None
    scores: tuple[float, ...] | None
    roc_curve: tuple[tuple[float, float, float], ...]
    metrics: tuple[tuple[str, float], ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    roc_auc: float | None
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class RegressionEvaluation:
    """Frozen holdout predictions, residuals, and regression metrics."""

    task: Literal["regression"]
    target: str
    holdout_positions: tuple[int, ...]
    predictions: pd.DataFrame
    metrics: tuple[tuple[str, float], ...]
    limitations: tuple[str, ...]


_TRAINING_SCHEMA_ERROR = "training result has invalid schema"
_EVALUATION_SCHEMA_ERROR = "classification evaluation result has invalid schema"
_REGRESSION_TRAINING_SCHEMA_ERROR = "regression training result has invalid schema"
_REGRESSION_EVALUATION_SCHEMA_ERROR = "regression evaluation result has invalid schema"


def _is_label(value: object) -> bool:
    try:
        _normalise_label(value)
    except (TypeError, ValueError):
        return False
    return True


def _labels(values: object) -> tuple[Label, ...] | None:
    if not isinstance(values, tuple):
        return None
    try:
        normalised = tuple(_normalise_label(value) for value in values)
    except (TypeError, ValueError):
        return None
    return normalised


def _validate_training_result(result: TrainingResult) -> None:
    """Validate all observable fields before a holdout estimator call."""
    message = _TRAINING_SCHEMA_ERROR
    if result.task != "classification" or not isinstance(result.target, str):
        raise ValueError(message)
    if (
        not isinstance(result.feature_columns, tuple)
        or not result.feature_columns
        or not all(isinstance(column, str) for column in result.feature_columns)
        or len(set(result.feature_columns)) != len(result.feature_columns)
        or not isinstance(result.excluded_columns, tuple)
        or not all(isinstance(column, str) for column in result.excluded_columns)
        or len(set(result.excluded_columns)) != len(result.excluded_columns)
        or result.target in result.excluded_columns
        or set(result.feature_columns).intersection(result.excluded_columns)
        or result.time_column is not None
        or not isinstance(result.schema, SchemaReport)
        or not isinstance(result.pipeline, Pipeline)
        or result.pipeline.named_steps.get("estimator") is not result.estimator
    ):
        raise ValueError(message)
    classes = _labels(result.classes)
    if classes is None or len(classes) < 2 or not _labels_are_unique(classes):
        raise ValueError(message)
    positions = (result.train_row_positions, result.test_row_positions)
    if any(
        not isinstance(part, tuple)
        or not all(
            isinstance(value, Integral)
            and not isinstance(value, (bool, np.bool_))
            and value >= 0
            for value in part
        )
        for part in positions
    ):
        raise ValueError(message)
    train_positions = tuple(int(value) for value in result.train_row_positions)
    test_positions = tuple(int(value) for value in result.test_row_positions)
    if (
        not train_positions
        or not test_positions
        or len(set(train_positions)) != len(train_positions)
        or len(set(test_positions)) != len(test_positions)
        or set(train_positions).intersection(test_positions)
        or set(train_positions).union(test_positions)
        != set(range(len(train_positions) + len(test_positions)))
    ):
        raise ValueError(message)
    if (
        not isinstance(result.X_test, pd.DataFrame)
        or tuple(result.X_test.columns) != result.feature_columns
        or len(result.X_test) != len(test_positions)
    ):
        raise ValueError(message)
    y_test = _labels(result.y_test)
    if (
        y_test is None
        or len(y_test) != len(test_positions)
        or any(
            _label_key(value) not in {_label_key(label) for label in classes}
            for value in y_test
        )
    ):
        raise ValueError(message)
    if (
        not isinstance(result.test_size, float)
        or not np.isfinite(result.test_size)
        or not 0.0 < result.test_size < 1.0
        or (
            result.random_state is not None
            and (
                not isinstance(result.random_state, int)
                or isinstance(result.random_state, bool)
                or result.random_state < 0
            )
        )
        or not isinstance(result.warnings, tuple)
        or not isinstance(result.limitations, tuple)
        or any(
            value
            not in {
                "duplicate_index",
                "duplicate_rows",
                "custom_estimator_random_state_not_managed",
            }
            for value in result.warnings
        )
        or any(
            value
            not in {"random_state_none", "custom_estimator_determinism_not_guaranteed"}
            for value in result.limitations
        )
        or len(set(result.warnings)) != len(result.warnings)
        or len(set(result.limitations)) != len(result.limitations)
    ):
        raise ValueError(message)
    if result.random_state is None and "random_state_none" not in result.limitations:
        raise ValueError(message)
    if result.random_state is not None and "random_state_none" in result.limitations:
        raise ValueError(message)
    if "custom_estimator_random_state_not_managed" in result.warnings:
        if "custom_estimator_determinism_not_guaranteed" not in result.limitations:
            raise ValueError(message)
    elif "custom_estimator_determinism_not_guaranteed" in result.limitations:
        raise ValueError(message)
    if tuple(column.name for column in result.schema.columns) == ():
        raise ValueError(message)
    try:
        estimator_classes = tuple(
            _normalise_label(value) for value in list(result.estimator.classes_)
        )
    except (AttributeError, TypeError, ValueError):
        raise ValueError(message) from None
    if (
        not estimator_classes
        or not _labels_are_unique(estimator_classes)
        or not _same_label_set(estimator_classes, classes)
    ):
        raise ValueError(message)


def _estimator_interface(result: TrainingResult) -> None:
    if not callable(getattr(result.pipeline, "predict", None)):
        raise ValueError(
            "classifier estimator does not support required prediction interface"
        )


def _normalise_predictions(
    values: object, classes: tuple[Label, ...], n_test: int
) -> tuple[Label, ...]:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != n_test:
        raise ValueError("classifier estimator has invalid output")
    try:
        predicted = tuple(_normalise_label(value) for value in array.tolist())
    except (TypeError, ValueError) as error:
        raise ValueError("classifier estimator has invalid output") from error
    class_keys = {_label_key(value) for value in classes}
    if any(_label_key(value) not in class_keys for value in predicted):
        raise ValueError("classifier estimator has invalid output")
    return predicted


def _binary_scores(
    result: TrainingResult, classes: tuple[Label, ...], n_test: int
) -> tuple[str | None, tuple[float, ...] | None]:
    """Get one frozen binary score per row, preferring probabilities."""
    estimator_classes = tuple(
        _normalise_label(value) for value in list(result.estimator.classes_)
    )
    positive = classes[1]
    probability = getattr(result.pipeline, "predict_proba", None)
    if callable(probability):
        try:
            values = np.asarray(probability(result.X_test))
        except Exception as error:
            raise ValueError(
                "classifier estimator probability prediction failed"
            ) from error
        try:
            valid = (
                values.ndim == 2
                and values.shape == (n_test, len(estimator_classes))
                and np.isfinite(values).all()
                and not (values < 0.0).any()
                and not (values > 1.0).any()
                and np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
            )
        except Exception as error:
            raise ValueError("classifier estimator has invalid output") from error
        if not valid:
            raise ValueError("classifier estimator has invalid output")
        index = next(
            (
                i
                for i, value in enumerate(estimator_classes)
                if _label_key(value) == _label_key(positive)
            ),
            None,
        )
        if index is None:
            raise ValueError("classifier estimator has invalid output")
        return "predict_proba", tuple(float(value) for value in values[:, index])
    decision = getattr(result.pipeline, "decision_function", None)
    if not callable(decision):
        return None, None
    try:
        values = np.asarray(decision(result.X_test))
    except Exception as error:
        raise ValueError("classifier estimator score prediction failed") from error
    try:
        valid = values.ndim == 1 and len(values) == n_test and np.isfinite(values).all()
    except Exception as error:
        raise ValueError("classifier estimator has invalid output") from error
    if not valid:
        raise ValueError("classifier estimator has invalid output")
    if _label_key(positive) == _label_key(estimator_classes[1]):
        directed = values
    elif _label_key(positive) == _label_key(estimator_classes[0]):
        directed = -values
    else:
        raise ValueError("classifier estimator has invalid output")
    return "decision_function", tuple(float(value) for value in directed)


def evaluate_classifier(result: TrainingResult) -> ClassificationEvaluation:
    """Evaluate a valid fitted classifier once on its stored holdout data.

    The function never fits, splits, clones, or reads training observations. It
    returns accuracy, balanced accuracy, macro F1, a confusion matrix, and ROC
    detail only for binary score-capable estimators. Raises ``ValueError`` for a
    malformed result or malformed/failed estimator output.
    """
    if not isinstance(result, TrainingResult):
        raise ValueError("result must be a TrainingResult")
    _validate_training_result(result)
    _estimator_interface(result)
    classes = tuple(_normalise_label(value) for value in result.classes)
    try:
        raw_predictions = result.pipeline.predict(result.X_test)
    except Exception as error:
        raise ValueError("classifier estimator prediction failed") from error
    predicted = _normalise_predictions(raw_predictions, classes, len(result.y_test))
    y_true = tuple(_normalise_label(value) for value in result.y_test)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = (
            ("accuracy", float(accuracy_score(y_true, predicted))),
            ("balanced_accuracy", float(balanced_accuracy_score(y_true, predicted))),
            (
                "f1_macro",
                float(
                    f1_score(
                        y_true,
                        predicted,
                        labels=list(classes),
                        average="macro",
                        zero_division=0,
                    )
                ),
            ),
        )
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for _, value in metrics):
        raise ValueError("classifier estimator has invalid output")
    matrix = tuple(
        tuple(int(value) for value in row)
        for row in sklearn_confusion_matrix(
            y_true, predicted, labels=list(classes)
        ).tolist()
    )
    limitations: tuple[str, ...]
    score_kind: Literal["predict_proba", "decision_function"] | None = None
    scores: tuple[float, ...] | None = None
    positive_label: Label | None = None
    roc_detail: tuple[tuple[float, float, float], ...] = ()
    roc_auc: float | None = None
    if len(classes) != 2:
        limitations = ("multiclass_roc_unavailable",)
    else:
        kind, scores = _binary_scores(result, classes, len(y_true))
        if kind is None or scores is None:
            limitations = ("score_unavailable",)
        else:
            score_kind = kind  # type: ignore[assignment]
            positive_label = classes[1]
            false_positive, true_positive, thresholds = sklearn_roc_curve(
                y_true, scores, pos_label=positive_label, drop_intermediate=False
            )
            roc_detail = tuple(
                (float(fpr), float(tpr), float(threshold))
                for fpr, tpr, threshold in zip(
                    false_positive, true_positive, thresholds, strict=True
                )
            )
            roc_auc = float(
                roc_auc_score(
                    tuple(value == positive_label for value in y_true), scores
                )
            )
            if not np.isfinite(roc_auc) or not 0.0 <= roc_auc <= 1.0:
                raise ValueError("classifier estimator has invalid output")
            limitations = ()
    evaluation = ClassificationEvaluation(
        task="classification",
        target=result.target,
        holdout_positions=tuple(result.test_row_positions),
        classes=classes,
        y_true=y_true,
        y_pred=predicted,
        score_kind=score_kind,
        positive_label=positive_label,
        scores=scores,
        roc_curve=roc_detail,
        metrics=metrics,
        confusion_matrix=matrix,
        roc_auc=roc_auc,
        limitations=limitations,
    )
    _validate_classification_evaluation(evaluation)
    return evaluation


def _close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=1e-12, equal_nan=False))


def _validated_metrics(metrics: object) -> tuple[float, float, float] | None:
    """Validate frozen metric tuple structure before any unpacking occurs."""
    expected_names = ("accuracy", "balanced_accuracy", "f1_macro")
    if not isinstance(metrics, tuple) or len(metrics) != len(expected_names):
        return None
    values: list[float] = []
    for item, name in zip(metrics, expected_names, strict=True):
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or item[0] != name
            or isinstance(item[1], (bool, np.bool_))
            or not isinstance(item[1], Real)
            or not np.isfinite(float(item[1]))
        ):
            return None
        values.append(float(item[1]))
    return tuple(values)  # type: ignore[return-value]


def _validate_classification_evaluation_detail(
    result: ClassificationEvaluation,
) -> None:
    """Validate deterministic evaluation detail without consulting an estimator."""
    message = _EVALUATION_SCHEMA_ERROR
    if result.task != "classification" or not isinstance(result.target, str):
        raise ValueError(message)
    classes = _labels(result.classes)
    y_true = _labels(result.y_true)
    y_pred = _labels(result.y_pred)
    if (
        classes is None
        or len(classes) < 2
        or not _labels_are_unique(classes)
        or y_true is None
        or y_pred is None
    ):
        raise ValueError(message)
    if (
        not isinstance(result.holdout_positions, tuple)
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in result.holdout_positions
        )
        or len(set(result.holdout_positions)) != len(result.holdout_positions)
        or not result.holdout_positions
        or len(y_true) != len(result.holdout_positions)
        or len(y_pred) != len(result.holdout_positions)
    ):
        raise ValueError(message)
    class_keys = {_label_key(value) for value in classes}
    if any(_label_key(value) not in class_keys for value in (*y_true, *y_pred)):
        raise ValueError(message)
    metric_values = _validated_metrics(result.metrics)
    if (
        metric_values is None
        or not isinstance(result.confusion_matrix, tuple)
        or len(result.confusion_matrix) != len(classes)
        or any(
            not isinstance(row, tuple) or len(row) != len(classes)
            for row in result.confusion_matrix
        )
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for row in result.confusion_matrix
            for value in row
        )
        or sum(sum(row) for row in result.confusion_matrix) != len(y_true)
    ):
        raise ValueError(message)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected_metrics = (
            float(accuracy_score(y_true, y_pred)),
            float(balanced_accuracy_score(y_true, y_pred)),
            float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=list(classes),
                    average="macro",
                    zero_division=0,
                )
            ),
        )
    for value, expected in zip(metric_values, expected_metrics, strict=True):
        if not 0.0 <= value <= 1.0 or not _close(value, expected):
            raise ValueError(message)
    expected_matrix = tuple(
        tuple(int(value) for value in row)
        for row in sklearn_confusion_matrix(
            y_true, y_pred, labels=list(classes)
        ).tolist()
    )
    if result.confusion_matrix != expected_matrix:
        raise ValueError(message)
    if not isinstance(result.limitations, tuple) or len(set(result.limitations)) != len(
        result.limitations
    ):
        raise ValueError(message)
    if len(classes) != 2:
        if (
            result.score_kind is not None
            or result.positive_label is not None
            or result.scores is not None
            or result.roc_curve != ()
            or result.roc_auc is not None
            or result.limitations != ("multiclass_roc_unavailable",)
        ):
            raise ValueError(message)
        return
    if result.score_kind is None:
        if (
            result.positive_label is not None
            or result.scores is not None
            or result.roc_curve != ()
            or result.roc_auc is not None
            or result.limitations != ("score_unavailable",)
        ):
            raise ValueError(message)
        return
    if (
        result.score_kind not in {"predict_proba", "decision_function"}
        or result.positive_label is None
        or _label_key(_normalise_label(result.positive_label)) != _label_key(classes[1])
        or not isinstance(result.scores, tuple)
        or len(result.scores) != len(y_true)
        or not all(
            isinstance(value, float) and np.isfinite(value) for value in result.scores
        )
        or not isinstance(result.roc_curve, tuple)
        or not result.roc_curve
        or result.roc_auc is None
        or not isinstance(result.roc_auc, float)
        or not np.isfinite(result.roc_auc)
        or not 0.0 <= result.roc_auc <= 1.0
        or result.limitations != ()
    ):
        raise ValueError(message)
    false_positive, true_positive, thresholds = sklearn_roc_curve(
        y_true, result.scores, pos_label=classes[1], drop_intermediate=False
    )
    expected_curve = tuple(
        (float(fpr), float(tpr), float(threshold))
        for fpr, tpr, threshold in zip(
            false_positive, true_positive, thresholds, strict=True
        )
    )
    if len(result.roc_curve) != len(expected_curve):
        raise ValueError(message)
    for index, (actual, expected) in enumerate(
        zip(result.roc_curve, expected_curve, strict=True)
    ):
        if (
            not isinstance(actual, tuple)
            or len(actual) != 3
            or not all(isinstance(value, float) for value in actual)
            or not 0.0 <= actual[0] <= 1.0
            or not 0.0 <= actual[1] <= 1.0
            or (not np.isfinite(actual[2]) and index != 0)
            or not _close(actual[0], expected[0])
            or not _close(actual[1], expected[1])
            or (np.isfinite(expected[2]) and not _close(actual[2], expected[2]))
            or (np.isinf(expected[2]) and not (np.isposinf(actual[2])))
        ):
            raise ValueError(message)
    expected_auc = float(
        roc_auc_score(tuple(value == classes[1] for value in y_true), result.scores)
    )
    if not _close(result.roc_auc, expected_auc):
        raise ValueError(message)


def _validate_classification_evaluation(result: ClassificationEvaluation) -> None:
    """Map every malformed frozen evaluation member to the stable public error."""
    try:
        _validate_classification_evaluation_detail(result)
    except Exception as error:
        if isinstance(error, ValueError) and str(error) == _EVALUATION_SCHEMA_ERROR:
            raise
        raise ValueError(_EVALUATION_SCHEMA_ERROR) from error


def _regression_positions(values: object) -> tuple[int, ...] | None:
    if not isinstance(values, tuple) or not values:
        return None
    if not all(
        isinstance(value, Integral)
        and not isinstance(value, (bool, np.bool_))
        and value >= 0
        for value in values
    ):
        return None
    positions = tuple(int(value) for value in values)
    return positions if len(set(positions)) == len(positions) else None


def _regression_floats(values: object) -> tuple[float, ...] | None:
    if not isinstance(values, tuple):
        return None
    converted: list[float] = []
    for value in values:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, float)
            or not np.isfinite(value)
        ):
            return None
        converted.append(value)
    return tuple(converted)


def _ordered_subset(values: object, vocabulary: tuple[str, ...]) -> bool:
    if not isinstance(values, tuple) or not all(
        isinstance(value, str) for value in values
    ):
        return False
    return tuple(value for value in vocabulary if value in values) == values


def _validate_regression_training_result_detail(
    result: RegressionTrainingResult,
) -> None:
    """Validate observable regression training state before prediction."""
    message = _REGRESSION_TRAINING_SCHEMA_ERROR
    if result.task != "regression" or not isinstance(result.target, str):
        raise ValueError(message)
    if (
        not isinstance(result.feature_columns, tuple)
        or not result.feature_columns
        or not all(isinstance(column, str) for column in result.feature_columns)
        or len(set(result.feature_columns)) != len(result.feature_columns)
        or not isinstance(result.excluded_columns, tuple)
        or not all(isinstance(column, str) for column in result.excluded_columns)
        or len(set(result.excluded_columns)) != len(result.excluded_columns)
        or result.target in result.excluded_columns
        or set(result.feature_columns).intersection(result.excluded_columns)
        or result.time_column is not None
        or not isinstance(result.schema, SchemaReport)
        or not isinstance(result.pipeline, Pipeline)
        or result.pipeline.named_steps.get("estimator") is not result.estimator
        or not (
            is_regressor(result.estimator)
            or getattr(result.estimator, "_estimator_type", None) == "regressor"
        )
        or not callable(getattr(result.estimator, "predict", None))
    ):
        raise ValueError(message)
    train_positions = _regression_positions(result.train_row_positions)
    test_positions = _regression_positions(result.test_row_positions)
    if (
        train_positions is None
        or test_positions is None
        or len(train_positions) < 2
        or len(test_positions) < 2
        or set(train_positions).intersection(test_positions)
        or set(train_positions).union(test_positions)
        != set(range(len(train_positions) + len(test_positions)))
    ):
        raise ValueError(message)
    y_test = _regression_floats(result.y_test)
    if (
        not isinstance(result.X_test, pd.DataFrame)
        or tuple(result.X_test.columns) != result.feature_columns
        or len(result.X_test) != len(test_positions)
        or y_test is None
        or len(y_test) != len(test_positions)
        or not isinstance(result.test_size, float)
        or not np.isfinite(result.test_size)
        or not 0.0 < result.test_size < 1.0
        or (
            result.random_state is not None
            and (
                not isinstance(result.random_state, int)
                or isinstance(result.random_state, bool)
                or result.random_state < 0
            )
        )
    ):
        raise ValueError(message)
    warning_vocabulary = (
        "duplicate_index",
        "duplicate_rows",
        "custom_estimator_random_state_not_managed",
    )
    limitation_vocabulary = (
        "random_state_none",
        "custom_estimator_determinism_not_guaranteed",
    )
    if (
        not _ordered_subset(result.warnings, warning_vocabulary)
        or not _ordered_subset(result.limitations, limitation_vocabulary)
        or len(set(result.warnings)) != len(result.warnings)
        or len(set(result.limitations)) != len(result.limitations)
        or (result.random_state is None) != ("random_state_none" in result.limitations)
        or ("custom_estimator_random_state_not_managed" in result.warnings)
        != ("custom_estimator_determinism_not_guaranteed" in result.limitations)
    ):
        raise ValueError(message)


def _validate_regression_training_result(result: RegressionTrainingResult) -> None:
    """Map malformed regression training state to its stable public error."""
    try:
        _validate_regression_training_result_detail(result)
    except Exception as error:
        if (
            isinstance(error, ValueError)
            and str(error) == _REGRESSION_TRAINING_SCHEMA_ERROR
        ):
            raise
        raise ValueError(_REGRESSION_TRAINING_SCHEMA_ERROR) from error


def _normalise_regression_predictions(values: object, n_test: int) -> tuple[float, ...]:
    """Accept only the frozen real-numeric prediction output contract."""
    try:
        if isinstance(values, pd.Series):
            dtype = values.dtype
            if (
                not is_numeric_dtype(dtype)
                or is_bool_dtype(dtype)
                or is_complex_dtype(dtype)
                or is_object_dtype(dtype)
            ):
                raise ValueError
            array = values.to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            if isinstance(values, (list, tuple)):
                if any(
                    isinstance(value, (bool, np.bool_))
                    or isinstance(value, (list, tuple, np.ndarray, pd.Series))
                    for value in values
                ):
                    raise ValueError
            array = np.asarray(values)
            if (
                array.dtype == object
                or not np.issubdtype(array.dtype, np.number)
                or np.issubdtype(array.dtype, np.bool_)
                or np.issubdtype(array.dtype, np.complexfloating)
            ):
                raise ValueError
        if array.ndim != 1 or len(array) != n_test or not np.isfinite(array).all():
            raise ValueError
    except Exception as error:
        raise ValueError("regressor estimator has invalid output") from error
    return tuple(float(value) for value in array)


def _regression_metric_values(
    actual: tuple[float, ...], predicted: tuple[float, ...]
) -> tuple[float, float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mae = float(mean_absolute_error(actual, predicted))
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        r2 = float(r2_score(actual, predicted, force_finite=True))
    if not all(np.isfinite(value) for value in (mae, rmse, r2)):
        raise ValueError("regression evaluation result has invalid schema")
    return mae, rmse, r2


def _regression_metrics(values: object) -> tuple[float, float, float] | None:
    names = ("mae", "rmse", "r2")
    if not isinstance(values, tuple) or len(values) != len(names):
        return None
    result: list[float] = []
    for item, name in zip(values, names, strict=True):
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or item[0] != name
            or type(item[1]) is not float
            or not np.isfinite(item[1])
        ):
            return None
        result.append(item[1])
    return tuple(result)  # type: ignore[return-value]


def _validate_regression_evaluation_detail(result: RegressionEvaluation) -> None:
    """Validate frozen table detail and reconstruct its metrics without an estimator."""
    message = _REGRESSION_EVALUATION_SCHEMA_ERROR
    positions = _regression_positions(result.holdout_positions)
    if (
        result.task != "regression"
        or not isinstance(result.target, str)
        or positions is None
        or len(positions) < 2
        or not isinstance(result.predictions, pd.DataFrame)
        or tuple(result.predictions.columns)
        != ("row_position", "actual", "predicted", "residual")
        or tuple(str(dtype) for dtype in result.predictions.dtypes)
        != ("int64", "float64", "float64", "float64")
        or len(result.predictions) != len(positions)
        or result.predictions["row_position"].tolist() != list(positions)
        or not isinstance(result.limitations, tuple)
        or result.limitations != ()
    ):
        raise ValueError(message)
    actual = tuple(float(value) for value in result.predictions["actual"].tolist())
    predicted = tuple(
        float(value) for value in result.predictions["predicted"].tolist()
    )
    residuals = tuple(float(value) for value in result.predictions["residual"].tolist())
    if not all(np.isfinite(value) for value in (*actual, *predicted, *residuals)):
        raise ValueError(message)
    if any(
        not _close(residual, value - prediction)
        for residual, value, prediction in zip(
            residuals, actual, predicted, strict=True
        )
    ):
        raise ValueError(message)
    metrics = _regression_metrics(result.metrics)
    if metrics is None:
        raise ValueError(message)
    expected = _regression_metric_values(actual, predicted)
    if (
        metrics[0] < 0.0
        or metrics[1] < 0.0
        or any(
            not _close(value, expected_value)
            for value, expected_value in zip(metrics, expected, strict=True)
        )
    ):
        raise ValueError(message)


def _validate_regression_evaluation(result: RegressionEvaluation) -> None:
    """Map malformed regression evaluation state to its stable public error."""
    try:
        _validate_regression_evaluation_detail(result)
    except Exception as error:
        if (
            isinstance(error, ValueError)
            and str(error) == _REGRESSION_EVALUATION_SCHEMA_ERROR
        ):
            raise
        raise ValueError(_REGRESSION_EVALUATION_SCHEMA_ERROR) from error


def evaluate_regressor(result: RegressionTrainingResult) -> RegressionEvaluation:
    """Evaluate a fitted regressor exactly once on its stored holdout data.

    The function neither fits, splits, clones, nor reads raw training data. It
    returns finite MAE, RMSE, and force-finite R² with a copied prediction table.
    """
    if not isinstance(result, RegressionTrainingResult):
        raise ValueError("result must be a RegressionTrainingResult")
    _validate_regression_training_result(result)
    try:
        raw_predictions = result.pipeline.predict(result.X_test)
    except Exception as error:
        raise ValueError("regressor estimator prediction failed") from error
    predicted = _normalise_regression_predictions(raw_predictions, len(result.y_test))
    actual = tuple(result.y_test)
    residuals = tuple(
        value - prediction for value, prediction in zip(actual, predicted, strict=True)
    )
    mae, rmse, r2 = _regression_metric_values(actual, predicted)
    predictions = pd.DataFrame(
        {
            "row_position": pd.Series(result.test_row_positions, dtype="int64"),
            "actual": pd.Series(actual, dtype="float64"),
            "predicted": pd.Series(predicted, dtype="float64"),
            "residual": pd.Series(residuals, dtype="float64"),
        }
    )
    evaluation = RegressionEvaluation(
        task="regression",
        target=result.target,
        holdout_positions=tuple(result.test_row_positions),
        predictions=predictions,
        metrics=(("mae", mae), ("rmse", rmse), ("r2", r2)),
        limitations=(),
    )
    _validate_regression_evaluation(evaluation)
    return evaluation


def evaluate_model(
    result: TrainingResult | RegressionTrainingResult,
) -> ClassificationEvaluation | RegressionEvaluation:
    """Dispatch a frozen classification or regression holdout evaluator once."""
    if isinstance(result, TrainingResult):
        if result.task != "classification":
            raise ValueError(
                "evaluate_model supports only classification or regression"
            )
        return evaluate_classifier(result)
    if isinstance(result, RegressionTrainingResult):
        if result.task != "regression":
            raise ValueError(
                "evaluate_model supports only classification or regression"
            )
        return evaluate_regressor(result)
    raise ValueError("result must be a TrainingResult or RegressionTrainingResult")


# Task 15 private prediction-validation and metric primitives.  These helpers
# deliberately remain below the v0.1 public API and never fit or split data.
_RISK_METRIC_ORDER = (
    "roc_auc",
    "average_precision",
    "normalized_gini",
    "ks_statistic",
    "brier_score",
    "log_loss",
    "expected_calibration_error",
)
_RISK_METRIC_COLUMNS = (
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
_RISK_GAINS_COLUMNS = (
    "scope",
    "fold_id",
    "requested_fraction",
    "target_count",
    "boundary_score",
    "selected_n",
    "actual_fraction",
    "total_positive_n",
    "selected_positive_n",
    "event_rate",
    "event_rate_status",
    "event_rate_reason",
    "capture",
    "capture_status",
    "capture_reason",
    "lift",
    "lift_status",
    "lift_reason",
)
_RISK_CALIBRATION_COLUMNS = (
    "scope",
    "fold_id",
    "bin_id",
    "lower_bound",
    "upper_bound",
    "upper_inclusive",
    "n_rows",
    "mean_predicted_probability",
    "observed_event_rate",
    "absolute_gap",
    "weighted_gap",
    "status",
    "reason",
)
_RISK_THRESHOLD_COLUMNS = (
    "scope",
    "fold_id",
    "threshold_kind",
    "threshold",
    "tp",
    "fp",
    "tn",
    "fn",
    "sensitivity",
    "sensitivity_status",
    "sensitivity_reason",
    "specificity",
    "specificity_status",
    "specificity_reason",
    "precision",
    "precision_status",
    "precision_reason",
    "negative_predictive_value",
    "negative_predictive_value_status",
    "negative_predictive_value_reason",
    "f1",
    "f1_status",
    "f1_reason",
    "accuracy",
    "accuracy_status",
    "accuracy_reason",
    "predicted_positive_rate",
    "predicted_positive_rate_status",
    "predicted_positive_rate_reason",
)


@dataclass(frozen=True)
class _BinaryRiskEvaluationTables:
    metrics: pd.DataFrame
    gains: pd.DataFrame
    calibration: pd.DataFrame
    threshold_analysis: pd.DataFrame


def _risk_frame(
    rows: list[dict[str, object]],
    columns: tuple[str, ...],
    *,
    integers: tuple[str, ...] = (),
    floats: tuple[str, ...] = (),
    booleans: tuple[str, ...] = (),
) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=columns)
    for column in integers:
        frame[column] = pd.array(frame[column], dtype="Int64")
    for column in floats:
        frame[column] = pd.array(frame[column], dtype="Float64")
    for column in booleans:
        frame[column] = pd.array(frame[column], dtype="boolean")
    return frame


def _risk_float_array(values: object, *, probability: bool) -> np.ndarray:
    message = (
        "event probabilities must be finite values in [0, 1]"
        if probability
        else "ranking scores must be finite real values with explicit direction"
    )
    try:
        raw = np.asarray(values, dtype="object")
        if raw.ndim != 1 or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            for value in raw
        ):
            raise ValueError(message)
        array = np.asarray(values, dtype="float64")
    except (TypeError, ValueError) as error:
        raise ValueError(message) from error
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(message)
    if probability and ((array < 0.0).any() or (array > 1.0).any()):
        raise ValueError(message)
    return array


def _risk_calibration_rows(
    y: np.ndarray,
    probabilities: np.ndarray,
    *,
    scope: str,
    fold_id: object,
    bins: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_total = len(y)
    if n_total:
        bin_ids = np.minimum((probabilities * bins).astype(int), bins - 1)
    else:
        bin_ids = np.asarray([], dtype=int)
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        n_rows = int(mask.sum())
        lower = bin_id / bins
        upper = (bin_id + 1) / bins
        if n_rows:
            mean_probability = float(np.mean(probabilities[mask]))
            event_rate = float(np.mean(y[mask]))
            gap = abs(event_rate - mean_probability)
            weighted_gap = float((n_rows / n_total) * gap)
            status, reason = "available", pd.NA
        else:
            mean_probability = event_rate = gap = weighted_gap = pd.NA
            status, reason = "undefined", "empty_bin"
        rows.append(
            {
                "scope": scope,
                "fold_id": fold_id,
                "bin_id": bin_id,
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "upper_inclusive": bin_id == bins - 1,
                "n_rows": n_rows,
                "mean_predicted_probability": mean_probability,
                "observed_event_rate": event_rate,
                "absolute_gap": gap,
                "weighted_gap": weighted_gap,
                "status": status,
                "reason": reason,
            }
        )
    return rows


def _risk_direct_metric_rows(
    y: np.ndarray,
    scores: np.ndarray,
    probabilities: np.ndarray | None,
    *,
    scope: str,
    fold_id: object,
    calibration_bins: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    n_rows = len(y)
    n_positive = int(y.sum()) if n_rows else 0
    n_negative = n_rows - n_positive
    values: dict[str, tuple[object, object, str, object]] = {}
    calibration_rows: list[dict[str, object]] = []
    if n_rows == 0:
        for metric in _RISK_METRIC_ORDER:
            values[metric] = (pd.NA, pd.NA, "undefined", "no_evaluable_rows")
        if probabilities is not None:
            calibration_rows = _risk_calibration_rows(
                y,
                probabilities,
                scope=scope,
                fold_id=fold_id,
                bins=calibration_bins,
            )
    else:
        single_class = n_positive == 0 or n_negative == 0
        if single_class:
            for metric in (
                "roc_auc",
                "average_precision",
                "normalized_gini",
                "ks_statistic",
            ):
                values[metric] = (pd.NA, pd.NA, "undefined", "single_class")
        else:
            auc = float(roc_auc_score(y, scores))
            ap = float(average_precision_score(y, scores))
            best_ks = -1.0
            best_threshold = float("-inf")
            for threshold in sorted(
                set(float(value) for value in scores), reverse=True
            ):
                selected = scores >= threshold
                tpr = float(np.sum(selected & (y == 1)) / n_positive)
                fpr = float(np.sum(selected & (y == 0)) / n_negative)
                ks = tpr - fpr
                if ks > best_ks:
                    best_ks = ks
                    best_threshold = threshold
            values["roc_auc"] = (auc, pd.NA, "available", pd.NA)
            values["average_precision"] = (ap, pd.NA, "available", pd.NA)
            values["normalized_gini"] = (2.0 * auc - 1.0, pd.NA, "available", pd.NA)
            values["ks_statistic"] = (
                float(best_ks),
                float(best_threshold),
                "available",
                pd.NA,
            )
        if probabilities is None:
            for metric in (
                "brier_score",
                "log_loss",
                "expected_calibration_error",
            ):
                values[metric] = (pd.NA, pd.NA, "unavailable", "probability_absent")
        else:
            clipped = np.clip(probabilities.astype("float64"), 1e-15, 1.0 - 1e-15)
            brier = float(np.mean((probabilities - y) ** 2))
            log_loss = float(
                np.mean(-(y * np.log(clipped) + (1 - y) * np.log(1.0 - clipped)))
            )
            calibration_rows = _risk_calibration_rows(
                y,
                probabilities,
                scope=scope,
                fold_id=fold_id,
                bins=calibration_bins,
            )
            ece = float(
                sum(
                    float(row["weighted_gap"])
                    for row in calibration_rows
                    if row["status"] == "available"
                )
            )
            values["brier_score"] = (brier, pd.NA, "available", pd.NA)
            values["log_loss"] = (log_loss, pd.NA, "available", pd.NA)
            values["expected_calibration_error"] = (
                ece,
                pd.NA,
                "available",
                pd.NA,
            )
    rows: list[dict[str, object]] = []
    for metric in _RISK_METRIC_ORDER:
        value, at_threshold, status, reason = values[metric]
        rows.append(
            {
                "scope": scope,
                "fold_id": fold_id,
                "metric": metric,
                "statistic": "direct",
                "value": value,
                "at_threshold": at_threshold,
                "status": status,
                "reason": reason,
                "n_rows": n_rows,
                "n_positive": n_positive,
                "n_negative": n_negative,
            }
        )
    return rows, calibration_rows


def _risk_gains_rows(
    y: np.ndarray,
    scores: np.ndarray,
    *,
    scope: str,
    fold_id: object,
    fractions: tuple[float, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_rows = len(y)
    total_positive = int(y.sum()) if n_rows else 0
    for fraction in fractions:
        if n_rows == 0:
            rows.append(
                {
                    "scope": scope,
                    "fold_id": fold_id,
                    "requested_fraction": fraction,
                    "target_count": 0,
                    "boundary_score": pd.NA,
                    "selected_n": 0,
                    "actual_fraction": pd.NA,
                    "total_positive_n": 0,
                    "selected_positive_n": 0,
                    "event_rate": pd.NA,
                    "event_rate_status": "undefined",
                    "event_rate_reason": "no_evaluable_rows",
                    "capture": pd.NA,
                    "capture_status": "undefined",
                    "capture_reason": "no_evaluable_rows",
                    "lift": pd.NA,
                    "lift_status": "undefined",
                    "lift_reason": "no_evaluable_rows",
                }
            )
            continue
        target_count = max(1, int(np.ceil(fraction * n_rows)))
        boundary = float(np.sort(scores)[::-1][target_count - 1])
        selected = scores >= boundary
        selected_n = int(selected.sum())
        selected_positive = int(y[selected].sum())
        actual_fraction = selected_n / n_rows
        event_rate = selected_positive / selected_n
        if total_positive:
            capture = selected_positive / total_positive
            lift = capture / actual_fraction
            capture_status = lift_status = "available"
            capture_reason = lift_reason = pd.NA
        else:
            capture = lift = pd.NA
            capture_status = lift_status = "undefined"
            capture_reason = lift_reason = "zero_denominator"
        rows.append(
            {
                "scope": scope,
                "fold_id": fold_id,
                "requested_fraction": fraction,
                "target_count": target_count,
                "boundary_score": boundary,
                "selected_n": selected_n,
                "actual_fraction": actual_fraction,
                "total_positive_n": total_positive,
                "selected_positive_n": selected_positive,
                "event_rate": event_rate,
                "event_rate_status": "available",
                "event_rate_reason": pd.NA,
                "capture": capture,
                "capture_status": capture_status,
                "capture_reason": capture_reason,
                "lift": lift,
                "lift_status": lift_status,
                "lift_reason": lift_reason,
            }
        )
    return rows


def _risk_rate(value: int, denominator: int) -> tuple[object, str, object]:
    if denominator == 0:
        return pd.NA, "undefined", "zero_denominator"
    return float(value / denominator), "available", pd.NA


def _risk_threshold_rows(
    y: np.ndarray,
    values: np.ndarray,
    *,
    scope: str,
    fold_id: object,
    threshold_kind: str,
    thresholds: tuple[float, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_rows = len(y)
    for threshold in thresholds:
        selected = values >= threshold
        tp = int(np.sum(selected & (y == 1)))
        fp = int(np.sum(selected & (y == 0)))
        tn = int(np.sum(~selected & (y == 0)))
        fn = int(np.sum(~selected & (y == 1)))
        rate_values = {
            "sensitivity": _risk_rate(tp, tp + fn),
            "specificity": _risk_rate(tn, tn + fp),
            "precision": _risk_rate(tp, tp + fp),
            "negative_predictive_value": _risk_rate(tn, tn + fn),
            "f1": _risk_rate(2 * tp, 2 * tp + fp + fn),
            "accuracy": _risk_rate(tp + tn, n_rows),
            "predicted_positive_rate": _risk_rate(tp + fp, n_rows),
        }
        row: dict[str, object] = {
            "scope": scope,
            "fold_id": fold_id,
            "threshold_kind": threshold_kind,
            "threshold": threshold,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        for name, (value, status, reason) in rate_values.items():
            row[name] = value
            row[f"{name}_status"] = status
            row[f"{name}_reason"] = reason
        rows.append(row)
    return rows


def _binary_risk_evaluation(
    *,
    target_values: np.ndarray,
    evaluable: np.ndarray,
    ranking_scores: object,
    event_probabilities: object | None,
    fold_ids: np.ndarray,
    fold_order: tuple[int, ...],
    calibration_bins: int,
    gain_fractions: tuple[float, ...],
    thresholds: tuple[float, ...],
    threshold_kind: str | None,
) -> _BinaryRiskEvaluationTables:
    """Compute all Task 15 statistical tables from aligned frozen arrays."""
    scores = _risk_float_array(ranking_scores, probability=False)
    probabilities = (
        None
        if event_probabilities is None
        else _risk_float_array(event_probabilities, probability=True)
    )
    y_all = np.asarray(target_values, dtype="int8")
    evaluable_array = np.asarray(evaluable, dtype=bool)
    fold_array = np.asarray(fold_ids, dtype=int)
    n = len(scores)
    if (
        y_all.ndim != 1
        or evaluable_array.ndim != 1
        or fold_array.ndim != 1
        or len(y_all) != n
        or len(evaluable_array) != n
        or len(fold_array) != n
        or (probabilities is not None and len(probabilities) != n)
        or not np.isin(y_all[evaluable_array], [0, 1]).all()
    ):
        raise ValueError("binary risk validation result has invalid schema")

    metric_rows: list[dict[str, object]] = []
    gain_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    direct_by_fold: dict[int, list[dict[str, object]]] = {}
    scopes: list[tuple[str, object, np.ndarray]] = [
        ("fold", fold_id, (fold_array == fold_id) & evaluable_array)
        for fold_id in fold_order
    ]
    scopes.append(("overall", pd.NA, evaluable_array))
    for scope, fold_id, mask in scopes:
        y = y_all[mask]
        scope_scores = scores[mask]
        scope_probabilities = None if probabilities is None else probabilities[mask]
        rows, bins = _risk_direct_metric_rows(
            y,
            scope_scores,
            scope_probabilities,
            scope=scope,
            fold_id=fold_id,
            calibration_bins=calibration_bins,
        )
        metric_rows.extend(rows)
        calibration_rows.extend(bins)
        gain_rows.extend(
            _risk_gains_rows(
                y,
                scope_scores,
                scope=scope,
                fold_id=fold_id,
                fractions=gain_fractions,
            )
        )
        if thresholds:
            threshold_values = (
                scope_probabilities
                if threshold_kind == "event_probability"
                else scope_scores
            )
            if threshold_values is None:
                raise ValueError(
                    "event-probability thresholds require event probabilities"
                )
            threshold_rows.extend(
                _risk_threshold_rows(
                    y,
                    threshold_values,
                    scope=scope,
                    fold_id=fold_id,
                    threshold_kind=str(threshold_kind),
                    thresholds=thresholds,
                )
            )
        if scope == "fold":
            direct_by_fold[int(fold_id)] = rows

    for metric in _RISK_METRIC_ORDER:
        available = [
            row
            for fold_id in fold_order
            for row in direct_by_fold[fold_id]
            if row["metric"] == metric and row["status"] == "available"
        ]
        support_rows = [
            row
            for fold_id in fold_order
            for row in direct_by_fold[fold_id]
            if row["metric"] == metric
        ]
        n_rows = sum(int(row["n_rows"]) for row in support_rows)
        n_positive = sum(int(row["n_positive"]) for row in support_rows)
        n_negative = sum(int(row["n_negative"]) for row in support_rows)
        if available:
            mean_value: object = float(
                np.mean([float(row["value"]) for row in available])
            )
            mean_status, mean_reason = "available", pd.NA
        else:
            mean_value = pd.NA
            mean_status, mean_reason = "unavailable", "no_available_folds"
        metric_rows.append(
            {
                "scope": "fold_summary",
                "fold_id": pd.NA,
                "metric": metric,
                "statistic": "mean",
                "value": mean_value,
                "at_threshold": pd.NA,
                "status": mean_status,
                "reason": mean_reason,
                "n_rows": n_rows,
                "n_positive": n_positive,
                "n_negative": n_negative,
            }
        )
        if len(available) >= 2:
            std_value: object = float(
                np.std([float(row["value"]) for row in available], ddof=1)
            )
            std_status, std_reason = "available", pd.NA
        else:
            std_value = pd.NA
            std_status, std_reason = "undefined", "insufficient_available_folds"
        metric_rows.append(
            {
                "scope": "fold_summary",
                "fold_id": pd.NA,
                "metric": metric,
                "statistic": "sample_std",
                "value": std_value,
                "at_threshold": pd.NA,
                "status": std_status,
                "reason": std_reason,
                "n_rows": n_rows,
                "n_positive": n_positive,
                "n_negative": n_negative,
            }
        )

    return _BinaryRiskEvaluationTables(
        metrics=_risk_frame(
            metric_rows,
            _RISK_METRIC_COLUMNS,
            integers=("fold_id", "n_rows", "n_positive", "n_negative"),
            floats=("value", "at_threshold"),
        ),
        gains=_risk_frame(
            gain_rows,
            _RISK_GAINS_COLUMNS,
            integers=(
                "fold_id",
                "target_count",
                "selected_n",
                "total_positive_n",
                "selected_positive_n",
            ),
            floats=(
                "requested_fraction",
                "boundary_score",
                "actual_fraction",
                "event_rate",
                "capture",
                "lift",
            ),
        ),
        calibration=_risk_frame(
            calibration_rows,
            _RISK_CALIBRATION_COLUMNS,
            integers=("fold_id", "bin_id", "n_rows"),
            floats=(
                "lower_bound",
                "upper_bound",
                "mean_predicted_probability",
                "observed_event_rate",
                "absolute_gap",
                "weighted_gap",
            ),
            booleans=("upper_inclusive",),
        ),
        threshold_analysis=_risk_frame(
            threshold_rows,
            _RISK_THRESHOLD_COLUMNS,
            integers=("fold_id", "tp", "fp", "tn", "fn"),
            floats=(
                "threshold",
                "sensitivity",
                "specificity",
                "precision",
                "negative_predictive_value",
                "f1",
                "accuracy",
                "predicted_positive_rate",
            ),
        ),
    )


def _binary_risk_business_primitive(
    *,
    y: np.ndarray,
    evaluable: np.ndarray,
    selected: np.ndarray,
    probabilities: np.ndarray | None,
    exposure: np.ndarray | None,
    observed_loss: np.ndarray | None,
    observed_loss_mature: np.ndarray,
    loss_fraction: np.ndarray | None,
) -> dict[str, tuple[object, str, object]]:
    """Compute Task 15 action-free event/exposure/loss primitives."""
    selected = np.asarray(selected, dtype=bool)
    evaluable_selected = selected & np.asarray(evaluable, dtype=bool)
    n_evaluable = int(evaluable_selected.sum())
    if n_evaluable:
        event_rate: tuple[object, str, object] = (
            float(np.mean(np.asarray(y)[evaluable_selected])),
            "available",
            pd.NA,
        )
    else:
        event_rate = (pd.NA, "undefined", "no_evaluable_rows")
    exposure_sum = (
        (pd.NA, "unavailable", "exposure_absent")
        if exposure is None
        else (float(np.sum(exposure[selected])), "available", pd.NA)
    )
    mature_selected = selected & np.asarray(observed_loss_mature, dtype=bool)
    if observed_loss is None:
        observed_loss_sum = (pd.NA, "unavailable", "observed_loss_absent")
    elif mature_selected.any():
        observed_loss_sum = (
            float(np.sum(observed_loss[mature_selected])),
            "available",
            pd.NA,
        )
    else:
        observed_loss_sum = (pd.NA, "undefined", "no_evaluable_rows")
    if probabilities is None:
        expected_loss_sum = (pd.NA, "unavailable", "probability_absent")
    elif exposure is None:
        expected_loss_sum = (pd.NA, "unavailable", "exposure_absent")
    elif loss_fraction is None:
        expected_loss_sum = (pd.NA, "unavailable", "loss_fraction_absent")
    else:
        expected_loss_sum = (
            float(
                np.sum(
                    probabilities[selected]
                    * exposure[selected]
                    * loss_fraction[selected]
                )
            ),
            "available",
            pd.NA,
        )
    return {
        "event_rate": event_rate,
        "exposure_sum": exposure_sum,
        "observed_loss_sum": observed_loss_sum,
        "expected_loss_sum": expected_loss_sum,
    }
