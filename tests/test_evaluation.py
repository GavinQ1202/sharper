"""Task 11 classification evaluation contracts."""
# ruff: noqa: E501

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sharper import (
    evaluate_classifier,
    evaluate_model,
    evaluate_regressor,
    evaluation,
    train_classifier,
    train_regressor,
)
from sharper.visualization import plot_classification_evaluation


def _trained() -> object:
    frame = pd.DataFrame(
        {
            "x": list(range(20)),
            "category": ["a", "b"] * 10,
            "target": [0] * 10 + [1] * 10,
        }
    )
    return train_classifier(frame, "target", test_size=0.30, random_state=7)


def test_evaluation_matches_sklearn_and_evaluate_model_delegates() -> None:
    training = _trained()
    result = evaluate_classifier(training)
    delegated = evaluate_model(training)
    expected = (
        ("accuracy", accuracy_score(result.y_true, result.y_pred)),
        ("balanced_accuracy", balanced_accuracy_score(result.y_true, result.y_pred)),
        (
            "f1_macro",
            f1_score(
                result.y_true,
                result.y_pred,
                labels=list(result.classes),
                average="macro",
                zero_division=0,
            ),
        ),
    )
    assert result.metrics == tuple((name, float(value)) for name, value in expected)
    assert delegated.metrics == result.metrics
    assert result.score_kind == "predict_proba"
    assert result.positive_label == result.classes[1]
    assert result.roc_auc is not None


def test_malformed_training_and_evaluation_detail_are_rejected() -> None:
    training = _trained()
    with pytest.raises(ValueError, match="training result has invalid schema"):
        evaluate_classifier(
            replace(
                training,
                X_test=training.X_test.rename(
                    columns={training.feature_columns[0]: "bad"}
                ),
            )
        )

    evaluation = evaluate_classifier(training)
    broken = replace(
        evaluation, metrics=(("accuracy", np.nan), *evaluation.metrics[1:])
    )
    with pytest.raises(
        ValueError, match="classification evaluation result has invalid schema"
    ):
        plot_classification_evaluation(broken)


@pytest.mark.parametrize(
    "replacement",
    [
        {"metrics": (1, 1, 1)},
        {"metrics": (("accuracy", 1.0), ("f1_macro", 1.0), ("f1_macro", 1.0))},
        {
            "metrics": (
                ("accuracy", True),
                ("balanced_accuracy", 1.0),
                ("f1_macro", 1.0),
            )
        },
        {"positive_label": 1.5},
        {"positive_label": float("nan")},
        {"positive_label": (1,)},
        {"holdout_positions": (0, 0, 1, 2, 3, 4)},
    ],
)
def test_malformed_evaluation_members_use_one_stable_error(
    replacement: dict[str, object],
) -> None:
    result = evaluate_classifier(_trained())
    malformed = replace(result, **replacement)
    with pytest.raises(
        ValueError, match="^classification evaluation result has invalid schema$"
    ):
        plot_classification_evaluation(malformed)


def test_evaluation_tampering_matrix_uses_one_stable_error() -> None:
    result = evaluate_classifier(_trained())
    replacements = [
        {"metrics": (("accuracy", 1.0), ["balanced_accuracy", 1.0], ("f1_macro", 1.0))},
        {"metrics": (("accuracy", float("inf")), *result.metrics[1:])},
        {"confusion_matrix": ((0, 1),)},
        {"y_pred": (*result.y_pred[:-1], 99)},
        {"scores": result.scores[:-1]},
        {"scores": tuple(float("nan") for _ in result.scores)},
        {"roc_auc": 0.0},
        {"roc_curve": ((0.5, 0.5, 0.0), *result.roc_curve[1:])},
        {"classes": tuple(reversed(result.classes))},
        {"holdout_positions": result.holdout_positions[:-1]},
        {"score_kind": None},
    ]
    for replacement in replacements:
        with pytest.raises(
            ValueError, match="^classification evaluation result has invalid schema$"
        ):
            plot_classification_evaluation(replace(result, **replacement))


class _BaseClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def fit(self, X: object, y: list[int]) -> "_BaseClassifier":
        self.classes_ = [0, 1]
        return self

    def predict(self, X: object) -> np.ndarray:
        return np.zeros(len(X), dtype=int)  # type: ignore[arg-type]


class _FailPredictClassifier(_BaseClassifier):
    def predict(self, X: object) -> np.ndarray:
        raise RuntimeError("predict boom")


class _FailProbabilityClassifier(_BaseClassifier):
    def predict_proba(self, X: object) -> np.ndarray:
        raise RuntimeError("proba boom")


class _FailScoreClassifier(_BaseClassifier):
    def decision_function(self, X: object) -> np.ndarray:
        raise RuntimeError("score boom")


class _MalformedProbabilityClassifier(_BaseClassifier):
    def __init__(self, values: object) -> None:
        self.values = values

    def predict_proba(self, X: object) -> object:
        return self.values


@pytest.mark.parametrize(
    ("estimator", "message"),
    [
        (_FailPredictClassifier(), "classifier estimator prediction failed"),
        (
            _FailProbabilityClassifier(),
            "classifier estimator probability prediction failed",
        ),
        (_FailScoreClassifier(), "classifier estimator score prediction failed"),
    ],
)
def test_prediction_and_score_exceptions_preserve_causes(
    estimator: ClassifierMixin, message: str
) -> None:
    training = train_classifier(
        pd.DataFrame({"x": [0, 1] * 6, "target": [0] * 6 + [1] * 6}),
        "target",
        estimator=estimator,
        test_size=0.25,
    )
    with pytest.raises(ValueError, match=f"^{message}$") as error:
        evaluate_classifier(training)
    assert isinstance(error.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    "values",
    [
        np.array([0.5, 0.5]),
        np.array([[[0.5, 0.5]]]),
        np.array([[1.2, -0.2]] * 3),
        np.array([[0.6, 0.6]] * 3),
        np.array([["bad", "bad"]] * 3, dtype=object),
    ],
)
def test_malformed_probability_output_uses_stable_error(values: object) -> None:
    training = train_classifier(
        pd.DataFrame({"x": [0, 1] * 6, "target": [0] * 6 + [1] * 6}),
        "target",
        estimator=_MalformedProbabilityClassifier(values),
        test_size=0.25,
    )
    with pytest.raises(ValueError, match="^classifier estimator has invalid output$"):
        evaluate_classifier(training)


def test_evaluate_model_delegates_once_and_returns_the_spy_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = _trained()
    expected = evaluate_classifier(training)
    calls: list[object] = []

    def spy(result: object) -> object:
        calls.append(result)
        return expected

    monkeypatch.setattr(evaluation, "evaluate_classifier", spy)
    assert evaluate_model(training) is expected
    assert calls == [training]


def test_evaluation_does_not_mutate_training_holdout_snapshot() -> None:
    training = _trained()
    before = training.X_test.copy(deep=True)
    first = evaluate_classifier(training)
    second = evaluate_classifier(training)

    pd.testing.assert_frame_equal(training.X_test, before)
    assert first == second


def _regression_training(estimator: RegressorMixin | None = None) -> object:
    x = np.tile(np.arange(4, dtype=float), 6)
    frame = pd.DataFrame(
        {
            "x": x,
            "category": np.tile(["a", "b"], 12),
            "target": x * 2.0 + np.tile([0.0, 0.5], 12),
        }
    )
    return train_regressor(frame, "target", estimator=estimator, test_size=0.25)


class _RegressionOutput(RegressorMixin, BaseEstimator):
    _estimator_type = "regressor"

    def __init__(self, output: object | None = None) -> None:
        self.output = output

    def fit(self, X: object, y: list[float]) -> "_RegressionOutput":
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: object) -> object:
        if self.output is None:
            return np.full(len(X), self.mean_)  # type: ignore[arg-type]
        return self.output


class _FailPredictRegressor(_RegressionOutput):
    def predict(self, X: object) -> object:
        raise RuntimeError("predict boom")


def test_regression_evaluation_matches_sklearn_and_dispatches_once() -> None:
    training = _regression_training()
    result = evaluate_regressor(training)
    detail = result.predictions
    actual = detail["actual"].to_numpy()
    predicted = detail["predicted"].to_numpy()
    expected = (
        ("mae", float(mean_absolute_error(actual, predicted))),
        ("rmse", float(np.sqrt(mean_squared_error(actual, predicted)))),
        ("r2", float(r2_score(actual, predicted, force_finite=True))),
    )
    assert result.metrics == expected
    assert np.allclose(
        detail["residual"].to_numpy(), actual - predicted, rtol=0.0, atol=1e-12
    )

    calls: list[object] = []

    def spy(value: object) -> object:
        calls.append(value)
        return result

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(evaluation, "evaluate_regressor", spy)
    try:
        assert evaluate_model(training) is result
    finally:
        monkeypatch.undo()
    assert calls == [training]


@pytest.mark.parametrize(
    "output",
    [
        True,
        np.array(True),
        np.array([True] * 6),
        [True, 1, 1, 1, 1, 1],
        np.array(["1"] * 6),
        np.array(["1"] * 6, dtype=object),
        [np.array([1.0])] * 6,
        np.full((6, 1), 1.0),
        np.full(5, 1.0),
        np.array([np.nan] * 6),
        np.array([np.inf] * 6),
    ],
)
def test_regression_prediction_output_contract(output: object) -> None:
    training = _regression_training(_RegressionOutput(output))
    with pytest.raises(ValueError, match="^regressor estimator has invalid output$"):
        evaluate_regressor(training)


def test_regression_prediction_exception_keeps_cause() -> None:
    training = _regression_training(_FailPredictRegressor())
    with pytest.raises(
        ValueError, match="^regressor estimator prediction failed$"
    ) as error:
        evaluate_regressor(training)
    assert isinstance(error.value.__cause__, RuntimeError)


def test_regression_constant_holdout_r2_and_tampering_validation() -> None:
    training = _regression_training(_RegressionOutput())
    constant = replace(training, y_test=tuple(2.0 for _ in training.y_test))
    result = evaluate_regressor(constant)
    assert np.isfinite(dict(result.metrics)["r2"])
    assert result.limitations == ()

    with pytest.raises(
        ValueError, match="^regression evaluation result has invalid schema$"
    ):
        evaluation._validate_regression_evaluation(
            replace(result, metrics=(("mae", 0.0), ("rmse", 0.0), ("r2", 0.0)))
        )
    broken = result.predictions.copy(deep=True)
    broken.loc[0, "residual"] = 999.0
    with pytest.raises(
        ValueError, match="^regression evaluation result has invalid schema$"
    ):
        evaluation._validate_regression_evaluation(replace(result, predictions=broken))


def test_regression_negative_r2_and_constant_holdout_are_exact_and_deterministic() -> (
    None
):
    negative = evaluate_regressor(
        _regression_training(_RegressionOutput(np.full(6, 999.0)))
    )
    actual = negative.predictions["actual"].to_numpy()
    predicted = negative.predictions["predicted"].to_numpy()
    assert dict(negative.metrics)["r2"] == float(
        r2_score(actual, predicted, force_finite=True)
    )
    assert dict(negative.metrics)["r2"] < 0.0
    assert negative.limitations == ()

    training = _regression_training(_RegressionOutput())
    constant = replace(training, y_test=tuple(2.0 for _ in training.y_test))
    first = evaluate_regressor(constant)
    second = evaluate_regressor(constant)
    assert first.metrics == second.metrics
    assert dict(first.metrics)["r2"] == float(
        r2_score(
            first.predictions["actual"],
            first.predictions["predicted"],
            force_finite=True,
        )
    )


def test_regression_evaluation_training_result_tampering_is_stable() -> None:
    training = _regression_training()
    replacements = [
        {"task": "classification"},
        {"feature_columns": ("x", "x")},
        {"train_row_positions": (0, 0, *training.train_row_positions[2:])},
        {
            "test_row_positions": (training.test_row_positions[0],)
            * len(training.test_row_positions)
        },
        {"X_test": training.X_test.rename(columns={"x": "bad"})},
        {"X_test": training.X_test.iloc[:-1]},
        {"y_test": training.y_test[:-1]},
        {"y_test": (float("inf"), *training.y_test[1:])},
        {"excluded_columns": (training.feature_columns[0],)},
        {"random_state": -1},
        {"limitations": ("unknown",)},
    ]
    for replacement in replacements:
        with pytest.raises(
            ValueError, match="^regression training result has invalid schema$"
        ):
            evaluate_regressor(replace(training, **replacement))
