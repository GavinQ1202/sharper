"""Task 11 classification evaluation contracts."""
# ruff: noqa: E501

import math
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


def test_task15_public_orchestration_uses_the_single_evaluation_implementation() -> (
    None
):
    from sklearn.model_selection import StratifiedKFold

    from sharper import (
        BinaryRiskValidationConfig,
        ExternalRiskPredictions,
        validate_binary_risk,
    )

    target = np.asarray([0, 1] * 6, dtype="int8")
    scores = np.linspace(-0.8, 0.9, len(target), dtype="float64")
    probabilities = np.linspace(0.05, 0.95, len(target), dtype="float64")
    fold_ids = np.empty(len(target), dtype=int)
    fit_rows = []
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for fold_id, (train, validation) in enumerate(
        splitter.split(np.arange(len(target)), target)
    ):
        fit_rows.append((fold_id, tuple(sorted(int(value) for value in train))))
        fold_ids[validation] = fold_id
    config = BinaryRiskValidationConfig(
        validation_mode="stratified_kfold",
        n_splits=3,
        thresholds=(-0.2, 0.4),
        threshold_kind="ranking_score",
        calibration_bins=3,
        gain_fractions=(0.25, 0.5, 1.0),
    )
    result = validate_binary_risk(
        pd.DataFrame({"target": target}),
        "target",
        config=config,
        external_predictions=ExternalRiskPredictions(
            row_positions=tuple(range(len(target))),
            fold_ids=tuple(int(value) for value in fold_ids),
            fold_fit_row_positions=tuple(fit_rows),
            ranking_scores=tuple(float(value) for value in scores),
            ranking_direction="higher_risk",
            event_probabilities=tuple(float(value) for value in probabilities),
            probability_positive_label=1,
            probability_provenance="external_declared",
        ),
    )
    expected = evaluation._binary_risk_evaluation(
        target_values=target,
        evaluable=np.ones(len(target), dtype=bool),
        ranking_scores=scores,
        event_probabilities=probabilities,
        fold_ids=fold_ids,
        fold_order=(0, 1, 2),
        calibration_bins=3,
        gain_fractions=(0.25, 0.5, 1.0),
        thresholds=(-0.2, 0.4),
        threshold_kind="ranking_score",
    )
    pd.testing.assert_frame_equal(result.metrics, expected.metrics)
    pd.testing.assert_frame_equal(result.gains, expected.gains)
    pd.testing.assert_frame_equal(result.calibration, expected.calibration)
    pd.testing.assert_frame_equal(
        result.threshold_analysis, expected.threshold_analysis
    )


def test_task15_metrics_gains_calibration_and_thresholds_use_hand_values() -> None:
    target = np.asarray([0, 1, 0, 1], dtype="int8")
    scores = np.asarray([0.1, 0.4, 0.4, 0.9], dtype="float64")
    probabilities = np.asarray([0.1, 0.7, 0.2, 0.8], dtype="float64")
    tables = evaluation._binary_risk_evaluation(
        target_values=target,
        evaluable=np.ones(4, dtype=bool),
        ranking_scores=scores,
        event_probabilities=probabilities,
        fold_ids=np.asarray([0, 0, 1, 1]),
        fold_order=(0, 1),
        calibration_bins=2,
        gain_fractions=(0.5, 1.0),
        thresholds=(0.4,),
        threshold_kind="ranking_score",
    )
    overall = tables.metrics.loc[tables.metrics["scope"] == "overall"].set_index(
        "metric"
    )
    expected_log_loss = (
        -(math.log(0.9) + math.log(0.7) + math.log(0.8) + math.log(0.8)) / 4.0
    )
    expected_metrics = {
        "roc_auc": 0.875,
        "average_precision": 5.0 / 6.0,
        "normalized_gini": 0.75,
        "ks_statistic": 0.5,
        "brier_score": 0.045,
        "log_loss": expected_log_loss,
        "expected_calibration_error": 0.2,
    }
    for metric, expected_value in expected_metrics.items():
        assert overall.loc[metric, "status"] == "available"
        assert overall.loc[metric, "value"] == pytest.approx(expected_value, abs=1e-12)
    assert overall.loc["ks_statistic", "at_threshold"] == pytest.approx(0.9)

    calibration = tables.calibration.loc[
        (tables.calibration["scope"] == "overall")
        & (tables.calibration["status"] == "available")
    ]
    assert calibration["bin_id"].tolist() == [0, 1]
    assert calibration["n_rows"].tolist() == [2, 2]
    assert calibration["mean_predicted_probability"].tolist() == pytest.approx(
        [0.15, 0.75]
    )
    assert calibration["observed_event_rate"].tolist() == pytest.approx([0.0, 1.0])
    assert calibration["weighted_gap"].tolist() == pytest.approx([0.075, 0.125])

    gain = tables.gains.loc[
        (tables.gains["scope"] == "overall")
        & (tables.gains["requested_fraction"] == 0.5)
    ].iloc[0]
    assert (gain["target_count"], gain["selected_n"], gain["selected_positive_n"]) == (
        2,
        3,
        2,
    )
    assert gain["actual_fraction"] == pytest.approx(0.75)
    assert gain["event_rate"] == pytest.approx(2.0 / 3.0)
    assert gain["capture"] == pytest.approx(1.0)
    assert gain["lift"] == pytest.approx(4.0 / 3.0)

    threshold = tables.threshold_analysis.loc[
        tables.threshold_analysis["scope"] == "overall"
    ].iloc[0]
    assert (threshold["tp"], threshold["fp"], threshold["tn"], threshold["fn"]) == (
        2,
        1,
        1,
        0,
    )
    expected_rates = {
        "sensitivity": 1.0,
        "specificity": 0.5,
        "precision": 2.0 / 3.0,
        "negative_predictive_value": 1.0,
        "f1": 0.8,
        "accuracy": 0.75,
        "predicted_positive_rate": 0.75,
    }
    for metric, expected_value in expected_rates.items():
        assert threshold[f"{metric}_status"] == "available"
        assert threshold[metric] == pytest.approx(expected_value)

    fold_mean = tables.metrics.loc[
        (tables.metrics["scope"] == "fold_summary")
        & (tables.metrics["metric"] == "roc_auc")
        & (tables.metrics["statistic"] == "mean")
    ].iloc[0]
    assert fold_mean["value"] == pytest.approx(1.0)
    assert overall.loc["roc_auc", "value"] == pytest.approx(0.875)


def test_task15_perfect_calibration_includes_probability_endpoints() -> None:
    target = np.asarray([0, 0, 1, 1], dtype="int8")
    probabilities = np.asarray([0.0, 0.0, 1.0, 1.0], dtype="float64")
    tables = evaluation._binary_risk_evaluation(
        target_values=target,
        evaluable=np.ones(4, dtype=bool),
        ranking_scores=probabilities,
        event_probabilities=probabilities,
        fold_ids=np.zeros(4, dtype=int),
        fold_order=(0,),
        calibration_bins=2,
        gain_fractions=(1.0,),
        thresholds=(),
        threshold_kind=None,
    )

    bins = tables.calibration.loc[tables.calibration["scope"] == "overall"]
    assert bins["bin_id"].tolist() == [0, 1]
    assert bins["n_rows"].tolist() == [2, 2]
    assert bins["mean_predicted_probability"].tolist() == pytest.approx([0.0, 1.0])
    assert bins["observed_event_rate"].tolist() == pytest.approx([0.0, 1.0])
    assert bins["absolute_gap"].tolist() == pytest.approx([0.0, 0.0])
    assert bins["upper_inclusive"].tolist() == [False, True]
    ece = tables.metrics.loc[
        (tables.metrics["scope"] == "overall")
        & (tables.metrics["metric"] == "expected_calibration_error")
    ].iloc[0]
    assert ece["value"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("probability", "available_bin"),
    [(0.8, 1), (0.2, 0)],
)
def test_task15_over_and_under_prediction_calibration_are_hand_calculated(
    probability: float, available_bin: int
) -> None:
    target = np.asarray([0, 1], dtype="int8")
    probabilities = np.full(2, probability, dtype="float64")
    tables = evaluation._binary_risk_evaluation(
        target_values=target,
        evaluable=np.ones(2, dtype=bool),
        ranking_scores=probabilities,
        event_probabilities=probabilities,
        fold_ids=np.zeros(2, dtype=int),
        fold_order=(0,),
        calibration_bins=2,
        gain_fractions=(1.0,),
        thresholds=(),
        threshold_kind=None,
    )

    bins = tables.calibration.loc[tables.calibration["scope"] == "overall"].set_index(
        "bin_id"
    )
    row = bins.loc[available_bin]
    assert row["n_rows"] == 2
    assert row["mean_predicted_probability"] == pytest.approx(probability)
    assert row["observed_event_rate"] == pytest.approx(0.5)
    assert row["absolute_gap"] == pytest.approx(0.3)
    assert row["weighted_gap"] == pytest.approx(0.3)
    empty = bins.loc[1 - available_bin]
    assert empty["n_rows"] == 0
    assert pd.isna(empty["mean_predicted_probability"])
    assert (empty["status"], empty["reason"]) == ("undefined", "empty_bin")
    ece = tables.metrics.loc[
        (tables.metrics["scope"] == "overall")
        & (tables.metrics["metric"] == "expected_calibration_error")
    ].iloc[0]
    assert ece["value"] == pytest.approx(0.3)


def test_task15_ranking_only_has_zero_calibration_rows_and_unavailable_metrics() -> (
    None
):
    tables = evaluation._binary_risk_evaluation(
        target_values=np.asarray([0, 1, 0, 1], dtype="int8"),
        evaluable=np.ones(4, dtype=bool),
        ranking_scores=np.asarray([-1.0, 0.2, 0.1, 1.0]),
        event_probabilities=None,
        fold_ids=np.zeros(4, dtype=int),
        fold_order=(0,),
        calibration_bins=2,
        gain_fractions=(1.0,),
        thresholds=(),
        threshold_kind=None,
    )

    assert tables.calibration.empty
    overall = tables.metrics.loc[tables.metrics["scope"] == "overall"].set_index(
        "metric"
    )
    for metric in ("brier_score", "log_loss", "expected_calibration_error"):
        assert (overall.loc[metric, "status"], overall.loc[metric, "reason"]) == (
            "unavailable",
            "probability_absent",
        )


def test_task15_equal_score_ties_and_single_class_statuses_are_independent() -> None:
    tied = evaluation._binary_risk_evaluation(
        target_values=np.asarray([0, 1, 0, 1], dtype="int8"),
        evaluable=np.ones(4, dtype=bool),
        ranking_scores=np.full(4, 0.5),
        event_probabilities=np.full(4, 0.5),
        fold_ids=np.zeros(4, dtype=int),
        fold_order=(0,),
        calibration_bins=2,
        gain_fractions=(0.25, 1.0),
        thresholds=(0.5,),
        threshold_kind="ranking_score",
    )
    tied_metrics = tied.metrics.loc[tied.metrics["scope"] == "overall"].set_index(
        "metric"
    )
    assert tied_metrics.loc["roc_auc", "value"] == pytest.approx(0.5)
    assert tied_metrics.loc["average_precision", "value"] == pytest.approx(0.5)
    assert tied_metrics.loc["normalized_gini", "value"] == pytest.approx(0.0)
    assert tied_metrics.loc["ks_statistic", "value"] == pytest.approx(0.0)
    assert tied_metrics.loc["ks_statistic", "at_threshold"] == pytest.approx(0.5)
    tied_gain = tied.gains.loc[
        (tied.gains["scope"] == "overall") & (tied.gains["requested_fraction"] == 0.25)
    ].iloc[0]
    assert tied_gain["target_count"] == 1
    assert tied_gain["selected_n"] == 4
    assert tied_gain["capture"] == pytest.approx(1.0)
    assert tied_gain["lift"] == pytest.approx(1.0)

    single = evaluation._binary_risk_evaluation(
        target_values=np.zeros(3, dtype="int8"),
        evaluable=np.ones(3, dtype=bool),
        ranking_scores=np.asarray([0.1, 0.2, 0.3]),
        event_probabilities=np.asarray([0.1, 0.2, 0.3]),
        fold_ids=np.zeros(3, dtype=int),
        fold_order=(0,),
        calibration_bins=2,
        gain_fractions=(1.0,),
        thresholds=(0.2,),
        threshold_kind="event_probability",
    )
    metrics = single.metrics.loc[single.metrics["scope"] == "overall"].set_index(
        "metric"
    )
    for metric in ("roc_auc", "average_precision", "normalized_gini", "ks_statistic"):
        assert (metrics.loc[metric, "status"], metrics.loc[metric, "reason"]) == (
            "undefined",
            "single_class",
        )
    for metric in ("brier_score", "log_loss", "expected_calibration_error"):
        assert metrics.loc[metric, "status"] == "available"
    gain = single.gains.loc[single.gains["scope"] == "overall"].iloc[0]
    assert (gain["event_rate"], gain["event_rate_status"]) == (0.0, "available")
    assert (gain["capture_status"], gain["capture_reason"]) == (
        "undefined",
        "zero_denominator",
    )
    assert (gain["lift_status"], gain["lift_reason"]) == (
        "undefined",
        "zero_denominator",
    )
    threshold = single.threshold_analysis.loc[
        single.threshold_analysis["scope"] == "overall"
    ].iloc[0]
    assert threshold["sensitivity_status"] == "undefined"
    assert threshold["sensitivity_reason"] == "zero_denominator"
    assert threshold["specificity"] == pytest.approx(1.0 / 3.0)
    assert threshold["predicted_positive_rate"] == pytest.approx(2.0 / 3.0)


def test_task15_fixed_epsilon_boundaries_and_raw_range_validation() -> None:
    epsilon = 1e-15
    probabilities = np.asarray(
        [0.0, epsilon / 2.0, epsilon, 1.0 - epsilon, 1.0 - epsilon / 2.0, 1.0]
    )
    target = np.asarray([0, 1, 0, 1, 0, 1], dtype="int8")
    tables = evaluation._binary_risk_evaluation(
        target_values=target,
        evaluable=np.ones(6, dtype=bool),
        ranking_scores=probabilities,
        event_probabilities=probabilities,
        fold_ids=np.zeros(6, dtype=int),
        fold_order=(0,),
        calibration_bins=2,
        gain_fractions=(1.0,),
        thresholds=(),
        threshold_kind=None,
    )
    terms = []
    for label, raw in zip(target, probabilities, strict=True):
        clipped = min(max(float(raw), epsilon), 1.0 - epsilon)
        terms.append(
            -(
                int(label) * math.log(clipped)
                + (1 - int(label)) * math.log(1.0 - clipped)
            )
        )
    expected = sum(terms) / 6.0
    row = tables.metrics.loc[
        (tables.metrics["scope"] == "overall")
        & (tables.metrics["metric"] == "log_loss")
    ].iloc[0]
    assert row["value"] == pytest.approx(expected, abs=1e-12)

    for invalid in (-epsilon, 1.0 + epsilon):
        with pytest.raises(
            ValueError,
            match=r"^event probabilities must be finite values in \[0, 1\]$",
        ):
            evaluation._binary_risk_evaluation(
                target_values=np.asarray([0, 1], dtype="int8"),
                evaluable=np.ones(2, dtype=bool),
                ranking_scores=np.asarray([0.0, 1.0]),
                event_probabilities=np.asarray([0.0, invalid]),
                fold_ids=np.zeros(2, dtype=int),
                fold_order=(0,),
                calibration_bins=2,
                gain_fractions=(1.0,),
                thresholds=(),
                threshold_kind=None,
            )


def test_task15_observed_loss_zero_mature_rows_uses_approved_reason() -> None:
    common = {
        "y": np.asarray([0, 1], dtype="int8"),
        "evaluable": np.ones(2, dtype=bool),
        "selected": np.ones(2, dtype=bool),
        "probabilities": np.asarray([0.2, 0.8]),
        "exposure": np.asarray([100.0, 200.0]),
        "loss_fraction": np.asarray([0.5, 0.5]),
    }
    zero = evaluation._binary_risk_business_primitive(
        **common,
        observed_loss=np.asarray([np.nan, np.nan]),
        observed_loss_mature=np.zeros(2, dtype=bool),
    )
    assert zero["observed_loss_sum"][1:] == ("undefined", "no_evaluable_rows")
    assert zero["expected_loss_sum"] == (90.0, "available", pd.NA)

    mature = evaluation._binary_risk_business_primitive(
        **common,
        observed_loss=np.asarray([10.0, np.nan]),
        observed_loss_mature=np.asarray([True, False]),
    )
    assert mature["observed_loss_sum"] == (10.0, "available", pd.NA)
